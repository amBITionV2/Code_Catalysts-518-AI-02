"""
KCET Learning-to-Rank (LambdaMART) — Training + Inference

Requirements:
    pip install pandas numpy scikit-learn lightgbm pyarrow

Usage:
    # Train (writes artifacts to ./artifacts/)
    python kcet_l2r_train_infer.py train --data KCET_cleaned_with_2025.csv

    # Recommend (after training)
    python kcet_l2r_train_infer.py recommend --rank 42000 --category GM --location "tumkuru" --topn 10

Artifacts:
    artifacts/
        items.parquet        # item-level aggregates per (College, Branch, Category)
        model_l2r.txt        # trained LightGBM ranker
        feature_meta.json    # features + metrics
        cutoff_stats.json    # quick stats
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else None

try:
    import lightgbm as lgb
except Exception as e:
    raise SystemExit(
        "LightGBM is required. Install via: pip install lightgbm"
    ) from e


# -----------------------------
# Utilities
# -----------------------------

def parse_round(x):
    import re
    m = re.findall(r"\d+", str(x))
    return int(m[0]) if m else np.nan

def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v_sorted = np.array(values)[order]
    w_sorted = np.array(weights)[order]
    cum_w = np.cumsum(w_sorted)
    cutoff = 0.5 * np.sum(w_sorted)
    return float(v_sorted[np.searchsorted(cum_w, cutoff)])

def recent_weight(year: int) -> float:
    # Emphasize recent years (cap to avoid extremes)
    return float(np.clip(1 + 0.5*(year - 2020), 0.5, 4.0))

def robust_city_filter(series: pd.Series, q: Optional[str]) -> pd.Series:
    if not q:
        return pd.Series([True] * len(series), index=series.index)
    q = str(q).casefold()
    return series.astype(str).str.casefold().str.contains(q, na=False)


# -----------------------------
# Build item aggregates (per College, Branch, Category)
# -----------------------------

def build_item_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Clean text
    for c in ["College", "Branch", "Category"]:
        df[c] = df[c].astype(str).str.strip()

    # Year, Round, Closing Rank
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Round"] = df["Round"].apply(parse_round)
    df["Round"] = pd.to_numeric(df["Round"], errors="coerce").astype("Int64")
    df["Closing Rank"] = pd.to_numeric(df["Closing Rank"], errors="coerce")

    # Drop rows without essentials
    df = df.dropna(subset=["Year", "Closing Rank"])
    df["Year"] = df["Year"].astype(int)
    df["Round"] = df["Round"].fillna(3).astype(int)

    # Recency weight
    df["_w"] = df["Year"].apply(recent_weight)

    # Per-year medians to compute simple trend
    gb_cols = ["College", "Branch", "Category", "Year"]
    per_year = (
        df.groupby(gb_cols)["Closing Rank"]
          .agg(med_close_y="median", mean="mean", min="min", max="max", count="count")
          .reset_index()
    )

    # Aggregate per (C,B,Cat)
    agg_rows = []
    for (college, branch, cat), g in df.groupby(["College", "Branch", "Category"]):
        vals = g["Closing Rank"].values
        w = g["_w"].values
        base_cutoff = weighted_median(vals, w)
        recent_year = int(g["Year"].max())
        samples = int(len(g))
        mn = float(np.min(vals))
        mx = float(np.max(vals))
        mean = float(np.mean(vals))

        gy = per_year[(per_year["College"]==college) &
                      (per_year["Branch"]==branch) &
                      (per_year["Category"]==cat)]
        slope = 0.0
        if len(gy) >= 2:
            x = gy["Year"].values.astype(float)
            y = gy["med_close_y"].values.astype(float)
            x_mean, y_mean = x.mean(), y.mean()
            denom = np.sum((x - x_mean)**2)
            if denom > 0:
                slope = float(np.sum((x - x_mean)*(y - y_mean)) / denom)

        last_year_close = float(
            df[(df["College"]==college) & (df["Branch"]==branch) & (df["Category"]==cat) & (df["Year"]==recent_year)]
            ["Closing Rank"].median()
        )

        agg_rows.append({
            "College": college,
            "Branch": branch,
            "Category": cat,
            "base_cutoff": float(base_cutoff),
            "mean_close": mean,
            "min_close": mn,
            "max_close": mx,
            "samples": samples,
            "recent_year": recent_year,
            "trend_slope": slope,
            "last_year_close": last_year_close
        })

    items = pd.DataFrame(agg_rows)

    # Backoff/shrink for sparse items (empirical Bayes)
    bc = (
        items.groupby(["Branch", "Category"])["base_cutoff"]
             .median()
             .rename("bc_median")
             .reset_index()
    )
    items = items.merge(bc, on=["Branch","Category"], how="left")
    k = 10.0
    items["shrunken_cutoff"] = (
        (items["samples"] * items["base_cutoff"] + k * items["bc_median"]) / (items["samples"] + k)
    )

    # Final cutoff used
    items["cutoff_used"] = items["shrunken_cutoff"].fillna(items["base_cutoff"])

    # Keep only needed columns (ordered)
    items = items[[
        "College","Branch","Category",
        "base_cutoff","mean_close","min_close","max_close",
        "samples","recent_year","trend_slope","last_year_close",
        "bc_median","shrunken_cutoff","cutoff_used"
    ]]

    return items


# -----------------------------
# Build training queries
# -----------------------------

@dataclass
class QuerySpec:
    year: int
    category: str
    rank: float

def sample_queries(df: pd.DataFrame, max_per_year_cat: int = 4000, seed: int = 42) -> List[QuerySpec]:
    rng = np.random.default_rng(seed)
    df = df.dropna(subset=["Year", "Category", "Closing Rank"]).copy()
    df["Year"] = df["Year"].astype(int)

    qs: List[QuerySpec] = []
    # sample realistic queries from actual closes (bounded to max_per_year_cat per (Y,Cat))
    for (y, cat), g in df.groupby(["Year", "Category"]):
        n = min(len(g), max_per_year_cat)
        g2 = g.sample(n=n, random_state=seed, replace=False)
        for r in g2["Closing Rank"].values:
            qs.append(QuerySpec(year=int(y), category=str(cat), rank=float(r)))
    return qs


def make_l2r_dataset(items: pd.DataFrame,
                     queries: List[QuerySpec],
                     candidate_limit: int = 300) -> Tuple[pd.DataFrame, List[int]]:
    """
    For each query:
      - candidate set = items in same Category, trimmed to candidate_limit by |mean_close - rank|
      - label = continuous score (1.0 at rank=0, decreasing to 0 as rank increases)
      - features = historical stats + query features
    """
    records = []
    group_sizes = []
    gid = 0

    # Pre-split items by category
    by_cat = {cat: g.reset_index(drop=True) for cat, g in items.groupby("Category")}

    for q in queries:
        cat_items = by_cat.get(q.category)
        if cat_items is None or len(cat_items) == 0:
            continue

        # Use mean_close for candidate selection
        diffs = np.abs(cat_items["mean_close"].values - q.rank)
        idx = np.argsort(diffs)[:candidate_limit]
        cand = cat_items.iloc[idx].copy()

        # Calculate relevance scores (higher is better)
        # Scale to integer range 0-100 for better ranking performance
        # We'll use a sigmoid-like function to map rank ratios to scores
        rank_ratio = q.rank / np.maximum(1, cand["cutoff_used"].values)
        scores = (1 / (1 + np.exp((rank_ratio - 1.0) * 5.0))) * 100  # 100 to 0 as rank_ratio increases
        scores = scores.astype(int)  # Convert to integer for ranking
        
        recs = pd.DataFrame({
            "group_id": gid,
            "label": scores,  # Using integer scores for ranking
            "query_year": q.year,
            "query_rank": q.rank,
            # Historical features
            "mean_close": cand["mean_close"].values,
            "min_close": cand["min_close"].values,
            "max_close": cand["max_close"].values,
            "samples": cand["samples"].values,
            "recent_year": cand["recent_year"].values,
            "trend_slope": cand["trend_slope"].values,
            "last_year_close": cand["last_year_close"].values,
            # For reference
            "cutoff_used": cand["cutoff_used"].values,
            "College": cand["College"].values,
            "Branch": cand["Branch"].values,
            "Category": cand["Category"].values
        })

        records.append(recs)
        group_sizes.append(len(recs))
        gid += 1

    if not records:
        raise RuntimeError("No training records were created. Check input data or sampling params.")

    return pd.concat(records, ignore_index=True), group_sizes


# -----------------------------
# Split by year (time-aware)
# -----------------------------

def split_by_year(queries: List[QuerySpec], train_until=2023, valid_year=2024, test_year=2025):
    train_q = [q for q in queries if q.year <= train_until]
    valid_q = [q for q in queries if q.year == valid_year]
    test_q  = [q for q in queries if q.year == test_year]
    return train_q, valid_q, test_q


# -----------------------------
# Train LightGBM Ranker
# -----------------------------

def train_l2r(X: pd.DataFrame, group_sizes: List[int]) -> Tuple[lgb.Booster, List[str]]:
    # Define columns to drop
    drop_cols = ["group_id", "label", "College", "Branch", "Category", "cutoff_used"]
    feature_cols = [c for c in X.columns if c not in drop_cols]
    
    print(f"Training on {len(feature_cols)} features: {', '.join(feature_cols[:5])}...")
    print(f"Training data: {len(X):,} examples in {len(group_sizes):,} groups")

    # Split data into train and validation sets (80-20 split)
    train_idx = []
    val_idx = []
    start = 0
    
    for g in group_sizes:
        if g <= 1:
            start += max(g, 1)
            continue
            
        group_indices = list(range(start, start + g))
        # Take first 80% for training, rest for validation
        split = int(0.8 * g)
        train_idx.extend(group_indices[:split])
        val_idx.extend(group_indices[split:])
        start += g
    
    X_train = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()
    
    # Create training dataset
    train_data = lgb.Dataset(
        X_train[feature_cols],
        label=X_train["label"].values,
        group=[]  # Will be set in the params
    )
    
    # Create validation dataset
    valid_data = lgb.Dataset(
        X_val[feature_cols],
        label=X_val["label"].values,
        reference=train_data,
        group=[]  # Will be set in the params
    )
    
    # Calculate group sizes for train and validation
    train_group_sizes = [g for g in group_sizes if g > 1]
    train_group_sizes = [int(0.8 * g) for g in train_group_sizes]
    valid_group_sizes = [g - int(0.8 * g) for g in group_sizes if g > 1]
    
    # Set group information in the dataset
    train_data.set_group(train_group_sizes)
    valid_data.set_group(valid_group_sizes)
    
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
        "lambdarank_truncation_level": 10,
        "label_gain": [i for i in range(101)],  # For integer labels 0-100
        "learning_rate": 0.1,
        "num_leaves": 31,  # Reduced to prevent overfitting
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "force_row_wise": True,
        "seed": 42,
        "num_threads": max(1, os.cpu_count() - 1 or 1),
        "verbosity": 1
    }
    
    print(f"Training on {len(train_idx):,} examples, validating on {len(val_idx):,} examples")
    print("\nStarting training...")
    
    # Train with early stopping
    model = lgb.train(
        params=params,
        train_set=train_data,
        num_boost_round=1000,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=True),
            lgb.log_evaluation(period=10)
        ]
    )
    
    return model, feature_cols


def evaluate_metrics(X: pd.DataFrame, group_sizes: List[int], model: lgb.Booster, feature_cols: List[str], k: int=10) -> dict:
    """Evaluate model and return ranking metrics."""
    scores = model.predict(X[feature_cols])
    labels = X["label"].values
    
    ndcgs = []
    ndcg5s = []  # NDCG@5
    
    start = 0
    for g in group_sizes:
        if g <= 1:  # Skip groups with 0 or 1 document
            start += max(g, 1)
            continue
            
        y = labels[start:start+g]
        p = scores[start:start+g]
        start += g
        
        # Calculate NDCG@k and NDCG@5
        try:
            ndcg = ndcg_score(y_true=[y], y_score=[p], k=min(k, len(y)))
            ndcgs.append(ndcg)
            
            if g >= 5:  # Only calculate NDCG@5 for groups with 5+ items
                ndcg5 = ndcg_score(y_true=[y], y_score=[p], k=min(5, len(y)))
                ndcg5s.append(ndcg5)
        except ValueError:
            # Skip this group if NDCG can't be computed
            continue
    
    # Calculate metrics
    metrics = {
        'ndcg@5': float(np.mean(ndcg5s)) if ndcg5s else 0.0,
        'ndcg@10': float(np.mean(ndcgs)) if ndcgs else 0.0,
        'num_groups': len(ndcgs)
    }
    return metrics

# Inference
# -----------------------------

def clean_branch_name(branch: str) -> str:
    """Clean and standardize branch names."""
    if not isinstance(branch, str) or not branch.strip():
        return ""

    # Convert to lowercase for case-insensitive matching
    branch = branch.lower().strip()
    
    # Map of common branch name variations to standard names
    branch_mapping = {
        'comp': 'Computer Science',
        'cse': 'Computer Science',
        'cs': 'Computer Science',
        'ise': 'Information Science',
        'mech': 'Mechanical',
        'me': 'Mechanical',
        'ece': 'Electronics and Communication',
        'eee': 'Electrical and Electronics',
        'ec': 'Electronics and Communication',
        'ee': 'Electrical and Electronics',
        'civil': 'Civil',
        'cve': 'Civil',
        'ai': 'Artificial Intelligence',
        'ml': 'Machine Learning',
        'ds': 'Data Science',
        'aiml': 'AI & ML',
        'aids': 'AI & DS'
    }
    
    # Check for exact matches first
    for key, value in branch_mapping.items():
        if branch == key.lower():
            return value
    
    # Check for partial matches
    for key, value in branch_mapping.items():
        if key in branch:
            return value
    
    # If no match found, clean up the input
    cleaned = (
        branch.title()
        .replace('Engg', 'Engineering')
        .replace('Eng', 'Engineering')
        .replace('Tech', 'Technology')
        .replace('  ', ' ')
        .strip()
    )
    
    # Remove any remaining "Engineering" duplicates
    cleaned = cleaned.replace('Engineering Engineering', 'Engineering')
    
    return cleaned if cleaned else branch
    return cleaned

def recommend(items: pd.DataFrame,
              model: lgb.Booster,
              feature_cols: List[str],
              user_rank: int,
              category: str,
              location_substr: Optional[str] = None,
              branch_substr: Optional[str] = None,
              topn: int = 15) -> pd.DataFrame:
    """
    Generate college recommendations based on rank, category, and optional filters.
    
    Args:
        items: DataFrame containing college data
        model: Trained ranking model
        feature_cols: List of feature columns
        user_rank: User's KCET rank
        category: Category (GM, 1G, etc.)
        location_substr: Comma-separated list of locations to filter by (e.g., "Bangalore,Mysore")
        branch_substr: Comma-separated list of branches to filter by (e.g., "Computer Science,Mechanical")
        topn: Number of top recommendations to return
    """
    """
    Generate college recommendations based on rank and category.
    
    Args:
        items: DataFrame containing college and branch information
        model: Trained LightGBM model
        feature_cols: List of feature column names
        user_rank: User's KCET rank
        category: Category (e.g., 'GM', '1G', etc.)
        location_substr: Optional location filter (e.g., 'Bangalore')
        topn: Number of top recommendations to return
        
    Returns:
        DataFrame with recommendations and metadata
    """
    # Clean branch names first
    items = items.copy()
    items["Branch"] = items["Branch"].apply(clean_branch_name)
    
    # Filter by category
    cands = items[items["Category"].str.casefold() == str(category).casefold()].copy()
    if len(cands) == 0:
        return pd.DataFrame(columns=["College", "Branch", "Match_Score", "Last_Year_Cutoff", "Data_Points", "Rank_Advantage_%"])

    # Handle multiple locations if specified
    if location_substr:
        locations = [loc.strip().lower() for loc in location_substr.split(',') if loc.strip()]
        if locations:
            location_mask = pd.Series(False, index=cands.index)
            for loc in locations:
                location_mask = location_mask | cands['College'].str.lower().str.contains(loc.lower(), na=False)
            cands = cands[location_mask].copy()
    
    # Handle multiple branches if specified
    if branch_substr:
        # Convert branch names to lowercase and clean them
        search_terms = [term.strip().lower() for term in branch_substr.split(',') if term.strip()]
        
        if search_terms:
            # First, create a clean version of the branch names for matching
            cands['Branch_Clean'] = cands['Branch'].str.lower()
            
            # Create a mask for branches that match any of the search terms
            branch_mask = pd.Series(False, index=cands.index)
            
            for term in search_terms:
                # Try direct match first (e.g., 'comp' in 'computer science')
                direct_match = cands['Branch_Clean'].str.contains(term, na=False)
                
                # Try cleaned version of the term (e.g., 'cs' -> 'computer science')
                cleaned_term = clean_branch_name(term).lower()
                cleaned_match = cands['Branch_Clean'].str.contains(cleaned_term, na=False)
                
                # Also check for common abbreviations
                abbr_matches = {
                    'comp': 'computer',
                    'mech': 'mechanical',
                    'cse': 'computer',
                    'cs': 'computer',
                    'ece': 'electronics',
                    'eee': 'electrical',
                    'civil': 'civil',
                    'ise': 'information',
                    'ec': 'electronics',
                    'ee': 'electrical'
                }
                
                abbr_match = pd.Series(False, index=cands.index)
                for abbr, full in abbr_matches.items():
                    if term == abbr:
                        abbr_match = abbr_match | cands['Branch_Clean'].str.contains(full, na=False)
                
                # Combine all matching conditions
                branch_mask = branch_mask | direct_match | cleaned_match | abbr_match
            
            # Apply the mask to filter candidates
            cands = cands[branch_mask].copy()
            
            # Clean up the temporary column
            if 'Branch_Clean' in cands.columns:
                cands = cands.drop(columns=['Branch_Clean'])
                
            if len(cands) == 0:
                return pd.DataFrame(columns=["College", "Branch", "Match_Score", "Last_Year_Cutoff", "Data_Points", "Rank_Advantage_%"])
                
            # Debug: Print found branches
            print("\nFound branches:", cands['Branch'].unique().tolist())
        if len(cands) == 0:
            return pd.DataFrame(columns=["College", "Branch", "Match_Score", "Last_Year_Cutoff", "Data_Points", "Rank_Advantage_%"])

    # Add query features
    cands["query_rank"] = float(user_rank)
    cands["query_year"] = items["recent_year"].max()

    # Calculate features
    cands["rank_ratio"] = user_rank / cands["last_year_close"].clip(lower=1)
    cands["safety_margin"] = cands["last_year_close"] - user_rank
    cands["rank_diff_pct"] = (cands["last_year_close"] - user_rank) / cands["last_year_close"] * 100

    # Ensure all required features are present
    for col in feature_cols:
        if col not in cands.columns:
            if col in ["query_rank", "query_year", "rank_ratio", "safety_margin"]:
                continue
            cands[col] = 0.0

    # Make predictions
    X = cands[feature_cols]
    cands["score"] = model.predict(X, num_iteration=model.best_iteration or 0)
    
    # Filter out colleges where rank is too far below cutoff (more than 50% worse)
    cands = cands[cands["rank_ratio"] < 1.5].copy()
    if len(cands) == 0:
        return pd.DataFrame(columns=["College", "Branch", "Match_Score", "Last_Year_Cutoff", "Data_Points", "Rank_Advantage_%"])

    # Calculate admission probability based on rank difference
    # If rank equals cutoff: 85%
    # If rank is better than cutoff: 75% at cutoff, decreases as difference increases
    # If rank is worse than cutoff: 85% at cutoff, increases up to 95% as difference increases
    
    # Calculate rank difference (positive means cutoff is higher than your rank = better chance)
    rank_diff = cands["last_year_close"] - user_rank
    
    # Initialize probability column
    cands["admission_prob"] = 0.0
    
    # Calculate rank difference (positive means your rank is better than cutoff)
    rank_diff = cands["last_year_close"] - user_rank
    
    # Initialize probability column
    cands["admission_prob"] = 0.0
    
    # Case 1: Your rank is better than cutoff (e.g., rank 50k vs cutoff 55k)
    better_mask = rank_diff > 0
    # Start at 85% and increase by 1% per 1000 rank difference
    prob_better = 0.85 + (0.01 * (rank_diff[better_mask] / 1000))
    cands.loc[better_mask, "admission_prob"] = prob_better.clip(0.85, 0.95)  # Min 85%, Max 95%
    
    # Case 2: Your rank equals cutoff (rank_diff = 0)
    equal_mask = rank_diff == 0
    cands.loc[equal_mask, "admission_prob"] = 0.85
    
    # Case 3: Your rank is worse than cutoff (e.g., rank 50k vs cutoff 49k)
    worse_mask = rank_diff < 0
    positive_diff = -rank_diff[worse_mask]  # Convert to positive for calculation
    # Start at 75% and decrease by 1% per 1000 rank difference
    prob_worse = 0.75 - (0.01 * (positive_diff / 1000))
    cands.loc[worse_mask, "admission_prob"] = prob_worse.clip(0.05, 0.75)  # Cap at 75%, min 5%
    
    # Ensure all probabilities are within 1-99% range (sanity check)
    cands["admission_prob"] = cands["admission_prob"].clip(0.01, 0.99)
    
    # Calculate confidence based on rank difference and data points
    cands["confidence"] = 50 + (50 * (1 - np.exp(-0.0001 * (cands["last_year_close"] - user_rank))))
    cands["confidence"] = cands["confidence"].clip(5, 95).round(1)
    
    # Filter to show colleges with cutoffs near the user's rank
    lower_bound = user_rank * 0.8
    upper_bound = user_rank * 1.2
    cands = cands[
        (cands["last_year_close"] >= lower_bound) &
        (cands["last_year_close"] <= upper_bound)
    ]

    # Categorize recommendations
    conditions = [
        (cands["last_year_close"] > 1.2 * user_rank),  # Reach
        (cands["last_year_close"] > user_rank),       # Ambitious
        (cands["last_year_close"] > 0.8 * user_rank), # Match
        (cands["last_year_close"] > 0.5 * user_rank), # Good
        (True)                                        # Safe
    ]
    choices = ["Reach", "Ambitious", "Match", "Good", "Safe"]
    cands["chance"] = np.select(conditions, choices, default="Safe")
    
    # Group by College and Branch, keeping the best entry for each combination
    # First, create a composite key of College + Branch
    cands["college_branch"] = cands["College"] + " | " + cands["Branch"]
    
    # Sort by a combination of score and admission probability
    cands["sort_score"] = cands["score"] * cands["admission_prob"]
    
    # Keep only the best entry for each college-branch combination
    cands = cands.sort_values("sort_score", ascending=False).drop_duplicates("college_branch")
    
    # Get the top N results
    cands = cands.head(topn)
    
    # Format the output
    result = cands[[
        "College", "Branch", "last_year_close", 
        "admission_prob", "confidence", "chance", "samples"
    ]].rename(columns={
        "last_year_close": "Last_Year_Cutoff",
        "admission_prob": "Admission_Probability",
        "confidence": "Confidence",
        "chance": "Category",
        "samples": "Data_Points"
    })

    # Format percentages
    result["Admission_Probability"] = (result["Admission_Probability"] * 100).round(1).astype(str) + "%"
    result["Confidence"] = result["Confidence"].astype(str) + "%"
    
    # Ensure consistent ordering
    result = result[[
        "College", "Branch", "Last_Year_Cutoff", 
        "Admission_Probability", "Confidence", "Category", "Data_Points"
    ]]
    
    return result.reset_index(drop=True)


# -----------------------------
# CLI commands
# -----------------------------

def cmd_train(args):
    start_time = time.time()
    out_dir = args.out_dir
    sample_size = args.sample_size
    os.makedirs(out_dir, exist_ok=True)

    # Initialize progress tracking
    progress = {
        'current': 0,
        'total': 6,  # Total number of major steps
        'start_time': start_time,
        'steps': [
            'Loading data',
            'Building items',
            'Sampling queries',
            'Building datasets',
            'Training model',
            'Evaluating and saving'
        ]
    }

    def log_progress(step_name=None):
        if step_name:
            print(f"\n{'-'*50}")
            print(f"Step {progress['current']+1}/{progress['total']}: {step_name}")
            print(f"{'-'*50}")
            progress['current'] += 1
        elapsed = time.time() - progress['start_time']
        print(f"Elapsed: {elapsed:.1f}s")

    log_progress(progress['steps'][0])
    # Load raw data
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df):,} records")
    
    # Sample the data if sample_size is specified
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"Using sampled dataset of {len(df):,} records for testing")

    log_progress(progress['steps'][1])
    # Build items
    print("Building item features...")
    items = build_item_table(df)
    print(f"Built {len(items):,} unique items")

    # Save items for inference
    items_path = os.path.join(out_dir, "items.parquet")
    try:
        items.to_parquet(items_path, index=False)
        print(f"Saved items to {items_path}")
    except Exception as e:
        print(f"Warning: Could not save as parquet ({str(e)}), falling back to CSV")
        items_path = os.path.join(out_dir, "items.csv")
        items.to_csv(items_path, index=False)
        print(f"Saved items to {items_path}")

    log_progress(progress['steps'][2])
    print("Sampling training queries...")
    queries_all = sample_queries(df, max_per_year_cat=args.max_per_year_cat, seed=42)
    print(f"Sampled {len(queries_all):,} queries in total")

    # Split by year
    print("\nSplitting data by year...")
    train_q, valid_q, test_q = split_by_year(
        queries_all,
        train_until=args.train_until,
        valid_year=args.valid_year,
        test_year=args.test_year
    )
    print(f"Split: {len(train_q):,} train, {len(valid_q):,} validation, {len(test_q):,} test queries")

    log_progress(progress['steps'][3])
    print("Building training dataset...")
    X_tr, grp_tr = make_l2r_dataset(items, train_q, candidate_limit=args.candidate_limit)
    print(f"Built training set with {len(X_tr):,} examples in {len(grp_tr):,} groups")
    
    print("\nBuilding validation dataset...")
    X_va, grp_va = make_l2r_dataset(items, valid_q, candidate_limit=args.candidate_limit)
    print(f"Built validation set with {len(X_va):,} examples in {len(grp_va):,} groups")
    
    print("\nBuilding test dataset...")
    X_te, grp_te = make_l2r_dataset(items, test_q, candidate_limit=args.candidate_limit)
    print(f"Built test set with {len(X_te):,} examples in {len(grp_te):,} groups")

    log_progress(progress['steps'][4])
    print("Starting model training...")
    model, feature_cols = train_l2r(X_tr, grp_tr)

    log_progress(progress['steps'][5])
    print("Evaluating model...")
    
    # Evaluate metrics on all splits
    metrics_tr = evaluate_metrics(X_tr, grp_tr, model, feature_cols, k=10)
    metrics_va = evaluate_metrics(X_va, grp_va, model, feature_cols, k=10)
    metrics_te = evaluate_metrics(X_te, grp_te, model, feature_cols, k=10)

    # Save model
    model_path = os.path.join(out_dir, "model_l2r.txt")
    model.save_model(model_path)
    print(f"\nSaved model to {model_path}")

    meta = {
        "feature_cols": feature_cols,
        "train_metrics": metrics_tr,
        "valid_metrics": metrics_va,
        "test_metrics": metrics_te,
        "train_queries": len(train_q),
        "valid_queries": len(valid_q),
        "test_queries": len(test_q),
        "candidate_limit": args.candidate_limit,
        "max_per_year_cat": args.max_per_year_cat,
        "train_until": args.train_until,
        "valid_year": args.valid_year,
        "test_year": args.test_year,
        "items_path": items_path,
        "training_time_seconds": time.time() - start_time
    }
    with open(os.path.join(out_dir, "feature_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    stats = {
        "n_items": int(len(items)),
        "n_colleges": int(items["College"].nunique()),
        "n_branches": int(items["Branch"].nunique()),
        "n_categories": int(items["Category"].nunique())
    }
    with open(os.path.join(out_dir, "cutoff_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Artifacts saved to: {out_dir}")
    
    # Print detailed metrics
    print("\nModel Performance:")
    print("-" * 60)
    print(f"{'Metric':<15} | {'Training':<15} | {'Validation':<15} | {'Test':<15}")
    print("-" * 60)
    
    # NDCG metrics
    print(f"{'NDCG@5':<15} | {metrics_tr.get('ndcg@5', 0.0):>15.4f} | {metrics_va.get('ndcg@5', 0.0):>15.4f} | {metrics_te.get('ndcg@5', 0.0):>15.4f}")
    print(f"{'NDCG@10':<15} | {metrics_tr.get('ndcg@10', 0.0):>15.4f} | {metrics_va.get('ndcg@10', 0.0):>15.4f} | {metrics_te.get('ndcg@10', 0.0):>15.4f}")
    
    # Number of groups
    print(f"{'Num Groups':<15} | {metrics_tr.get('num_groups', 0):>15,} | {metrics_va.get('num_groups', 0):>15,} | {metrics_te.get('num_groups', 0):>15,}")
    
    # Additional metrics if available
    if 'accuracy' in metrics_tr:
        print(f"{'Accuracy':<15} | {metrics_tr.get('accuracy', 0.0):>15.4f} | {metrics_va.get('accuracy', 0.0):>15.4f} | {metrics_te.get('accuracy', 0.0):>15.4f}")
    
    # Print model info
    print("\nModel Info:")
    print(f"- Features: {len(feature_cols)}")
    print(f"- Training queries: {len(grp_tr)}")
    print(f"- Validation queries: {len(grp_va)}")
    print(f"- Test queries: {len(grp_te)}")
    
    print("\nTo use this model for recommendations, run:")
    print(f"  python {os.path.basename(__file__)} recommend --rank RANK --category CATEGORY [--location LOCATION]")


def cmd_recommend(args):
    # Define artifact paths
    items_parquet = "artifacts/items.parquet"
    model_path = "artifacts/model_l2r.txt"
    meta_path = "artifacts/feature_meta.json"

    # Check if required files exist
    if not os.path.exists(model_path) or not os.path.exists(meta_path) or not os.path.exists(items_parquet):
        raise SystemExit("Artifacts not found. Please run training first using: python kcet_l2r_train_infer.py train --data <your_data_file>")

    # Load items
    try:
        items = pd.read_parquet(items_parquet)
    except Exception as e:
        raise SystemExit(f"Error loading items: {str(e)}")

    # Load model + meta
    try:
        model = lgb.Booster(model_file=model_path)
        with open(meta_path, "r") as f:
            feature_cols = json.load(f)["feature_cols"]
    except Exception as e:
        raise SystemExit(f"Error loading model or feature metadata: {str(e)}")

    recs = recommend(
        items=items,
        model=model,
        feature_cols=feature_cols,
        user_rank=int(args.rank),
        category=args.category,
        location_substr=args.location,
        branch_substr=args.branch,
        topn=int(args.topn)
    )
    if len(recs) > 0:
        # Format the output for better readability
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', 40)
        
        # Create a copy to avoid modifying the original
        display_df = recs.copy()
        
        # Truncate long strings
        display_df['College'] = display_df['College'].str[:40] + '...'
        display_df['Branch'] = display_df['Branch'].str[:30] + '...'
        
        # Format percentages - handle both numeric and string percentages
        def format_percent(x):
            if isinstance(x, str) and '%' in x:
                return x  # Already formatted as percentage
            try:
                # Try to convert to float if it's a string without %
                x_float = float(x) if isinstance(x, str) else x
                return f"{x_float:.1f}%"
            except (ValueError, TypeError):
                return str(x)  # Return as is if conversion fails
                
        if 'Admission_Probability' in display_df.columns:
            display_df['Admission_Probability'] = display_df['Admission_Probability'].apply(format_percent)
        if 'Confidence' in display_df.columns:
            display_df['Confidence'] = display_df['Confidence'].apply(format_percent)
        
        # Reorder and select columns that exist in the DataFrame
        available_columns = [col for col in display_df.columns.tolist() 
                           if col not in ['Confidence', 'Data_Points']]
        columns_to_show = [
            'College_Code', 'College', 'Branch', 'Last_Year_Cutoff', 
            'Admission_Probability', 'Category'
        ]
        # Only include columns that exist in the DataFrame
        display_columns = [col for col in columns_to_show if col in available_columns]
        display_df = display_df[display_columns]
        
        # Rename columns for better display
        rename_map = {
            'College_Code': 'College Code',
            'College': 'College Name (Top 40 chars)',
            'Branch': 'Branch (Top 30 chars)',
            'Last_Year_Cutoff': 'Last Year Cutoff',
            'Admission_Probability': 'Admission %',
            'Category': 'Category'
        }
        # Only rename columns that exist in the DataFrame
        rename_map = {k: v for k, v in rename_map.items() if k in display_df.columns}
        if rename_map:
            display_df = display_df.rename(columns=rename_map)
        
        # Print the formatted table
        print("\n" + "="*120)
        print(f"RECOMMENDATIONS FOR RANK {args.rank} - {args.category} (Location: {args.location or 'All'}, Branch: {args.branch or 'All'})")
        print("="*120)
        print(display_df.to_string(index=False))
        print("\nNote: Colleges and branches are truncated for display. Use --topn to see more options.")
        
        # Print full details for top 3 recommendations
        print("\n" + "-"*120)
        print("DETAILS FOR TOP RECOMMENDATIONS")
        print("-"*120)
        for i, (_, row) in enumerate(recs.head(3).iterrows(), 1):
            print(f"\n{i}. {row['College']}")
            print(f"   College Code: {row.get('College_Code', 'N/A')}")
            # Clean up branch name for display
            branch_name = str(row['Branch']).replace('Engineeringineering', 'Engineering').replace('anical', '').strip()
            print(f"   Branch: {branch_name}")
            print(f"   Last Year Cutoff: {int(row['Last_Year_Cutoff']):,}")
            
            # Handle Admission_Probability which might be a string with % or a number
            prob_col = 'Admission_Probability' if 'Admission_Probability' in row else 'Admission %'
            if prob_col in row:
                prob = row[prob_col]
                if isinstance(prob, str):
                    # Remove % if present and convert to float
                    prob = float(prob.replace('%', '').strip()) / 100.0 if '%' in str(prob) else float(prob)
                print(f"   Admission Probability: {prob*100:.1f}%")
    else:
        print("\nNo recommendations found for the given criteria. Try adjusting your rank or location.")

def main():
    p = argparse.ArgumentParser(description='KCET College Recommendation System')
    sub = p.add_subparsers(dest='command')

    # Train command
    p_train = sub.add_parser("train", help="Train the LightGBM LambdaMART model")
    p_train.add_argument("--data", type=str, required=True, help="Path to KCET_cleaned_with_2025.csv")
    p_train.add_argument("--out_dir", type=str, default="artifacts", help="Directory to save model artifacts")
    p_train.add_argument("--sample_size", type=int, default=None, help="Sample size for testing (None for full dataset)")
    p_train.add_argument("--candidate_limit", type=int, default=300, help="Candidates per query (trim by cutoff proximity)")
    p_train.add_argument("--max_per_year_cat", type=int, default=4000, help="Max queries sampled per (year, category)")
    p_train.add_argument("--train_until", type=int, default=2023, help="Last year to include in training data")
    p_train.add_argument("--valid_year", type=int, default=2024, help="Year to use for validation")
    p_train.add_argument("--test_year", type=int, default=2025, help="Year to use for testing")
    p_train.set_defaults(func=cmd_train)

    # Recommend command
    p_recommend = sub.add_parser('recommend', help='Get recommendations')
    p_recommend.add_argument('--rank', type=int, required=True, help='Your KCET rank')
    p_recommend.add_argument('--category', type=str, required=True, help='Your category (GM, 1G, 2A, 2B, 3A, 3B, SC, ST)')
    p_recommend.add_argument('--location', type=str, 
                           help='Filter by one or more locations (comma-separated, e.g., "Bangalore,Mysore,Hubli")')
    p_recommend.add_argument('--branch', type=str, 
                           help='Filter by one or more branches (comma-separated, e.g., "Computer Science,Mechanical,Electronics")'
                           'Note: Use partial names (e.g., "Comp" for Computer Science, "Mech" for Mechanical)')
    p_recommend.add_argument('--topn', type=int, default=10, help='Number of recommendations to show (default: 10)')
    p_recommend.set_defaults(func=cmd_recommend)

    args = p.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        p.print_help()

if __name__ == "__main__":
    main()
