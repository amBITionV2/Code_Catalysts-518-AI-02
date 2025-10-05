#!/usr/bin/env python3
"""
Create a comprehensive test model with realistic KCET data
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import os
from pathlib import Path

def create_realistic_test_data():
    """Create realistic test data that mimics real KCET patterns"""
    
    print("Creating comprehensive test dataset...")
    
    # Real Karnataka engineering college names and locations
    colleges = [
        ("RV College of Engineering", "Bangalore"),
        ("BMS College of Engineering", "Bangalore"), 
        ("PES University", "Bangalore"),
        ("MS Ramaiah Institute of Technology", "Bangalore"),
        ("Dayananda Sagar College of Engineering", "Bangalore"),
        ("Sir M Visvesvaraya Institute of Technology", "Bangalore"),
        ("Bangalore Institute of Technology", "Bangalore"),
        ("New Horizon College of Engineering", "Bangalore"),
        ("JSS Science and Technology University", "Mysore"),
        ("Maharaja Institute of Technology", "Mysore"),
        ("National Institute of Engineering", "Mysore"),
        ("Siddaganga Institute of Technology", "Tumkur"),
        ("Sri Jayachamarajendra College of Engineering", "Mysore"),
        ("KLE Dr. M.S. Sheshgiri College of Engineering", "Belgaum"),
        ("Basaveshwar Engineering College", "Bagalkot"),
        ("SDM College of Engineering and Technology", "Dharwad"),
        ("KLS Gogte Institute of Technology", "Belgaum"),
        ("B.V. Bhoomaraddi College of Engineering", "Hubli"),
        ("NMAM Institute of Technology", "Mangalore"),
        ("Sahyadri College of Engineering", "Mangalore"),
        ("Manipal Institute of Technology", "Manipal"),
        ("NIE Institute of Technology", "Mysore"),
        ("Kalpataru Institute of Technology", "Tiptur"),
        ("Atria Institute of Technology", "Bangalore"),
        ("CMR Institute of Technology", "Bangalore")
    ]
    
    # Engineering branches
    branches = [
        "Computer Science and Engineering",
        "Information Science and Engineering", 
        "Electronics and Communication Engineering",
        "Mechanical Engineering",
        "Civil Engineering",
        "Electrical and Electronics Engineering",
        "Chemical Engineering",
        "Industrial Engineering",
        "Biotechnology",
        "Aeronautical Engineering"
    ]
    
    # Categories
    categories = ["GM", "1G", "2AK", "2AG", "2BK", "2BG", "3AK", "3AG", "3BK", "3BG", "SC", "ST"]
    
    # Create dataset
    data = []
    
    for college, location in colleges:
        for branch in branches:
            for category in categories:
                # Create realistic cutoff ranges based on category and branch popularity
                if category == "GM":
                    base_cutoff = np.random.randint(1000, 50000)
                elif category in ["1G", "2AK"]:
                    base_cutoff = np.random.randint(500, 30000) 
                else:
                    base_cutoff = np.random.randint(200, 20000)
                
                # Adjust for branch popularity
                if "Computer Science" in branch or "Information Science" in branch:
                    base_cutoff = int(base_cutoff * 0.7)  # More competitive
                elif "Electronics" in branch:
                    base_cutoff = int(base_cutoff * 0.8)
                elif "Mechanical" in branch or "Civil" in branch:
                    base_cutoff = int(base_cutoff * 1.2)  # Less competitive
                
                # Create multiple years of data
                for year in range(2020, 2026):
                    yearly_variation = np.random.normal(1.0, 0.1)
                    cutoff = max(1, int(base_cutoff * yearly_variation))
                    
                    data.append({
                        'College': college,
                        'Branch': branch,
                        'Category': category,
                        'last_year_close': cutoff,
                        'recent_year': year,
                        'samples': np.random.randint(5, 50),
                        'location': location
                    })
    
    df = pd.DataFrame(data)
    print(f"Created dataset with {len(df)} rows")
    print(f"Colleges: {df['College'].nunique()}")
    print(f"Branches: {df['Branch'].nunique()}")
    print(f"Categories: {df['Category'].nunique()}")
    print(f"Cutoff range: {df['last_year_close'].min()} - {df['last_year_close'].max()}")
    
    return df

def build_features(df):
    """Build features for the LightGBM model"""
    
    print("Building features...")
    
    # Group by College, Branch, Category for final items
    items = df.groupby(['College', 'Branch', 'Category']).agg({
        'last_year_close': 'mean',
        'samples': 'sum',
        'recent_year': 'max'
    }).reset_index()
    
    # Add location back
    location_map = df.groupby('College')['location'].first().to_dict()
    items['location'] = items['College'].map(location_map)
    
    # Create features
    feature_cols = [
        'last_year_close', 'samples', 'recent_year'
    ]
    
    return items, feature_cols

def create_training_data(items, feature_cols):
    """Create training data for LightGBM"""
    
    print("Creating training data...")
    
    # Create query groups (simulate different user queries)
    queries = []
    for rank in [1000, 5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000]:
        for category in ['GM', '1G', '2AK', 'SC']:
            query_items = items[items['Category'] == category].copy()
            if len(query_items) == 0:
                continue
                
            # Add query features
            query_items['query_rank'] = rank
            query_items['query_year'] = 2024
            
            # Create relevance scores (higher for closer matches to rank)
            query_items['relevance'] = np.where(
                abs(query_items['last_year_close'] - rank) < rank * 0.2,
                2,  # Highly relevant
                np.where(
                    abs(query_items['last_year_close'] - rank) < rank * 0.5,
                    1,  # Somewhat relevant
                    0   # Not relevant
                )
            )
            
            queries.append(query_items)
    
    if not queries:
        print("No training queries created!")
        return None, None, None
    
    # Combine all queries
    train_data = pd.concat(queries, ignore_index=True)
    
    # Extended feature columns including query features
    extended_feature_cols = feature_cols + ['query_rank', 'query_year']
    
    # Group sizes for LightGBM ranking
    group_sizes = []
    current_query = None
    current_size = 0
    
    for _, row in train_data.iterrows():
        query_id = f"{row['query_rank']}_{row['Category']}"
        if current_query != query_id:
            if current_size > 0:
                group_sizes.append(current_size)
            current_query = query_id
            current_size = 1
        else:
            current_size += 1
    
    if current_size > 0:
        group_sizes.append(current_size)
    
    print(f"Created {len(train_data)} training samples with {len(group_sizes)} query groups")
    
    return train_data, extended_feature_cols, group_sizes

def train_model(train_data, feature_cols, group_sizes):
    """Train LightGBM ranking model"""
    
    print("Training LightGBM model...")
    
    X = train_data[feature_cols].values
    y = train_data['relevance'].values
    
    # Create LightGBM dataset for ranking
    train_dataset = lgb.Dataset(
        X, label=y, group=group_sizes,
        feature_name=feature_cols
    )
    
    # Parameters for LambdaMART
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [5, 10],
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'num_threads': 4
    }
    
    # Train model
    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=100,
        valid_sets=[train_dataset],
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
    )
    
    print("Model training completed!")
    return model

def main():
    # Set up paths
    project_root = Path(__file__).parent.parent
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    
    print("🚀 Creating comprehensive test model for KCET recommendations...")
    
    # Create realistic test data
    df = create_realistic_test_data()
    
    # Build features
    items, feature_cols = build_features(df)
    
    # Create training data
    train_data, extended_feature_cols, group_sizes = create_training_data(items, feature_cols)
    
    if train_data is None:
        print("❌ Failed to create training data")
        return
    
    # Train model
    model = train_model(train_data, extended_feature_cols, group_sizes)
    
    # Save artifacts
    print("Saving artifacts...")
    
    # Save items
    items_path = artifacts_dir / "items.parquet"
    items.to_parquet(items_path, index=False)
    print(f"✅ Saved items to {items_path}")
    
    # Save model
    model_path = artifacts_dir / "model_l2r.txt"
    model.save_model(str(model_path))
    print(f"✅ Saved model to {model_path}")
    
    # Save feature metadata
    feature_meta = {
        "feature_cols": extended_feature_cols,
        "total_features": len(extended_feature_cols),
        "training_samples": len(train_data),
        "query_groups": len(group_sizes)
    }
    
    feature_meta_path = artifacts_dir / "feature_meta.json"
    with open(feature_meta_path, 'w') as f:
        json.dump(feature_meta, f, indent=2)
    print(f"✅ Saved feature metadata to {feature_meta_path}")
    
    # Save cutoff stats
    cutoff_stats = {
        "total_items": len(items),
        "unique_colleges": items['College'].nunique(),
        "unique_branches": items['Branch'].nunique(), 
        "unique_categories": items['Category'].nunique(),
        "min_cutoff": int(items['last_year_close'].min()),
        "max_cutoff": int(items['last_year_close'].max()),
        "avg_cutoff": int(items['last_year_close'].mean())
    }
    
    cutoff_stats_path = artifacts_dir / "cutoff_stats.json"
    with open(cutoff_stats_path, 'w') as f:
        json.dump(cutoff_stats, f, indent=2)
    print(f"✅ Saved cutoff stats to {cutoff_stats_path}")
    
    print(f"\n🎉 Comprehensive test model created successfully!")
    print(f"📊 Dataset Summary:")
    print(f"   - {cutoff_stats['unique_colleges']} colleges")
    print(f"   - {cutoff_stats['unique_branches']} branches") 
    print(f"   - {cutoff_stats['unique_categories']} categories")
    print(f"   - Cutoff range: {cutoff_stats['min_cutoff']:,} - {cutoff_stats['max_cutoff']:,}")
    print(f"\n🔄 Restart your backend to load the new model!")

if __name__ == "__main__":
    main()