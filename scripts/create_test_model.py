"""
Simple test script to create a basic LightGBM model for testing
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from pathlib import Path

# Create a simple synthetic dataset for testing
print("Creating test model...")

# Generate sample data
np.random.seed(42)
n_samples = 1000

data = {
    'query_year': np.random.choice([2020, 2021, 2022, 2023, 2024], n_samples),
    'query_rank': np.random.randint(1000, 100000, n_samples),
    'mean_close': np.random.randint(5000, 120000, n_samples),
    'min_close': np.random.randint(1000, 80000, n_samples),
    'max_close': np.random.randint(10000, 150000, n_samples),
    'samples': np.random.randint(1, 50, n_samples),
    'recent_year': np.random.choice([2023, 2024], n_samples),
    'trend_slope': np.random.normal(0, 100, n_samples),
    'last_year_close': np.random.randint(5000, 120000, n_samples),
    'label': np.random.randint(0, 101, n_samples)  # 0-100 for ranking
}

df = pd.DataFrame(data)
feature_cols = [col for col in df.columns if col != 'label']

print(f"Created dataset with {len(df)} rows and {len(feature_cols)} features")

# Create training dataset
train_data = lgb.Dataset(df[feature_cols], label=df['label'])

# Simple LightGBM parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'seed': 42,
    'verbosity': -1
}

print("Training LightGBM model...")
model = lgb.train(params, train_data, num_boost_round=100)

# Save model and metadata
artifacts_dir = Path("../artifacts")
artifacts_dir.mkdir(exist_ok=True)

model_path = artifacts_dir / "model_l2r.txt"
model.save_model(str(model_path))
print(f"Saved model to {model_path}")

# Create feature metadata
meta = {
    "feature_cols": feature_cols,
    "train_metrics": {"rmse": 0.1},
    "model_type": "test_model",
    "created": "2025-10-05"
}

meta_path = artifacts_dir / "feature_meta.json"
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)
print(f"Saved metadata to {meta_path}")

# Create simple items data
items_data = []
colleges = ["Test College A", "Test College B", "Test College C"]
branches = ["Computer Science", "Electronics", "Mechanical"]
categories = ["GM", "1G", "2AK"]

for college in colleges:
    for branch in branches:
        for category in categories:
            items_data.append({
                "College": college,
                "Branch": branch,
                "Category": category,
                "base_cutoff": np.random.randint(10000, 80000),
                "mean_close": np.random.randint(10000, 80000),
                "min_close": np.random.randint(5000, 50000),
                "max_close": np.random.randint(20000, 100000),
                "samples": np.random.randint(5, 30),
                "recent_year": 2024,
                "trend_slope": np.random.normal(0, 50),
                "last_year_close": np.random.randint(10000, 80000)
            })

items_df = pd.DataFrame(items_data)
items_path = artifacts_dir / "items.parquet"
items_df.to_parquet(items_path, index=False)
print(f"Saved items data to {items_path}")

print("✅ Test model created successfully!")
print("\nNext steps:")
print("1. Go back to backend directory: cd ../backend")
print("2. Start the server: python main.py")