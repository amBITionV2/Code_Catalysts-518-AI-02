"""
FastAPI Backend for KCET College Recommendation System
"""

import os
import sys
import json
from typing import List, Optional
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import lightgbm as lgb

# Add the scripts directory to Python path to import our model functions
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))

from kcet_l2r_train_infer import recommend

# Global variables for model and data
model = None
items_df = None
feature_cols = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    global model, items_df, feature_cols
    
    try:
        print("🚀 Initializing KCET Recommendation System...")
        
        # Define paths relative to backend directory
        backend_dir = Path(__file__).parent
        project_root = backend_dir.parent
        artifacts_dir = project_root / "artifacts"
        
        # Load model
        model_path = artifacts_dir / "model_l2r.txt"
        if model_path.exists():
            model = lgb.Booster(model_file=str(model_path))
            print(f"✅ Model loaded from {model_path}")
        else:
            print(f"❌ Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load items data
        items_path = artifacts_dir / "items.parquet"
        if items_path.exists():
            items_df = pd.read_parquet(items_path)
            print(f"✅ Items data loaded: {len(items_df)} items")
        else:
            # Try CSV fallback
            items_csv_path = artifacts_dir / "items.csv"
            if items_csv_path.exists():
                items_df = pd.read_csv(items_csv_path)
                print(f"✅ Items data loaded from CSV: {len(items_df)} items")
            else:
                print(f"❌ Items file not found at {items_path} or {items_csv_path}")
                raise FileNotFoundError(f"Items file not found")
        
        # Load feature columns
        feature_meta_path = artifacts_dir / "feature_meta.json"
        if feature_meta_path.exists():
            with open(feature_meta_path, 'r') as f:
                meta = json.load(f)
                feature_cols = meta.get("feature_cols", [])
                print(f"✅ Feature columns loaded: {len(feature_cols)} features")
        else:
            print(f"❌ Feature metadata not found at {feature_meta_path}")
            raise FileNotFoundError(f"Feature metadata not found")
        
        print("🎉 Backend initialized successfully!")
        
    except Exception as e:
        print(f"❌ Error during startup: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    print("🔄 Shutting down KCET Recommendation System...")

app = FastAPI(
    title="KCET College Recommendation API",
    description="Get personalized college recommendations based on your KCET rank and preferences",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration to allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendationRequest(BaseModel):
    rank: int = Field(..., ge=1, le=200000, description="KCET rank (1-200000)")
    category: str = Field(..., description="Category (e.g., GM, 1G, 2AK, etc.)")
    location: Optional[str] = Field(None, description="Optional location filter (e.g., Bangalore, Mysore)")
    top_n: int = Field(10, ge=1, le=50, description="Number of recommendations (1-50)")

class CollegeRecommendation(BaseModel):
    college: str
    branch: str
    match_score: float
    last_year_cutoff: int
    data_points: int
    rank_advantage_percent: float
    admission_probability: float
    accuracy: float
    chance: str

class RecommendationResponse(BaseModel):
    recommendations: List[CollegeRecommendation]
    top_3_details: List[CollegeRecommendation]
    user_info: dict
    message: str

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    model_loaded: bool
    data_loaded: bool
    total_colleges: int
    total_branches: int

@app.get("/", response_model=dict)
async def root():
    """Welcome endpoint"""
    return {
        "message": "KCET College Recommendation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global model, items_df
    
    model_loaded = model is not None
    data_loaded = items_df is not None
    
    total_colleges = 0
    total_branches = 0
    
    if data_loaded:
        total_colleges = items_df["College"].nunique() if "College" in items_df.columns else 0
        total_branches = items_df["Branch"].nunique() if "Branch" in items_df.columns else 0
    
    return HealthResponse(
        status="healthy" if model_loaded and data_loaded else "unhealthy",
        model_loaded=model_loaded,
        data_loaded=data_loaded,
        total_colleges=total_colleges,
        total_branches=total_branches
    )

@app.get("/categories", response_model=List[str])
async def get_categories():
    """Get all available categories"""
    global items_df
    
    if items_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    if "Category" not in items_df.columns:
        raise HTTPException(status_code=500, detail="Category column not found in data")
    
    categories = sorted(items_df["Category"].unique().tolist())
    return categories

@app.get("/locations", response_model=List[str])
async def get_locations():
    """Get popular locations (cities) from college names"""
    global items_df
    
    if items_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    if "College" not in items_df.columns:
        raise HTTPException(status_code=500, detail="College column not found in data")
    
    # Extract common city names from college names
    common_cities = [
        "Bangalore", "Bengaluru", "Mysore", "Mangalore", "Hubli", "Dharwad", 
        "Belgaum", "Gulbarga", "Davangere", "Shimoga", "Tumkur", "Hassan",
        "Mandya", "Udupi", "Bijapur", "Raichur", "Bellary", "Chitradurga"
    ]
    
    # Find cities that actually exist in college names
    available_cities = []
    college_names = items_df["College"].str.lower().str.cat(sep=" ")
    
    for city in common_cities:
        if city.lower() in college_names:
            available_cities.append(city)
    
    return sorted(available_cities)

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get college recommendations based on user preferences"""
    global model, items_df, feature_cols
    
    if model is None or items_df is None:
        raise HTTPException(status_code=503, detail="Model or data not loaded")
    
    try:
        # Call the recommendation function from your existing code
        recommendations_df = recommend(
            items=items_df,
            model=model,
            feature_cols=feature_cols,
            user_rank=request.rank,
            category=request.category,
            location_substr=request.location,
            topn=request.top_n
        )
        
        if recommendations_df.empty:
            return RecommendationResponse(
                recommendations=[],
                top_3_details=[],
                user_info={
                    "rank": request.rank,
                    "category": request.category,
                    "location": request.location or "All"
                },
                message="No recommendations found for the given criteria. Try adjusting your rank or location preference."
            )
        
        # Convert DataFrame to Pydantic models
        recommendations = []
        for _, row in recommendations_df.iterrows():
            rec = CollegeRecommendation(
                college=str(row["College"]),
                branch=str(row["Branch"]),
                match_score=float(row["Match_Score"]),
                last_year_cutoff=int(row["Last_Year_Cutoff"]),
                data_points=int(row["Data_Points"]),
                rank_advantage_percent=float(row["Rank_Advantage_%"]),
                admission_probability=float(row["Admission_Probability"]),
                accuracy=float(row["Accuracy"]),
                chance=str(row["Chance"])
            )
            recommendations.append(rec)
        
        # Get top 3 for detailed view
        top_3_details = recommendations[:3]
        
        return RecommendationResponse(
            recommendations=recommendations,
            top_3_details=top_3_details,
            user_info={
                "rank": request.rank,
                "category": request.category,
                "location": request.location or "All"
            },
            message=f"Found {len(recommendations)} recommendations for rank {request.rank} in {request.category} category."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.get("/stats", response_model=dict)
async def get_statistics():
    """Get general statistics about the dataset"""
    global items_df
    
    if items_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    try:
        stats = {
            "total_items": len(items_df),
            "unique_colleges": items_df["College"].nunique() if "College" in items_df.columns else 0,
            "unique_branches": items_df["Branch"].nunique() if "Branch" in items_df.columns else 0,
            "unique_categories": items_df["Category"].nunique() if "Category" in items_df.columns else 0,
            "data_years": f"2015-{items_df['recent_year'].max()}" if "recent_year" in items_df.columns else "Unknown",
            "min_cutoff": int(items_df["last_year_close"].min()) if "last_year_close" in items_df.columns else 0,
            "max_cutoff": int(items_df["last_year_close"].max()) if "last_year_close" in items_df.columns else 0,
        }
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)