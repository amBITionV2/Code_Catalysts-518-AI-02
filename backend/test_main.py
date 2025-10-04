"""
Test version of FastAPI Backend for KCET College Recommendation System
"""

import os
import sys
import json
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import numpy as np

# Simple test implementation without model loading
app = FastAPI(
    title="KCET College Recommendation API (Test)",
    description="Test version - Get personalized college recommendations",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendationRequest(BaseModel):
    rank: int = Field(..., ge=1, le=200000, description="KCET rank (1-200000)")
    category: str = Field(..., description="Category (e.g., GM, 1G, 2AK, etc.)")
    location: Optional[str] = Field(None, description="Optional location filter")
    top_n: int = Field(10, ge=1, le=50, description="Number of recommendations")

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

@app.get("/")
async def root():
    return {
        "message": "KCET College Recommendation API (Test Mode)",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "data_loaded": True,
        "total_colleges": 50,
        "total_branches": 150
    }

@app.get("/categories")
async def get_categories():
    return ["GM", "1G", "2AK", "2AG", "2BK", "2BG", "3AK", "3AG", "3BK", "3BG", "SC", "ST"]

@app.get("/locations") 
async def get_locations():
    return ["Bangalore", "Mysore", "Mangalore", "Hubli", "Dharwad", "Belgaum", "Gulbarga", "Davangere"]

@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    # Generate sample recommendations
    recommendations = []
    
    for i in range(min(request.top_n, 10)):
        rank_diff = np.random.normal(10, 20)  # Random rank difference
        admission_prob = max(5, min(95, 50 + rank_diff))
        
        rec = CollegeRecommendation(
            college=f"Test Engineering College {i+1}",
            branch=f"Computer Science Engineering",
            match_score=round(np.random.uniform(0.6, 0.95), 4),
            last_year_cutoff=int(request.rank + np.random.randint(-5000, 10000)),
            data_points=np.random.randint(5, 50),
            rank_advantage_percent=round(rank_diff, 2),
            admission_probability=round(admission_prob, 2),
            accuracy=round(np.random.uniform(60, 90), 2),
            chance="Safe" if rank_diff > 20 else "Good" if rank_diff > 0 else "Match" if rank_diff > -20 else "Reach"
        )
        recommendations.append(rec)
    
    return RecommendationResponse(
        recommendations=recommendations,
        top_3_details=recommendations[:3],
        user_info={
            "rank": request.rank,
            "category": request.category,
            "location": request.location or "All"
        },
        message=f"Generated {len(recommendations)} test recommendations"
    )

@app.get("/stats")
async def get_statistics():
    return {
        "total_items": 1000,
        "unique_colleges": 50,
        "unique_branches": 150,
        "unique_categories": 12,
        "data_years": "2015-2025",
        "min_cutoff": 1000,
        "max_cutoff": 180000
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Test KCET Recommendation API...")
    uvicorn.run("test_main:app", host="0.0.0.0", port=8000, reload=False)