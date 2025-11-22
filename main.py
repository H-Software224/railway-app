from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
from typing import List
import random
import os

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로드 (초기화 시 한 번만 로드)
scaler = None
kmeans = None

def load_models():
    global scaler, kmeans
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('kmeans.pkl', 'rb') as f:
            kmeans = pickle.load(f)
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")

@app.on_event("startup")
async def startup_event():
    load_models()

class NutrientDeficiency(BaseModel):
    protein: float
    fat: float
    carbohydrate: float
    vitamin_a: float
    thiamine: float
    riboflavin: float
    vitamin_c: float
    calcium: float
    iron: float

class FoodRecommendationRequest(BaseModel):
    deficiencies: NutrientDeficiency

class FoodItem(BaseModel):
    food_code: str
    food_name: str
    calories: float
    protein: float
    fat: float
    carbohydrate: float
    vitamin_a: float
    thiamine: float
    riboflavin: float
    vitamin_c: float
    calcium: float
    iron: float

@app.post("/predict-cluster")
async def predict_cluster(request: FoodRecommendationRequest):
    """영양소 부족량을 입력받아 클러스터 ID를 예측"""
    if scaler is None or kmeans is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        # feature vector 생성 (9개 영양소)
        feature_vector = np.array([[
            request.deficiencies.protein,
            request.deficiencies.fat,
            request.deficiencies.carbohydrate,
            request.deficiencies.vitamin_a,
            request.deficiencies.thiamine,
            request.deficiencies.riboflavin,
            request.deficiencies.vitamin_c,
            request.deficiencies.calcium,
            request.deficiencies.iron
        ]])
        
        # 스케일링
        scaled_vector = scaler.transform(feature_vector)
        
        # 클러스터 예측
        cluster_id = int(kmeans.predict(scaled_vector)[0])
        
        return {
            "cluster_id": cluster_id,
            "scaled_features": scaled_vector.tolist()[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "scaler_loaded": scaler is not None,
        "kmeans_loaded": kmeans is not None
    }
