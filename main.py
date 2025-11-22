from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

app = FastAPI()

# 모델과 스케일러 로드 (전역 변수로 선언)
model = None
scaler = None

@app.on_event("startup")
def load_models():
    global model, scaler
    try:
        # 파일이 같은 폴더에 있다고 가정
        model = joblib.load('kmeans_model.pkl')
        scaler = joblib.load('scaler.pkl')
        print("✅ 모델 및 스케일러 로드 완료")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")

# 입력 데이터 정의 (scaler.pkl의 feature 순서와 이름을 반영)
class NutrientInput(BaseModel):
    # Field(..., alias="한글명")을 사용하면 JSON에서 한글 키로 값을 보낼 수 있습니다.
    carbs: float = Field(..., alias="탄수화물(g)")
    protein: float = Field(..., alias="단백질(g)")
    fat: float = Field(..., alias="지방(g)")
    vitamin_a: float = Field(..., alias="비타민 A(μg RAE)")
    thiamine: float = Field(..., alias="티아민(mg)")
    riboflavin: float = Field(..., alias="리보플라빈(mg)")
    vitamin_c: float = Field(..., alias="비타민 C(mg)")
    calcium: float = Field(..., alias="칼슘(mg)")
    iron: float = Field(..., alias="철(mg)")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "탄수화물(g)": 10.5,
                "단백질(g)": 20.0,
                "지방(g)": 5.0,
                "비타민 A(μg RAE)": 0.0,
                "티아민(mg)": 0.1,
                "리보플라빈(mg)": 0.2,
                "비타민 C(mg)": 10.0,
                "칼슘(mg)": 100.0,
                "철(mg)": 1.5
            }
        }

@app.get("/")
def read_root():
    return {"message": "K-Means Clustering API is running via FastAPI"}

@app.post("/predict")
def predict_cluster(data: NutrientInput):
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # 1. 데이터를 scaler가 학습된 순서대로 리스트 변환
    # 순서: 탄수화물, 단백질, 지방, 비타민A, 티아민, 리보플라빈, 비타민C, 칼슘, 철
    features = [
        data.carbs,
        data.protein,
        data.fat,
        data.vitamin_a,
        data.thiamine,
        data.riboflavin,
        data.vitamin_c,
        data.calcium,
        data.iron
    ]

    # 2. 2차원 배열로 변환 및 스케일링
    input_array = np.array([features])
    scaled_data = scaler.transform(input_array)

    # 3. 예측
    prediction = model.predict(scaled_data)
    cluster_id = int(prediction[0])

    return {
        "cluster": cluster_id,
        "message": f"입력된 데이터는 클러스터 {cluster_id}에 속합니다."
    }

if __name__ == "__main__":
    import uvicorn
    # 로컬 테스트용 (Railway에서는 아래 Procfile 설정이 사용됨)
    uvicorn.run(app, host="0.0.0.0", port=8000)