from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# 1. 모델과 스케일러 로드
try:
    model = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("모델 로드 완료")
except Exception as e:
    print(f"모델 로드 실패: {e}")
    model = None
    scaler = None

@app.route('/')
def home():
    return "K-Means Model API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json()
    
    # 2. 입력 데이터 순서 정의 (scaler.pkl에 저장된 feature 순서 준수)
    # 업로드된 scaler 파일 분석 결과 
    features = [
        data.get('탄수화물(g)', 0),
        data.get('단백질(g)', 0),
        data.get('지방(g)', 0),
        data.get('비타민 A(μg RAE)', 0),
        data.get('티아민(mg)', 0),
        data.get('리보플라빈(mg)', 0),
        data.get('비타민 C(mg)', 0),
        data.get('칼슘(mg)', 0),
        data.get('철(mg)', 0)
    ]

    # 3. 데이터 전처리 (2차원 배열 변환 및 스케일링)
    input_data = np.array([features])
    scaled_data = scaler.transform(input_data)

    # 4. 예측 (Cluster 번호 반환)
    prediction = model.predict(scaled_data)
    
    return jsonify({
        'cluster': int(prediction[0]),
        'message': f'This data belongs to cluster {prediction[0]}'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)