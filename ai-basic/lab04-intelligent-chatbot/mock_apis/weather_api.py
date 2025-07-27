"""
Lab 4 - Mock Weather API Server
날씨 정보를 제공하는 Mock API 서버
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import random
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# FastAPI 앱 생성
app = FastAPI(
    title="Mock Weather API",
    description="날씨 정보를 제공하는 Mock API 서버",
    version="1.0.0"
)

# 데이터 모델 정의
class WeatherInfo(BaseModel):
    city: str
    temperature: int
    condition: str
    humidity: int
    wind_speed: int
    uv_index: int
    timestamp: str

class WeatherForecast(BaseModel):
    city: str
    date: str
    high_temp: int
    low_temp: int
    condition: str
    rain_chance: int

# Mock 데이터 로드
def load_weather_data():
    """날씨 데이터 로드"""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'weather_data.json')
    
    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # 기본 데이터
        return {
            'seoul': {'temp': 24, 'condition': '맑음', 'humidity': 65},
            'busan': {'temp': 26, 'condition': '흐림', 'humidity': 70},
            'incheon': {'temp': 23, 'condition': '비', 'humidity': 80},
            'daegu': {'temp': 28, 'condition': '맑음', 'humidity': 55},
            'gwangju': {'temp': 25, 'condition': '구름많음', 'humidity': 72}
        }

# 글로벌 데이터
WEATHER_DATA = load_weather_data()

# 날씨 조건 리스트
WEATHER_CONDITIONS = ['맑음', '흐림', '비', '눈', '구름많음', '안개', '뇌우']

@app.get("/")
async def root():
    """API 루트 엔드포인트"""
    return {
        "service": "Mock Weather API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/weather/{city}",
            "/weather/{city}/forecast",
            "/weather/current/{city}",
            "/cities"
        ]
    }

@app.get("/weather/{city}", response_model=WeatherInfo)
async def get_weather(city: str):
    """특정 도시의 현재 날씨 정보 조회"""
    city_lower = city.lower()
    
    if city_lower in WEATHER_DATA:
        base_data = WEATHER_DATA[city_lower]
        
        # 실시간 변화 시뮬레이션 (±2도 범위)
        temp_variation = random.randint(-2, 2)
        current_temp = base_data['temp'] + temp_variation
        
        return WeatherInfo(
            city=city,
            temperature=current_temp,
            condition=base_data['condition'],
            humidity=base_data['humidity'] + random.randint(-5, 5),
            wind_speed=random.randint(5, 25),
            uv_index=random.randint(1, 10),
            timestamp=datetime.now().isoformat()
        )
    else:
        # 알려지지 않은 도시는 랜덤 데이터 생성
        return WeatherInfo(
            city=city,
            temperature=random.randint(15, 35),
            condition=random.choice(WEATHER_CONDITIONS),
            humidity=random.randint(40, 90),
            wind_speed=random.randint(5, 25),
            uv_index=random.randint(1, 10),
            timestamp=datetime.now().isoformat()
        )

@app.get("/weather/{city}/forecast", response_model=List[WeatherForecast])
async def get_weather_forecast(city: str, days: int = 3):
    """특정 도시의 일기 예보 (기본 3일)"""
    if days > 7:
        raise HTTPException(status_code=400, detail="최대 7일까지만 예보 가능합니다")
    
    forecasts = []
    base_temp = WEATHER_DATA.get(city.lower(), {'temp': 25})['temp']
    
    for i in range(days):
        date = (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d')
        
        # 날씨 변화 시뮬레이션
        high_temp = base_temp + random.randint(-3, 3) + random.randint(3, 8)
        low_temp = high_temp - random.randint(5, 12)
        
        forecasts.append(WeatherForecast(
            city=city,
            date=date,
            high_temp=high_temp,
            low_temp=low_temp,
            condition=random.choice(WEATHER_CONDITIONS),
            rain_chance=random.randint(0, 100)
        ))
    
    return forecasts

@app.get("/weather/current/{city}")
async def get_current_conditions(city: str):
    """현재 날씨 상태 (간단한 형태)"""
    weather = await get_weather(city)
    
    return {
        "city": weather.city,
        "temperature": f"{weather.temperature}°C",
        "condition": weather.condition,
        "summary": f"{weather.city}은 현재 {weather.temperature}도, {weather.condition}입니다."
    }

@app.get("/cities")
async def get_available_cities():
    """지원하는 도시 목록"""
    return {
        "cities": list(WEATHER_DATA.keys()),
        "total": len(WEATHER_DATA),
        "note": "등록되지 않은 도시도 조회 가능합니다 (랜덤 데이터)"
    }

@app.post("/weather/{city}/update")
async def update_weather_data(city: str, weather: WeatherInfo):
    """날씨 데이터 업데이트 (테스트용)"""
    WEATHER_DATA[city.lower()] = {
        'temp': weather.temperature,
        'condition': weather.condition,
        'humidity': weather.humidity
    }
    
    return {
        "message": f"{city} 날씨 데이터가 업데이트되었습니다",
        "updated_data": WEATHER_DATA[city.lower()]
    }

# 서버 실행 함수
def run_server():
    """Weather API 서버 실행"""
    print("🌤️  Weather API 서버 시작 중...")
    print("📍 URL: http://localhost:8001")
    print("📖 API 문서: http://localhost:8001/docs")
    
    uvicorn.run(
        "weather_api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    run_server() 