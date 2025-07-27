"""
Lab 4 - Weather Agent
Weather API 전문 호출 및 날씨 정보 처리 에이전트
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import httpx
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import json
import time

class WeatherAgent:
    """Weather API 전문 호출 에이전트"""
    
    def __init__(self, api_base_url: str = "http://localhost:8001"):
        """Weather Agent 초기화"""
        self.name = "Weather Agent"
        self.version = "1.0.0"
        self.api_base_url = api_base_url.rstrip('/')
        
        # HTTP 클라이언트 설정
        self.client = httpx.Client(timeout=30.0)
        
        # 캐시 설정 (간단한 메모리 캐시)
        self.cache = {}
        self.cache_ttl = 300  # 5분 캐시
        
        # 도시명 정규화 매핑
        self.city_mapping = {
            '서울': 'seoul',
            '부산': 'busan', 
            '인천': 'incheon',
            '대구': 'daegu',
            '광주': 'gwangju',
            'seoul': 'seoul',
            'busan': 'busan',
            'incheon': 'incheon',
            'daegu': 'daegu',
            'gwangju': 'gwangju'
        }
        
        print(f"{self.name} 초기화 완료 (API: {self.api_base_url})")
    
    def normalize_city(self, city: str) -> str:
        """도시명 정규화"""
        city_lower = city.lower().strip()
        
        # 직접 매핑
        if city_lower in self.city_mapping:
            return self.city_mapping[city_lower]
        
        # 부분 매칭
        for korean, english in self.city_mapping.items():
            if korean in city or city in korean:
                return english
        
        # 기본값: 입력 그대로 (API에서 처리)
        return city_lower
    
    def get_cache_key(self, endpoint: str, params: Dict = None) -> str:
        """캐시 키 생성"""
        if params:
            param_str = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
            return f"{endpoint}?{param_str}"
        return endpoint
    
    def get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """캐시에서 데이터 조회"""
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return data
            else:
                # 만료된 캐시 삭제
                del self.cache[cache_key]
        return None
    
    def set_cache(self, cache_key: str, data: Dict):
        """캐시에 데이터 저장"""
        self.cache[cache_key] = (data, time.time())
    
    def get_current_weather(self, city: str) -> Dict:
        """현재 날씨 정보 조회"""
        try:
            # 도시명 정규화
            normalized_city = self.normalize_city(city)
            
            # 캐시 확인
            cache_key = self.get_cache_key(f"weather/{normalized_city}")
            cached_data = self.get_from_cache(cache_key)
            if cached_data:
                print(f"캐시에서 날씨 정보 반환: {normalized_city}")
                return {
                    "success": True,
                    "data": cached_data,
                    "source": "cache",
                    "city": city,
                    "normalized_city": normalized_city
                }
            
            # API 호출
            response = self.client.get(f"{self.api_base_url}/weather/{normalized_city}")
            
            if response.status_code == 200:
                weather_data = response.json()
                
                # 캐시에 저장
                self.set_cache(cache_key, weather_data)
                
                return {
                    "success": True,
                    "data": weather_data,
                    "source": "api",
                    "city": city,
                    "normalized_city": normalized_city
                }
            else:
                return {
                    "success": False,
                    "error": f"API 호출 실패: {response.status_code}",
                    "city": city,
                    "normalized_city": normalized_city
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"날씨 정보 조회 실패: {str(e)}",
                "city": city
            }
    
    def get_weather_forecast(self, city: str, days: int = 3) -> Dict:
        """일기 예보 조회"""
        try:
            normalized_city = self.normalize_city(city)
            
            # 캐시 확인
            cache_key = self.get_cache_key(f"weather/{normalized_city}/forecast", {"days": days})
            cached_data = self.get_from_cache(cache_key)
            if cached_data:
                print(f"캐시에서 예보 정보 반환: {normalized_city}")
                return {
                    "success": True,
                    "data": cached_data,
                    "source": "cache",
                    "city": city,
                    "days": days
                }
            
            # API 호출
            response = self.client.get(
                f"{self.api_base_url}/weather/{normalized_city}/forecast",
                params={"days": days}
            )
            
            if response.status_code == 200:
                forecast_data = response.json()
                
                # 캐시에 저장
                self.set_cache(cache_key, forecast_data)
                
                return {
                    "success": True,
                    "data": forecast_data,
                    "source": "api",
                    "city": city,
                    "days": days
                }
            else:
                return {
                    "success": False,
                    "error": f"예보 조회 실패: {response.status_code}",
                    "city": city,
                    "days": days
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"예보 조회 실패: {str(e)}",
                "city": city,
                "days": days
            }
    
    def get_simple_weather(self, city: str) -> Dict:
        """간단한 날씨 정보 조회"""
        try:
            normalized_city = self.normalize_city(city)
            
            response = self.client.get(f"{self.api_base_url}/weather/current/{normalized_city}")
            
            if response.status_code == 200:
                simple_data = response.json()
                return {
                    "success": True,
                    "data": simple_data,
                    "city": city
                }
            else:
                return {
                    "success": False,
                    "error": f"간단 날씨 조회 실패: {response.status_code}",
                    "city": city
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"간단 날씨 조회 실패: {str(e)}",
                "city": city
            }
    
    def format_weather_response(self, weather_result: Dict, request_type: str = "current") -> str:
        """날씨 정보를 사용자 친화적 메시지로 변환"""
        if not weather_result.get("success"):
            error_msg = weather_result.get("error", "알 수 없는 오류")
            return f"죄송합니다. 날씨 정보를 가져올 수 없습니다. ({error_msg})"
        
        data = weather_result["data"]
        city = weather_result.get("city", "지역")
        
        if request_type == "current":
            # 현재 날씨 포맷
            temp = data.get("temperature", "정보없음")
            condition = data.get("condition", "정보없음")
            humidity = data.get("humidity", "정보없음")
            wind_speed = data.get("wind_speed", "정보없음")
            
            response = f"🌤️ {city} 현재 날씨\n"
            response += f"🌡️ 온도: {temp}°C\n"
            response += f"☁️ 날씨: {condition}\n"
            response += f"💧 습도: {humidity}%\n"
            response += f"💨 풍속: {wind_speed}km/h"
            
            # 캐시 정보 추가
            if weather_result.get("source") == "cache":
                response += "\n\n📱 캐시된 정보입니다 (최근 5분 이내)"
            
            return response
            
        elif request_type == "forecast":
            # 일기 예보 포맷
            if isinstance(data, list):
                response = f"📅 {city} {len(data)}일 예보\n\n"
                for i, forecast in enumerate(data, 1):
                    date = forecast.get("date", "날짜불명")
                    high = forecast.get("high_temp", "?")
                    low = forecast.get("low_temp", "?") 
                    condition = forecast.get("condition", "정보없음")
                    rain_chance = forecast.get("rain_chance", "?")
                    
                    response += f"📆 {date}\n"
                    response += f"   🌡️ {low}°C ~ {high}°C\n"
                    response += f"   ☁️ {condition}\n"
                    response += f"   🌧️ 강수확률: {rain_chance}%\n"
                    if i < len(data):
                        response += "\n"
                
                return response
            else:
                return f"{city} 예보 정보 형식이 올바르지 않습니다."
                
        elif request_type == "simple":
            # 간단한 포맷
            if "summary" in data:
                return f"🌤️ {data['summary']}"
            else:
                temp = data.get("temperature", "정보없음")
                condition = data.get("condition", "정보없음")
                return f"🌤️ {city}은 현재 {temp}, {condition}입니다."
        
        return "날씨 정보를 처리할 수 없습니다."
    
    def process_weather_request(self, parameters: Dict) -> Dict:
        """날씨 요청 처리 (Intent Classifier에서 호출)"""
        try:
            # 매개변수 추출
            city = parameters.get("city", parameters.get("location", "서울"))
            request_type = parameters.get("type", "current")  # current, forecast, simple
            days = parameters.get("days", 3)
            
            # 도시명이 None이거나 빈 문자열인 경우 기본값 설정
            if not city or city == "None":
                city = "서울"
                print(f"🔄 [자동 도시 설정] 기본값 '서울' 사용")
            
            print(f"날씨 요청 처리: {city} ({request_type})")
            
            # 요청 타입에 따른 처리
            if request_type == "forecast":
                result = self.get_weather_forecast(city, days)
            elif request_type == "simple":
                result = self.get_simple_weather(city)
            else:  # current (기본값)
                result = self.get_current_weather(city)
            
            # 응답 포맷팅
            formatted_response = self.format_weather_response(result, request_type)
            
            return {
                "success": result.get("success", False),
                "agent": self.name,
                "response": formatted_response,
                "raw_data": result,
                "processed_at": datetime.now().isoformat(),
                "cache_used": result.get("source") == "cache"
            }
            
        except Exception as e:
            return {
                "success": False,
                "agent": self.name,
                "response": f"날씨 정보 처리 중 오류가 발생했습니다: {str(e)}",
                "error": str(e),
                "processed_at": datetime.now().isoformat()
            }
    
    def get_weather_summary_for_notification(self, city: str) -> str:
        """알림용 간단한 날씨 요약"""
        result = self.get_simple_weather(city)
        if result.get("success"):
            data = result["data"]
            return data.get("summary", f"{city} 날씨 정보")
        else:
            return f"{city} 날씨 정보를 가져올 수 없습니다."
    
    def clear_cache(self):
        """캐시 초기화"""
        self.cache.clear()
        print("Weather Agent 캐시가 초기화되었습니다.")
    
    def get_cache_stats(self) -> Dict:
        """캐시 통계 정보"""
        current_time = time.time()
        valid_entries = 0
        expired_entries = 0
        
        for key, (data, timestamp) in self.cache.items():
            if current_time - timestamp < self.cache_ttl:
                valid_entries += 1
            else:
                expired_entries += 1
        
        return {
            "total_entries": len(self.cache),
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "cache_ttl_seconds": self.cache_ttl
        }
    
    def get_capabilities(self) -> Dict:
        """에이전트 능력 정보"""
        return {
            "name": self.name,
            "version": self.version,
            "description": "Weather API 전문 호출 및 날씨 정보 처리",
            "supported_operations": [
                "현재 날씨 조회",
                "일기 예보 조회 (최대 7일)",
                "간단한 날씨 요약",
                "도시명 정규화",
                "응답 캐싱"
            ],
            "supported_cities": list(self.city_mapping.keys()),
            "api_endpoint": self.api_base_url,
            "cache_ttl": f"{self.cache_ttl}초"
        }
    
    def __del__(self):
        """소멸자 - HTTP 클라이언트 정리"""
        try:
            self.client.close()
        except:
            pass

def test_weather_agent():
    """Weather Agent 테스트"""
    print("=" * 60)
    print("Weather Agent 테스트")
    print("=" * 60)
    
    # Weather Agent 초기화
    agent = WeatherAgent()
    
    # 테스트 케이스들
    test_cases = [
        {"city": "서울", "type": "current"},
        {"city": "부산", "type": "simple"},
        {"city": "Seoul", "type": "forecast", "days": 3},
        {"city": "인천", "type": "current"},
        {"city": "존재하지않는도시", "type": "current"}  # 에러 케이스
    ]
    
    print(f"\n{len(test_cases)}개 테스트 케이스로 Weather Agent 테스트:")
    print("-" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}] 테스트: {test_case}")
        
        # Weather Agent 호출
        result = agent.process_weather_request(test_case)
        
        print(f"성공 여부: {result['success']}")
        print(f"응답:\n{result['response']}")
        
        if result.get('cache_used'):
            print("📱 캐시 사용됨")
        
        if not result['success']:
            print(f"오류: {result.get('error', '알 수 없는 오류')}")
    
    # 캐시 통계 확인
    print(f"\n" + "=" * 60)
    print("캐시 통계:")
    cache_stats = agent.get_cache_stats()
    for key, value in cache_stats.items():
        print(f"  {key}: {value}")
    
    # 에이전트 능력 정보
    print(f"\n📋 Weather Agent 정보:")
    capabilities = agent.get_capabilities()
    print(f"  이름: {capabilities['name']}")
    print(f"  설명: {capabilities['description']}")
    print(f"  지원 도시: {len(capabilities['supported_cities'])}개")
    print(f"  API 엔드포인트: {capabilities['api_endpoint']}")
    
    print(f"\n" + "=" * 60)
    print("Weather Agent 테스트 완료!")

if __name__ == "__main__":
    test_weather_agent() 