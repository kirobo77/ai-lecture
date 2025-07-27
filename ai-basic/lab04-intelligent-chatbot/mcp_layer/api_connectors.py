"""
Lab 4 - API 연결 관리자
MCP 레이어와 실제 API 서버들 간의 연결을 관리
"""

import asyncio
import httpx
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

class APIConnector:
    """개별 API 서버 연결 관리자"""
    
    def __init__(self, name: str, base_url: str, timeout: int = 30):
        """API 연결자 초기화"""
        self.name = name
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        
        # HTTP 클라이언트
        self.client = httpx.AsyncClient(timeout=timeout)
        
        # 연결 상태
        self.is_connected = False
        self.last_check = None
        self.response_times = []
        
        # 에러 통계
        self.error_count = 0
        self.total_requests = 0
        
        self.logger = logging.getLogger(f"APIConnector.{name}")
    
    async def check_connection(self) -> bool:
        """API 서버 연결 상태 확인"""
        try:
            start_time = datetime.now()
            response = await self.client.get(f"{self.base_url}/")
            
            # 응답 시간 기록
            response_time = (datetime.now() - start_time).total_seconds()
            self.response_times.append(response_time)
            
            # 최근 10개만 유지
            if len(self.response_times) > 10:
                self.response_times = self.response_times[-10:]
            
            self.is_connected = response.status_code == 200
            self.last_check = datetime.now()
            self.total_requests += 1
            
            if not self.is_connected:
                self.error_count += 1
                self.logger.warning(f"연결 실패: HTTP {response.status_code}")
            
            return self.is_connected
            
        except Exception as e:
            self.is_connected = False
            self.last_check = datetime.now()
            self.error_count += 1
            self.total_requests += 1
            self.logger.error(f"연결 오류: {str(e)}")
            return False
    
    async def make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """API 요청 실행"""
        try:
            start_time = datetime.now()
            
            # 전체 URL 구성
            url = f"{self.base_url}{endpoint}"
            
            # HTTP 메서드에 따른 요청
            if method.upper() == "GET":
                response = await self.client.get(url, **kwargs)
            elif method.upper() == "POST":
                response = await self.client.post(url, **kwargs)
            elif method.upper() == "PUT":
                response = await self.client.put(url, **kwargs)
            elif method.upper() == "DELETE":
                response = await self.client.delete(url, **kwargs)
            else:
                raise ValueError(f"지원하지 않는 HTTP 메서드: {method}")
            
            # 응답 시간 기록
            response_time = (datetime.now() - start_time).total_seconds()
            self.response_times.append(response_time)
            if len(self.response_times) > 10:
                self.response_times = self.response_times[-10:]
            
            self.total_requests += 1
            
            # 성공 응답
            if response.status_code in [200, 201]:
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "data": response.json() if response.content else {},
                    "response_time": response_time
                }
            else:
                self.error_count += 1
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "error": response.text,
                    "response_time": response_time
                }
                
        except Exception as e:
            self.error_count += 1
            self.total_requests += 1
            return {
                "success": False,
                "error": str(e),
                "response_time": 0
            }
    
    def get_stats(self) -> Dict:
        """연결 통계 정보"""
        avg_response_time = 0
        if self.response_times:
            avg_response_time = sum(self.response_times) / len(self.response_times)
        
        success_rate = 0
        if self.total_requests > 0:
            success_rate = ((self.total_requests - self.error_count) / self.total_requests) * 100
        
        return {
            "name": self.name,
            "base_url": self.base_url,
            "is_connected": self.is_connected,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "success_rate": round(success_rate, 2),
            "avg_response_time": round(avg_response_time, 3),
            "recent_response_times": self.response_times[-5:]  # 최근 5개
        }
    
    async def close(self):
        """연결 해제"""
        await self.client.aclose()

class APIConnectorManager:
    """API 연결 관리자"""
    
    def __init__(self):
        """API 연결 관리자 초기화"""
        self.connectors: Dict[str, APIConnector] = {}
        self.logger = logging.getLogger(__name__)
        self.health_check_interval = 60  # 60초마다 헬스 체크
        self.health_check_task = None
    
    def add_connector(self, name: str, base_url: str, timeout: int = 30):
        """새 API 연결자 추가"""
        connector = APIConnector(name, base_url, timeout)
        self.connectors[name] = connector
        print(f"API 연결자 추가: {name} ({base_url})")
    
    async def initialize_all(self) -> Dict[str, bool]:
        """모든 연결자 초기화"""
        results = {}
        
        print("🔌 API 연결자들 초기화 중...")
        
        for name, connector in self.connectors.items():
            result = await connector.check_connection()
            results[name] = result
            
            status = "✅ 연결됨" if result else "❌ 연결 실패"
            print(f"  {name}: {status}")
        
        # 자동 헬스 체크 시작
        if any(results.values()):
            await self.start_health_check()
        
        return results
    
    async def start_health_check(self):
        """자동 헬스 체크 시작"""
        if self.health_check_task:
            return
        
        async def health_check_loop():
            while True:
                try:
                    await asyncio.sleep(self.health_check_interval)
                    await self.check_all_connections()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"헬스 체크 오류: {e}")
        
        self.health_check_task = asyncio.create_task(health_check_loop())
        print(f"🔄 자동 헬스 체크 시작 ({self.health_check_interval}초 간격)")
    
    async def stop_health_check(self):
        """자동 헬스 체크 중지"""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
            self.health_check_task = None
            print("🔄 자동 헬스 체크 중지")
    
    async def check_all_connections(self) -> Dict[str, bool]:
        """모든 연결 상태 확인"""
        results = {}
        
        for name, connector in self.connectors.items():
            result = await connector.check_connection()
            results[name] = result
        
        # 연결 실패한 API가 있으면 로그
        failed_apis = [name for name, status in results.items() if not status]
        if failed_apis:
            self.logger.warning(f"연결 실패 API: {', '.join(failed_apis)}")
        
        return results
    
    async def make_request(self, api_name: str, method: str, endpoint: str, **kwargs) -> Dict:
        """특정 API에 요청"""
        if api_name not in self.connectors:
            return {
                "success": False,
                "error": f"알 수 없는 API: {api_name}"
            }
        
        connector = self.connectors[api_name]
        
        # 연결 상태 확인 (캐시된 상태 사용)
        if not connector.is_connected:
            # 재연결 시도
            await connector.check_connection()
            
            if not connector.is_connected:
                return {
                    "success": False,
                    "error": f"{api_name} API 서버에 연결할 수 없습니다"
                }
        
        # 요청 실행
        return await connector.make_request(method, endpoint, **kwargs)
    
    def get_connector(self, name: str) -> Optional[APIConnector]:
        """연결자 가져오기"""
        return self.connectors.get(name)
    
    def get_all_stats(self) -> Dict:
        """모든 연결자 통계"""
        stats = {}
        
        for name, connector in self.connectors.items():
            stats[name] = connector.get_stats()
        
        # 전체 요약
        total_requests = sum(stat["total_requests"] for stat in stats.values())
        total_errors = sum(stat["error_count"] for stat in stats.values())
        connected_count = sum(1 for stat in stats.values() if stat["is_connected"])
        
        overall_success_rate = 0
        if total_requests > 0:
            overall_success_rate = ((total_requests - total_errors) / total_requests) * 100
        
        return {
            "individual_stats": stats,
            "summary": {
                "total_apis": len(self.connectors),
                "connected_apis": connected_count,
                "total_requests": total_requests,
                "total_errors": total_errors,
                "overall_success_rate": round(overall_success_rate, 2),
                "health_check_interval": self.health_check_interval,
                "health_check_active": self.health_check_task is not None
            }
        }
    
    async def cleanup(self):
        """모든 연결 정리"""
        await self.stop_health_check()
        
        for connector in self.connectors.values():
            await connector.close()
        
        self.connectors.clear()
        print("모든 API 연결이 정리되었습니다.")

# 전역 API 연결 관리자 인스턴스
api_manager = APIConnectorManager()

async def initialize_default_apis():
    """기본 API들 초기화"""
    # 기본 API 서버들 추가
    default_apis = [
        ("weather", "http://localhost:8001"),
        ("calendar", "http://localhost:8002"),
        ("file_manager", "http://localhost:8003"),
        ("notification", "http://localhost:8004")
    ]
    
    for name, url in default_apis:
        api_manager.add_connector(name, url)
    
    # 모든 API 초기화
    results = await api_manager.initialize_all()
    
    return results

async def get_api_stats():
    """API 통계 정보 조회"""
    return api_manager.get_all_stats()

async def make_api_call(api_name: str, method: str, endpoint: str, **kwargs):
    """API 호출 헬퍼 함수"""
    return await api_manager.make_request(api_name, method, endpoint, **kwargs)

# 개별 API 호출 헬퍼 함수들
async def call_weather_api(endpoint: str, method: str = "GET", **kwargs):
    """날씨 API 호출"""
    return await make_api_call("weather", method, endpoint, **kwargs)

async def call_calendar_api(endpoint: str, method: str = "GET", **kwargs):
    """일정 API 호출"""
    return await make_api_call("calendar", method, endpoint, **kwargs)

async def call_file_api(endpoint: str, method: str = "GET", **kwargs):
    """파일 API 호출"""
    return await make_api_call("file_manager", method, endpoint, **kwargs)

async def call_notification_api(endpoint: str, method: str = "GET", **kwargs):
    """알림 API 호출"""
    return await make_api_call("notification", method, endpoint, **kwargs)

async def test_api_connectors():
    """API 연결자 테스트"""
    print("=" * 60)
    print("API 연결자 테스트")
    print("=" * 60)
    
    # 기본 API들 초기화
    results = await initialize_default_apis()
    
    if any(results.values()):
        print(f"\n🧪 API 호출 테스트:")
        
        # 테스트 호출들
        test_calls = [
            ("weather", "GET", "/weather/seoul"),
            ("calendar", "GET", "/calendar/today"),
            ("file_manager", "GET", "/files/search", {"params": {"q": "test"}}),
            ("notification", "POST", "/notifications/slack", {"json": {"channel": "#test", "text": "test"}})
        ]
        
        for api_name, method, endpoint, *extra_args in test_calls:
            kwargs = extra_args[0] if extra_args else {}
            
            print(f"\n📡 {api_name}: {method} {endpoint}")
            result = await make_api_call(api_name, method, endpoint, **kwargs)
            
            if result["success"]:
                print(f"   ✅ 성공 (응답시간: {result.get('response_time', 0):.3f}s)")
            else:
                print(f"   ❌ 실패: {result.get('error', '알 수 없는 오류')}")
        
        # 통계 정보 출력
        print(f"\n📊 API 통계:")
        stats = await get_api_stats()
        
        for api_name, stat in stats["individual_stats"].items():
            print(f"  {api_name}:")
            print(f"    연결: {'✅' if stat['is_connected'] else '❌'}")
            print(f"    요청: {stat['total_requests']}회")
            print(f"    성공률: {stat['success_rate']}%")
            print(f"    평균 응답시간: {stat['avg_response_time']}s")
    
    # 정리
    await api_manager.cleanup()
    print(f"\n" + "=" * 60)
    print("API 연결자 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(test_api_connectors()) 