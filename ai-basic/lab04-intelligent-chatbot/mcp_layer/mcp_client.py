"""
Lab 4 - MCP (Model Context Protocol) 클라이언트
표준화된 API 통신과 도구 연동을 위한 MCP 클라이언트 구현
"""

import httpx
import json
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

@dataclass
class MCPRequest:
    """MCP JSON-RPC 요청 메시지"""
    jsonrpc: str = "2.0"
    method: str = ""
    params: Optional[Dict] = None
    id: Optional[str] = None

@dataclass
class MCPResponse:
    """MCP JSON-RPC 응답 메시지"""
    jsonrpc: str = "2.0"
    result: Optional[Dict] = None
    error: Optional[Dict] = None
    id: Optional[str] = None

@dataclass
class MCPTool:
    """MCP 도구 정의"""
    name: str
    description: str
    input_schema: Dict
    output_schema: Optional[Dict] = None

@dataclass
class MCPResource:
    """MCP 리소스 정의"""
    uri: str
    name: str
    description: str
    mime_type: Optional[str] = None

class MCPClient:
    """MCP 클라이언트 - 동기 버전"""
    
    def __init__(self, client_name: str = "AI-Chatbot-Client"):
        self.client_name = client_name
        self.connected_servers: Dict[str, Dict] = {}
        self.available_tools: Dict[str, List[MCPTool]] = {}
        self.http_client = httpx.Client(timeout=30.0)
        self.executor = ThreadPoolExecutor(max_workers=4)
        print(f"MCP 클라이언트 초기화: {client_name} v1.0.0")
    
    def connect_to_server(self, server_name: str, base_url: str) -> bool:
        """서버에 연결 (동기 버전)"""
        try:
            print(f"MCP 서버 연결 시도: {server_name} ({base_url})")
            
            # 서버 상태 확인
            response = self.http_client.get(f"{base_url}/")
            if response.status_code != 200:
                print(f"❌ 서버 연결 실패: {server_name} (HTTP {response.status_code})")
                return False
            
            # 서버 정보 저장
            self.connected_servers[server_name] = {
                "base_url": base_url,
                "status": "connected",
                "last_check": "now"
            }
            
            # 도구 발견
            tools = self.map_api_endpoints_to_tools(server_name, base_url)
            self.available_tools[server_name] = tools
            
            print(f"🔧 {server_name} 도구 {len(tools)}개 발견")
            print(f"✅ MCP 서버 연결 완료: {server_name}")
            return True
            
        except Exception as e:
            print(f"❌ MCP 서버 연결 실패: {server_name} - {e}")
            return False
    
    def discover_server_capabilities(self, server_name: str, base_url: str):
        """서버 기능 발견 (동기 버전)"""
        try:
            # API 문서 확인
            docs_url = f"{base_url}/docs"
            response = self.http_client.get(docs_url)
            
            if response.status_code == 200:
                print(f"📖 {server_name} API 문서: {docs_url}")
            
            # 서버 정보 확인
            info_url = f"{base_url}/"
            response = self.http_client.get(info_url)
            
            if response.status_code == 200:
                server_info = response.json()
                print(f"📋 {server_name} 서버 정보: {server_info.get('service', 'Unknown')}")
                
        except Exception as e:
            print(f"⚠️ 서버 기능 발견 실패: {server_name} - {e}")
    
    def map_api_endpoints_to_tools(self, server_name: str, base_url: str) -> List[MCPTool]:
        """API 엔드포인트를 MCP 도구로 매핑 (동기 버전)"""
        tools = []
        
        if server_name == "weather_server":
            tools = [
                MCPTool(
                    name="get_current_weather",
                    description="현재 날씨 정보 조회",
                    input_schema={"type": "object", "properties": {"city": {"type": "string"}}}
                ),
                MCPTool(
                    name="get_weather_forecast",
                    description="날씨 예보 조회",
                    input_schema={"type": "object", "properties": {"city": {"type": "string"}}}
                )
            ]
        elif server_name == "calendar_server":
            tools = [
                MCPTool(
                    name="get_today_schedule",
                    description="오늘 일정 조회",
                    input_schema={"type": "object", "properties": {}}
                ),
                MCPTool(
                    name="create_event",
                    description="새 일정 생성",
                    input_schema={"type": "object", "properties": {
                        "title": {"type": "string"},
                        "start_time": {"type": "string"},
                        "end_time": {"type": "string"}
                    }}
                )
            ]
        elif server_name == "file_server":
            tools = [
                MCPTool(
                    name="search_files",
                    description="파일 검색",
                    input_schema={"type": "object", "properties": {"query": {"type": "string"}}}
                ),
                MCPTool(
                    name="get_file_content",
                    description="파일 내용 조회",
                    input_schema={"type": "object", "properties": {"file_id": {"type": "string"}}}
                )
            ]
        elif server_name == "notification_server":
            tools = [
                MCPTool(
                    name="send_slack_message",
                    description="Slack 메시지 전송",
                    input_schema={"type": "object", "properties": {
                        "message": {"type": "string"},
                        "channel": {"type": "string"}
                    }}
                ),
                MCPTool(
                    name="send_email",
                    description="이메일 전송",
                    input_schema={"type": "object", "properties": {
                        "to": {"type": "string"},
                        "subject": {"type": "string"},
                        "message": {"type": "string"}
                    }}
                )
            ]
        
        return tools
    
    def call_tool(self, tool_name: str, parameters: Dict) -> MCPResponse:
        """MCP 도구 호출 (동기 버전)"""
        try:
            # 도구가 속한 서버 찾기
            server_name = None
            for srv_name, tools in self.available_tools.items():
                if any(tool.name == tool_name for tool in tools):
                    server_name = srv_name
                    break
            
            if not server_name:
                return MCPResponse(
                    error={"code": -32601, "message": f"도구를 찾을 수 없음: {tool_name}"},
                    id=str(uuid.uuid4())
                )
            
            # MCP 요청 생성
            request = MCPRequest(
                method="tools/call",
                params={
                    "name": tool_name,
                    "arguments": parameters
                },
                id=str(uuid.uuid4())
            )
            
            # 실제 API 호출로 변환
            result = self.execute_tool_call(server_name, tool_name, parameters)
            
            return MCPResponse(
                result=result,
                id=request.id
            )
            
        except Exception as e:
            return MCPResponse(
                error={"code": -32603, "message": f"도구 호출 실패: {str(e)}"},
                id=str(uuid.uuid4())
            )
    
    def execute_tool_call(self, server_name: str, tool_name: str, parameters: Dict) -> Dict:
        """실제 API 호출 실행 (동기 버전)"""
        server_info = self.connected_servers.get(server_name)
        if not server_info:
            raise Exception(f"연결되지 않은 서버: {server_name}")
        
        base_url = server_info["base_url"]
        
        # 도구별 API 엔드포인트 매핑
        endpoint_mapping = {
            # Weather API
            "get_current_weather": f"{base_url}/weather/{{city}}",
            "get_weather_forecast": f"{base_url}/weather/{{city}}/forecast",
            
            # Calendar API  
            "get_today_schedule": f"{base_url}/calendar/today",
            "create_event": f"{base_url}/calendar/events",
            
            # File API
            "search_files": f"{base_url}/files/search",
            "get_file_content": f"{base_url}/files/{{file_id}}/content",
            
            # Notification API
            "send_slack_message": f"{base_url}/notifications/slack",
            "send_email": f"{base_url}/notifications/email"
        }
        
        endpoint = endpoint_mapping.get(tool_name)
        if not endpoint:
            raise Exception(f"알 수 없는 도구: {tool_name}")
        
        # URL 템플릿 처리
        for param, value in parameters.items():
            endpoint = endpoint.replace(f"{{{param}}}", str(value))
        
        # 도구별 파라미터 정규화
        normalized_params = {}
        if tool_name == "search_files":
            # File API는 'q' 파라미터만 사용
            if "query" in parameters:
                normalized_params["q"] = parameters["query"]
            elif "q" in parameters:
                normalized_params["q"] = parameters["q"]
        elif tool_name == "get_current_weather":
            # Weather API는 city 파라미터만 사용
            if "city" in parameters:
                normalized_params["city"] = parameters["city"]
        elif tool_name == "send_slack_message":
            # Slack API는 JSON body 사용 - SlackMessage 모델에 맞춤
            normalized_params = {
                "channel": parameters.get("channel", "general"),
                "text": parameters.get("message", parameters.get("text", "")),
                "username": parameters.get("username", "ChatBot"),
                "icon": parameters.get("icon", ":robot_face:")
            }
        elif tool_name == "send_email":
            # Email API는 JSON body 사용 - EmailMessage 모델에 맞춤
            normalized_params = {
                "to": parameters.get("to", ""),
                "subject": parameters.get("subject", "알림"),
                "body": parameters.get("message", parameters.get("body", ""))
            }
        else:
            # 기타 도구는 파라미터 그대로 사용
            normalized_params = parameters
        
        # HTTP 메서드 결정
        if tool_name in ["create_event", "send_slack_message", "send_email"]:
            response = self.http_client.post(endpoint, json=normalized_params)
        elif tool_name in ["search_files"]:
            response = self.http_client.get(endpoint, params=normalized_params)
        else:
            response = self.http_client.get(endpoint, params=normalized_params)
        
        if response.status_code == 200:
            return {
                "success": True,
                "data": response.json(),
                "tool_name": tool_name,
                "server": server_name
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
                "tool_name": tool_name,
                "server": server_name
            }
    
    def get_available_tools(self) -> Dict[str, List[MCPTool]]:
        """사용 가능한 도구 목록 반환"""
        return self.available_tools
    
    def get_tool_by_name(self, tool_name: str) -> Optional[MCPTool]:
        """이름으로 도구 찾기"""
        for tools in self.available_tools.values():
            for tool in tools:
                if tool.name == tool_name:
                    return tool
        return None
    
    def get_server_status(self) -> Dict:
        """서버 상태 정보 반환"""
        return {
            "connected_servers": len(self.connected_servers),
            "total_tools": sum(len(tools) for tools in self.available_tools.values()),
            "servers": list(self.connected_servers.keys())
        }
    
    def disconnect_all(self):
        """모든 연결 해제"""
        self.http_client.close()
        self.executor.shutdown(wait=True)
        print("MCP 클라이언트 연결 해제 완료")

class MCPOrchestrator:
    """MCP 오케스트레이터 - 동기 버전"""
    
    def __init__(self):
        self.mcp_client = MCPClient()
        self.initialized = False
    
    def initialize(self) -> bool:
        """MCP 시스템 초기화 (동기 버전)"""
        try:
            print("🔌 MCP 시스템 초기화 중...")
            servers = [
                ("weather_server", "http://localhost:8001"),
                ("calendar_server", "http://localhost:8002"),
                ("file_server", "http://localhost:8003"),
                ("notification_server", "http://localhost:8004")
            ]
            connected_count = 0
            for server_name, url in servers:
                if self.mcp_client.connect_to_server(server_name, url):
                    connected_count += 1
            self.initialized = connected_count > 0
            print(f"✅ MCP 초기화 완료: {connected_count}/{len(servers)} 서버 연결")
            return self.initialized
        except Exception as e:
            print(f"❌ MCP 초기화 실패: {e}")
            self.initialized = False
            return False
    
    def execute_intent(self, intent: str, parameters: Dict) -> Dict:
        """의도에 따른 MCP 도구 실행 (동기 버전)"""
        try:
            # 의도별 도구 매핑 (확장)
            intent_tool_mapping = {
                "weather_query": "get_current_weather",
                "weather_forecast": "get_weather_forecast",
                "weather_create": "get_current_weather",  # 폴백
                "calendar_query": "get_today_schedule",
                "calendar_create": "create_event",
                "file_query": "search_files",
                "file_search": "search_files",
                "file_create": "search_files",  # 폴백
                "notification_query": "send_slack_message",
                "notification_send": "send_slack_message",
                "notification_create": "send_slack_message"  # 폴백
            }
            
            tool_name = intent_tool_mapping.get(intent)
            if not tool_name:
                return {
                    "success": False,
                    "error": f"알 수 없는 의도: {intent}",
                    "tool_name": None
                }
            
            print(f"🔧 MCP: 도구 '{tool_name}' 실행 (의도: {intent})")
            
            # MCP 도구 호출
            response = self.mcp_client.call_tool(tool_name, parameters)
            
            if response.error:
                return {
                    "success": False,
                    "error": response.error.get("message", "알 수 없는 오류"),
                    "tool_name": tool_name
                }
            
            return {
                "success": True,
                "result": response.result,
                "tool_name": tool_name
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"도구 호출 실패: {str(e)}",
                "tool_name": None
            }
    
    def get_system_info(self) -> Dict:
        """시스템 정보 반환"""
        return {
            "initialized": self.initialized,
            "server_status": self.mcp_client.get_server_status(),
            "available_tools": len(self.mcp_client.get_available_tools())
        }

def test_mcp_client():
    """MCP 클라이언트 테스트 (동기 버전)"""
    print("🧪 MCP 클라이언트 테스트 시작")
    
    orchestrator = MCPOrchestrator()
    if orchestrator.initialize():
        print("✅ MCP 초기화 성공")
        
        # 테스트 도구 호출
        result = orchestrator.execute_intent("file_query", {"query": "test"})
        print(f"테스트 결과: {result}")
    else:
        print("❌ MCP 초기화 실패")
    
    print("🧪 MCP 클라이언트 테스트 완료")

if __name__ == "__main__":
    test_mcp_client() 