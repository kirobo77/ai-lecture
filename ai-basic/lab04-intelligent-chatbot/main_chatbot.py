"""
Lab 4 - Main Intelligent Chatbot System
지능형 API 라우팅 챗봇 - RAG + Multi-Agent + MCP 통합 시스템
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.config import validate_api_keys
from shared.utils import ChromaUtils
from agents.intent_classifier import IntentClassifier
from agents.weather_agent import WeatherAgent
from agents.calendar_agent import CalendarAgent
from agents.file_agent import FileAgent
from agents.notification_agent import NotificationAgent

# MCP 시스템 import
from mcp_layer.mcp_client import MCPOrchestrator
from mcp_layer.api_connectors import initialize_default_apis, get_api_stats

# RAG 시스템 import
from rag_system.knowledge_base import KnowledgeBase, initialize_default_knowledge
from rag_system.context_manager import ContextManager, get_or_create_session, add_conversation

from datetime import datetime
from typing import Dict, List, Optional, Union
import json
import asyncio
import time

class IntelligentChatbot:
    """지능형 API 라우팅 챗봇 메인 시스템 - RAG + Multi-Agent + MCP 통합"""
    
    def __init__(self, enable_mcp: bool = True, enable_rag: bool = True):
        """챗봇 시스템 초기화"""
        self.name = "Intelligent API Routing Chatbot"
        self.version = "2.0.0"  # MCP + RAG 통합 버전
        self.enable_mcp = enable_mcp
        self.enable_rag = enable_rag
        
        print(f"🚀 {self.name} v{self.version} 초기화 중...")
        print(f"   📡 MCP 시스템: {'활성화' if enable_mcp else '비활성화'}")
        print(f"   🧠 RAG 시스템: {'활성화' if enable_rag else '비활성화'}")
        
        # 에이전트들 초기화
        self.intent_classifier = IntentClassifier()
        self.weather_agent = WeatherAgent()
        self.calendar_agent = CalendarAgent()
        self.file_agent = FileAgent()
        self.notification_agent = NotificationAgent()
        
        # MCP 시스템 초기화
        self.mcp_orchestrator = None
        if self.enable_mcp:
            print("🔌 MCP 시스템 초기화 중...")
            self.mcp_orchestrator = MCPOrchestrator()
        
        # RAG 시스템 초기화
        self.knowledge_base = None
        self.context_manager = None
        if self.enable_rag:
            print("🧠 RAG 시스템 초기화 중...")
            try:
                self.knowledge_base = initialize_default_knowledge()
                self.context_manager = ContextManager()
                print("   ✅ RAG 지식베이스 로드 완료")
                print("   ✅ 컨텍스트 관리자 초기화 완료")
            except Exception as e:
                print(f"   ⚠️ RAG 시스템 초기화 실패: {e}")
                self.enable_rag = False
        
        # 세션 관리
        self.current_session_id = None
        
        # 대화 이력 (레거시 호환성)
        self.conversation_history = []
        
        # 시스템 상태
        self.system_status = {
            "initialized_at": datetime.now().isoformat(),
            "version": self.version,
            "mcp_enabled": self.enable_mcp,
            "rag_enabled": self.enable_rag,
            "total_conversations": 0,
            "successful_conversations": 0,
            "agent_usage": {
                "weather": 0,
                "calendar": 0,
                "file": 0,
                "notification": 0
            }
        }
        
        print(f"✅ {self.name} 초기화 완료!")
        self.print_system_info()
    
    def print_system_info(self):
        """시스템 정보 출력"""
        print(f"\n📋 시스템 정보:")
        print(f"  💬 챗봇 이름: {self.name}")
        print(f"  🔖 버전: {self.version}")
        print(f"  🧠 Intent Classifier: 준비됨")
        print(f"  🌤️  Weather Agent: 준비됨")
        print(f"  📅 Calendar Agent: 준비됨")
        print(f"  📁 File Agent: 준비됨")
        print(f"  🔔 Notification Agent: 준비됨")
        print(f"  ⏰ 초기화 시간: {self.system_status['initialized_at']}")
    
    def process_message(self, user_input: str, session_id: Optional[str] = None, context: Optional[Dict] = None) -> Dict:
        """RAG + MCP 통합 메시지 처리"""
        try:
            start_time = time.time()
            
            # 세션 관리
            if not session_id and self.enable_rag:
                session_id = get_or_create_session()
                self.current_session_id = session_id
            elif session_id:
                self.current_session_id = session_id
            
            conversation_id = f"conv_{int(time.time())}_{len(self.conversation_history)}"
            
            print(f"\n🔄 메시지 처리 시작: '{user_input}' (ID: {conversation_id})")
            if session_id:
                print(f"   📝 세션 ID: {session_id}")
            
            # RAG: 컨텍스트 정보 수집
            rag_context = {}
            if self.enable_rag and self.knowledge_base:
                print("  🧠 RAG 컨텍스트 수집 중...")
                
                # 과거 대화 패턴 검색
                similar_patterns = self.knowledge_base.search_intent_patterns(user_input, top_k=3)
                
                # 도메인 지식 검색
                domain_knowledge = self.knowledge_base.search_domain_knowledge(user_input, top_k=3)
                
                # 세션 컨텍스트 (있는 경우)
                session_context = {}
                if session_id and self.context_manager:
                    session_context = self.context_manager.get_conversation_context(session_id)
                
                rag_context = {
                    "similar_patterns": similar_patterns,
                    "domain_knowledge": domain_knowledge,
                    "session_context": session_context
                }
                
                print(f"     ✅ 패턴 {len(similar_patterns)}개, 지식 {len(domain_knowledge)}개 수집")
            
            # 1. 의도 분석 (RAG 강화)
            print("  1️⃣ 의도 분석 중...")
            enhanced_context = {**(context or {}), "rag_context": rag_context}
            intent_result = self.intent_classifier.analyze_intent(user_input, enhanced_context)
            
            intent = intent_result.get("intent", "unknown")
            apis = intent_result.get("apis", [])
            confidence = intent_result.get("confidence", 0.0)
            parameters = intent_result.get("parameters", {})
            
            print(f"     ✅ 의도: {intent} (신뢰도: {confidence:.2f})")
            print(f"     📋 필요 API: {apis}")
            
            # MCP: 도구 실행 준비
            mcp_results = []
            if self.enable_mcp and self.mcp_orchestrator and apis:
                print("  📡 MCP 도구 실행 중...")
                
                # MCP 초기화 (아직 안 된 경우)
                if not self.mcp_orchestrator.initialized:
                    try:
                        self.mcp_orchestrator.initialize()
                    except Exception as e:
                        print(f"     ⚠️ MCP 초기화 실패: {e}")
                
                # 각 API를 MCP 도구로 실행
                for api in apis:
                    if api in ["weather", "calendar", "file", "notification"]:
                        try:
                            mcp_intent = f"{api}_{'create' if 'create' in intent else 'query'}"
                            
                            # 동기 MCP 실행
                            mcp_result = self.mcp_orchestrator.execute_intent(mcp_intent, parameters)
                            
                            mcp_results.append({
                                "api": api,
                                "mcp_result": mcp_result,
                                "success": mcp_result.get("success", False)
                            })
                            print(f"     📡 {api} MCP 실행: {'✅' if mcp_result.get('success') else '❌'}")
                        except Exception as e:
                            print(f"     📡 {api} MCP 실행 실패: {e}")
                            mcp_results.append({
                                "api": api,
                                "mcp_result": {"success": False, "error": str(e)},
                                "success": False
                            })
            
            # 2. 에이전트 라우팅 및 실행 (기존 방식 유지)
            print("  2️⃣ 에이전트 실행 중...")
            agent_results = []
            
            if not apis:
                # 일반 대화 처리
                response = self.handle_general_conversation(intent, user_input)
                agent_results.append({
                    "agent": "General",
                    "success": True,
                    "response": response
                })
            else:
                # API 기반 작업 처리
                agent_results = self.execute_agents(apis, parameters, intent_result)
            
            # 3. 응답 통합 (MCP 결과 포함)
            print("  3️⃣ 응답 통합 중...")
            final_response = self.combine_agent_responses(agent_results, intent, user_input)
            
            # MCP 결과가 더 좋으면 사용
            if mcp_results and any(r.get("success") for r in mcp_results):
                mcp_response_parts = []
                for mcp_result in mcp_results:
                    if mcp_result.get("success"):
                        api_name = mcp_result["api"]
                        result_data = mcp_result["mcp_result"].get("result", {})
                        
                        # 결과 데이터에서 실제 내용 추출
                        if isinstance(result_data, dict):
                            if "data" in result_data:
                                data = result_data["data"]
                            elif "response" in result_data:
                                data = result_data["response"]
                            else:
                                data = str(result_data)
                        else:
                            data = str(result_data)
                        
                        mcp_response_parts.append(f"[MCP-{api_name}] {data}")
                
                if mcp_response_parts:
                    final_response = f"{final_response}\n\n🔗 MCP 시스템 결과:\n" + "\n".join(mcp_response_parts)
            
            # 4. 대화 이력 저장 (RAG + 레거시)
            processing_time = time.time() - start_time
            conversation_record = {
                "id": conversation_id,
                "session_id": session_id,
                "user_input": user_input,
                "intent": intent,
                "confidence": confidence,
                "apis_used": apis,
                "agent_results": agent_results,
                "mcp_results": mcp_results,
                "final_response": final_response,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "success": any(r.get("success", False) for r in agent_results) if agent_results else True,
                "rag_enabled": self.enable_rag,
                "mcp_enabled": self.enable_mcp
            }
            
            # RAG: 대화 기록 저장
            if self.enable_rag and session_id and self.knowledge_base:
                try:
                    # 컨텍스트 매니저에 대화 기록
                    if self.context_manager:
                        add_conversation(
                            session_id, user_input, final_response,
                            intent, confidence, apis,
                            conversation_record["success"], processing_time,
                            {"conversation_id": conversation_id}
                        )
                    
                    # 지식베이스에 대화 기록
                    # 메타데이터 안전 처리 (리스트는 JSON 문자열로 변환)
                    safe_metadata = {
                        "agent_count": len(agent_results),
                        "mcp_count": len(mcp_results),
                        "processing_time": processing_time,
                        "rag_enabled": self.enable_rag,
                        "mcp_enabled": self.enable_mcp
                    }
                    
                    # 성공한 에이전트만 요약
                    successful_agents = [r.get("agent", "Unknown") for r in agent_results if r.get("success")]
                    if successful_agents:
                        safe_metadata["successful_agents"] = ", ".join(successful_agents)
                    
                    # 성공한 MCP 결과만 요약  
                    successful_mcp = [r.get("api", "Unknown") for r in mcp_results if r.get("success")]
                    if successful_mcp:
                        safe_metadata["successful_mcp"] = ", ".join(successful_mcp)
                    
                    self.knowledge_base.add_conversation_record(
                        user_input, final_response, intent,
                        conversation_record["success"],
                        safe_metadata
                    )
                    
                    print(f"     💾 RAG 시스템에 대화 기록 저장 완료")
                except Exception as e:
                    print(f"     ⚠️ RAG 대화 기록 저장 실패: {e}")
            
            # 레거시 대화 이력 저장
            self.conversation_history.append(conversation_record)
            self.update_system_stats(conversation_record)
            
            print(f"  ✅ 처리 완료 ({processing_time:.2f}초)")
            
            return {
                "success": True,
                "conversation_id": conversation_id,
                "session_id": session_id,
                "intent": intent,
                "confidence": confidence,
                "response": final_response,
                "apis_used": apis,
                "processing_time": processing_time,
                "agent_results": agent_results,
                "mcp_results": mcp_results,
                "agents_used": apis,  # Streamlit 호환성
                "user_input": user_input  # 디버깅용
            }
            
        except Exception as e:
            error_msg = f"챗봇 처리 중 오류 발생: {str(e)}"
            print(f"  ❌ {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "response": "죄송합니다. 요청을 처리하는 중에 오류가 발생했습니다. 다시 시도해 주세요.",
                "user_input": user_input,
                "session_id": session_id
            }
    
    def process_user_input(self, user_input: str, context: Optional[Dict] = None) -> Dict:
        """레거시 호환성을 위한 메서드 (process_message로 리다이렉트)"""
        return self.process_message(user_input, context=context)
    
    def normalize_api_name(self, api_name: str) -> str:
        """API 이름을 정규화 (예: 'Weather API' -> 'weather')"""
        api_mapping = {
            "weather": "weather",
            "weather api": "weather", 
            "weather_api": "weather",
            "Weather API": "weather",
            "WEATHER_API": "weather",
            "날씨": "weather",
            "날씨 api": "weather",
            "날씨_api": "weather",
            "날씨 API": "weather",
            
            "calendar": "calendar",
            "calendar api": "calendar",
            "calendar_api": "calendar", 
            "Calendar API": "calendar",
            "CALENDAR_API": "calendar",
            "일정": "calendar",
            "일정 api": "calendar",
            "일정_api": "calendar",
            "일정 API": "calendar",
            "스케줄": "calendar",
            "스케줄 api": "calendar",
            
            "file": "file",
            "file api": "file",
            "file_api": "file",
            "File API": "file", 
            "FILE_API": "file",
            "file manager": "file",
            "file_manager": "file",
            "파일": "file",
            "파일 api": "file",
            "파일_api": "file",
            "파일 API": "file",
            "문서": "file",
            "문서 api": "file",
            "문서_api": "file",
            "문서 API": "file",
            "문서 검색": "file",
            "문서 검색 api": "file",
            "문서 검색_api": "file",
            "문서 검색 API": "file",
            "파일 검색": "file",
            "파일 검색 api": "file",
            "파일 검색 API": "file",
            
            "notification": "notification",
            "notification api": "notification",
            "notification_api": "notification",
            "Notification API": "notification",
            "NOTIFICATION_API": "notification",
            "알림": "notification",
            "알림 api": "notification",
            "알림_api": "notification",
            "알림 API": "notification",
            "메시지": "notification",
            "메시지 api": "notification",
            "메시지 API": "notification"
        }
        
        normalized = api_mapping.get(api_name.lower(), api_name.lower())
        return normalized

    def execute_agents(self, apis: List[str], parameters: Dict, intent_result: Dict) -> List[Dict]:
        """여러 에이전트 순차 실행 (Agent 간 데이터 전달 지원)"""
        agent_results = []
        collected_data = {}  # 이전 agent 결과 저장
        
        # API 이름 정규화
        normalized_apis = [self.normalize_api_name(api) for api in apis]
        print(f"     🔄 API 정규화: {apis} -> {normalized_apis}")
        
        # 에이전트 매핑
        agent_mapping = {
            "weather": self.weather_agent,
            "calendar": self.calendar_agent,
            "file": self.file_agent,
            "notification": self.notification_agent
        }
        
        for api in normalized_apis:
            if api in agent_mapping:
                try:
                    print(f"     🔧 {api.title()} Agent 실행 중...")
                    agent = agent_mapping[api]
                    
                    # 파라미터에 이전 agent 결과 추가
                    enhanced_parameters = parameters.copy()
                    enhanced_parameters["collected_data"] = collected_data
                    enhanced_parameters["intent"] = intent_result.get("intent", "")
                    enhanced_parameters["user_input"] = intent_result.get("user_input", "")
                    
                    # 에이전트별 파라미터 추출 및 호출
                    if api == "weather":
                        result = agent.process_weather_request(enhanced_parameters)
                        if result.get("success"):
                            # 날씨 정보를 collected_data에 저장
                            collected_data["weather_info"] = result.get("response", "")
                            collected_data["weather_raw"] = result.get("raw_data", {})
                            
                    elif api == "calendar":
                        result = agent.process_calendar_request(enhanced_parameters)
                        if result.get("success"):
                            # 일정 정보를 collected_data에 저장
                            collected_data["calendar_info"] = result.get("response", "")
                            collected_data["calendar_raw"] = result.get("raw_data", {})
                            
                    elif api == "file":
                        result = agent.process_file_request(enhanced_parameters)
                        if result.get("success"):
                            # 파일 정보를 collected_data에 저장
                            collected_data["file_info"] = result.get("response", "")
                            collected_data["file_raw"] = result.get("raw_data", {})
                            
                    elif api == "notification":
                        result = agent.process_notification_request(enhanced_parameters)
                    
                    agent_results.append(result)
                    
                    # 통계 업데이트
                    self.system_status["agent_usage"][api] += 1
                    
                    success_icon = "✅" if result.get("success") else "❌"
                    print(f"       {success_icon} {api.title()} Agent 완료")
                    
                except Exception as e:
                    error_result = {
                        "success": False,
                        "agent": f"{api.title()} Agent",
                        "error": str(e),
                        "response": f"{api} 처리 중 오류가 발생했습니다."
                    }
                    agent_results.append(error_result)
                    print(f"       ❌ {api.title()} Agent 오류: {str(e)}")
            else:
                print(f"       ⚠️ 알 수 없는 API: {api}")
        
        return agent_results
    
    def handle_general_conversation(self, intent: str, user_input: str) -> str:
        """일반 대화 처리"""
        responses = {
            "greeting": "안녕하세요! 저는 지능형 AI 어시스턴트입니다. 날씨, 일정, 파일 검색, 알림 발송 등을 도와드릴 수 있습니다. 무엇을 도와드릴까요?",
            "thanks": "천만에요! 언제든지 도움이 필요하시면 말씀해 주세요.",
            "help": """🤖 **AI 어시스턴트 도움말**

다음과 같은 기능을 제공합니다:

🌤️ **날씨 정보**
- "오늘 서울 날씨 어때?"
- "내일 비 올까?"
- "부산 일기예보 알려줘"

📅 **일정 관리**  
- "오늘 일정 확인해줘"
- "내일 3시에 회의 잡아줘"
- "이번 주 빈 시간 언제야?"

📁 **파일 검색**
- "프로젝트 문서 찾아줘"
- "API 명세서 어디 있어?"
- "코드 파일들 보여줘"

🔔 **알림 발송**
- "팀에게 알려줘"
- "슬랙에 메시지 보내줘"
- "이메일로 공지해줘"

💡 **복합 작업도 가능해요!**
- "날씨 확인하고 팀에게 알려줘"
- "일정 보고 회의실 예약해줘"

궁금한 점이 있으시면 언제든지 물어보세요!""",
            "capability_query": "저는 날씨 조회, 일정 관리, 파일 검색, 알림 발송을 할 수 있습니다. 또한 여러 작업을 동시에 처리할 수도 있어요!",
            "unknown": "죄송합니다. 요청을 정확히 이해하지 못했습니다. '도움말'이라고 말씀하시면 사용 가능한 기능을 안내해 드릴게요."
        }
        
        return responses.get(intent, responses["unknown"])
    
    def combine_agent_responses(self, agent_results: List[Dict], intent: str, user_input: str) -> str:
        """여러 에이전트 결과를 통합하여 최종 응답 생성"""
        if not agent_results:
            return self.handle_general_conversation(intent, user_input)
        
        successful_results = [r for r in agent_results if r.get("success", False)]
        failed_results = [r for r in agent_results if not r.get("success", False)]
        
        if not successful_results and failed_results:
            # 모든 에이전트가 실패한 경우
            error_messages = [r.get("response", "오류") for r in failed_results]
            return f"죄송합니다. 요청을 처리하는 중에 문제가 발생했습니다:\n\n" + "\n".join(error_messages)
        
        # 성공한 결과들을 통합
        response_parts = []
        
        for result in successful_results:
            agent_name = result.get("agent", "Agent")
            response = result.get("response", "")
            
            if response:
                # 에이전트별 구분선 추가
                response_parts.append(f"**{agent_name}**\n{response}")
        
        # 실패한 결과가 있으면 경고 메시지 추가 (상세한 에러 정보 포함)
        if failed_results:
            failed_agents = [r.get("agent", "Agent") for r in failed_results]
            warning = f"\n\n⚠️ 일부 기능에서 오류가 발생했습니다: {', '.join(failed_agents)}"
            
            # 디버깅용 상세 에러 정보 추가
            for result in failed_results:
                agent = result.get("agent", "Unknown Agent")
                error = result.get("error", "알 수 없는 오류")
                print(f"🔍 [DEBUG] {agent} 에러: {error}")
                
            response_parts.append(warning)
        
        # 복합 작업의 경우 요약 메시지 추가
        if len(successful_results) > 1:
            summary = f"\n\n📊 **작업 요약**\n✅ {len(successful_results)}개 작업 완료"
            if failed_results:
                summary += f", ❌ {len(failed_results)}개 작업 실패"
            response_parts.append(summary)
        
        return "\n\n" + "─" * 40 + "\n\n".join(response_parts)
    
    def update_system_stats(self, conversation_record: Dict):
        """시스템 통계 업데이트"""
        self.system_status["total_conversations"] += 1
        
        if conversation_record.get("success", False):
            self.system_status["successful_conversations"] += 1
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """대화 이력 조회"""
        return self.conversation_history[-limit:] if limit > 0 else self.conversation_history
    
    def get_system_stats(self) -> Dict:
        """시스템 통계 정보"""
        total_conversations = self.system_status["total_conversations"]
        success_rate = 0
        
        if total_conversations > 0:
            success_rate = (self.system_status["successful_conversations"] / total_conversations) * 100
        
        return {
            "name": self.name,
            "version": self.version,
            "uptime": datetime.now().isoformat(),
            "total_conversations": total_conversations,
            "successful_conversations": self.system_status["successful_conversations"],
            "success_rate": f"{success_rate:.1f}%",
            "agent_usage": self.system_status["agent_usage"],
            "most_used_agent": max(self.system_status["agent_usage"], key=self.system_status["agent_usage"].get) if any(self.system_status["agent_usage"].values()) else "None"
        }
    
    def clear_conversation_history(self):
        """대화 이력 초기화"""
        self.conversation_history.clear()
        print("대화 이력이 초기화되었습니다.")
    
    def get_capabilities(self) -> Dict:
        """챗봇 전체 능력 정보"""
        return {
            "name": self.name,
            "version": self.version,
            "description": "지능형 API 라우팅 기반 멀티 에이전트 챗봇 시스템",
            "features": [
                "자연어 의도 분석 (RAG 기반)",
                "지능형 API 라우팅",
                "다중 에이전트 협력",
                "실시간 날씨 정보",
                "스마트 일정 관리",
                "고급 파일 검색",
                "멀티채널 알림 발송",
                "복합 작업 처리",
                "대화 이력 관리"
            ],
            "agents": {
                "Intent Classifier": self.intent_classifier.get_capabilities(),
                "Weather Agent": self.weather_agent.get_capabilities(),
                "Calendar Agent": self.calendar_agent.get_capabilities(),
                "File Agent": self.file_agent.get_capabilities(),
                "Notification Agent": self.notification_agent.get_capabilities()
            }
        }

def run_interactive_chat():
    """대화형 챗봇 실행"""
    print("=" * 80)
    print("🤖 지능형 API 라우팅 챗봇 시작")
    print("=" * 80)
    
    # API 키 검증
    if not validate_api_keys():
        print("❌ API 키 설정을 확인해주세요.")
        return
    
    # 챗봇 시스템 초기화
    chatbot = IntelligentChatbot()
    
    print(f"\n💬 채팅을 시작합니다! (종료: 'quit', 'exit', '종료')")
    print(f"💡 도움말을 보려면 '도움말' 또는 'help'를 입력하세요.")
    print("─" * 80)
    
    conversation_count = 0
    
    while True:
        try:
            # 사용자 입력 받기
            user_input = input(f"\n👤 You: ").strip()
            
            # 종료 명령 확인
            if user_input.lower() in ['quit', 'exit', '종료', 'q']:
                print(f"\n👋 챗봇을 종료합니다. 총 {conversation_count}번의 대화를 나누었습니다.")
                
                # 시스템 통계 출력
                stats = chatbot.get_system_stats()
                print(f"\n📊 세션 통계:")
                print(f"  💬 총 대화: {stats['total_conversations']}회")
                print(f"  ✅ 성공률: {stats['success_rate']}")
                print(f"  🏆 가장 많이 사용된 에이전트: {stats['most_used_agent']}")
                break
            
            if not user_input:
                print("❓ 메시지를 입력해주세요.")
                continue
            
            # 특별 명령 처리
            if user_input.lower() in ['stats', '통계', '상태']:
                stats = chatbot.get_system_stats()
                print(f"\n📊 시스템 통계:")
                for key, value in stats.items():
                    if key != 'agent_usage':
                        print(f"  {key}: {value}")
                print(f"  에이전트 사용량: {stats['agent_usage']}")
                continue
            
            if user_input.lower() in ['history', '이력']:
                history = chatbot.get_conversation_history(5)
                print(f"\n📋 최근 대화 이력 ({len(history)}개):")
                for i, conv in enumerate(history, 1):
                    print(f"  {i}. [{conv['timestamp'][:19]}] {conv['user_input'][:50]}...")
                continue
            
            # 챗봇 처리
            result = chatbot.process_user_input(user_input)
            
            if result.get("success"):
                response = result["response"]
                conversation_count += 1
                
                # 응답 출력
                print(f"\n🤖 Assistant: {response}")
                
                # 처리 정보 출력 (디버그 모드)
                if result.get("processing_time", 0) > 2:  # 2초 이상 걸린 경우만
                    print(f"\n⏱️ 처리 시간: {result['processing_time']:.2f}초")
                
            else:
                print(f"\n❌ 오류: {result.get('response', '알 수 없는 오류')}")
        
        except KeyboardInterrupt:
            print(f"\n\n👋 챗봇을 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 예상치 못한 오류: {str(e)}")

def run_demo_scenarios():
    """데모 시나리오 실행"""
    print("=" * 80)
    print("🎬 지능형 챗봇 데모 시나리오")
    print("=" * 80)
    
    # API 키 검증
    if not validate_api_keys():
        print("❌ API 키 설정을 확인해주세요.")
        return
    
    # 챗봇 시스템 초기화
    chatbot = IntelligentChatbot()
    
    # 데모 시나리오들
    demo_scenarios = [
        {
            "name": "🌤️ 날씨 조회",
            "input": "오늘 서울 날씨 어때?",
            "description": "단일 에이전트 호출 - Weather Agent"
        },
        {
            "name": "📅 일정 확인",
            "input": "내일 일정 확인해줘",
            "description": "단일 에이전트 호출 - Calendar Agent"
        },
        {
            "name": "📁 파일 검색",
            "input": "프로젝트 문서 찾아줘",
            "description": "단일 에이전트 호출 - File Agent"
        },
        {
            "name": "🔔 알림 발송",
            "input": "팀에게 슬랙으로 메시지 보내줘",
            "description": "단일 에이전트 호출 - Notification Agent"
        },
        {
            "name": "🔥 복합 작업",
            "input": "날씨 확인하고 일정 보고 팀에게 알려줘",
            "description": "다중 에이전트 협력 - Weather + Calendar + Notification"
        },
        {
            "name": "❓ 도움말",
            "input": "도움말",
            "description": "일반 대화 처리 - 시스템 정보"
        }
    ]
    
    print(f"총 {len(demo_scenarios)}개의 시나리오를 실행합니다:\n")
    
    for i, scenario in enumerate(demo_scenarios, 1):
        print(f"📋 시나리오 {i}: {scenario['name']}")
        print(f"   설명: {scenario['description']}")
        print(f"   입력: \"{scenario['input']}\"")
        print(f"   {'─' * 60}")
        
        # 챗봇 처리
        result = chatbot.process_user_input(scenario['input'])
        
        if result.get("success"):
            print(f"   응답: {result['response'][:200]}...")
            print(f"   ✅ 성공 (처리시간: {result.get('processing_time', 0):.2f}초)")
        else:
            print(f"   ❌ 실패: {result.get('response', '알 수 없는 오류')}")
        
        print(f"   {'─' * 60}\n")
        
        # 다음 시나리오 전에 잠시 대기
        time.sleep(1)
    
    # 최종 통계
    stats = chatbot.get_system_stats()
    print(f"🎯 데모 완료!")
    print(f"📊 최종 통계:")
    print(f"   💬 총 대화: {stats['total_conversations']}회")
    print(f"   ✅ 성공률: {stats['success_rate']}")
    print(f"   🏆 가장 많이 사용된 에이전트: {stats['most_used_agent']}")
    print(f"   📈 에이전트 사용량: {stats['agent_usage']}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="지능형 API 라우팅 챗봇")
    parser.add_argument("--mode", choices=["interactive", "demo"], default="interactive", 
                       help="실행 모드 선택 (interactive: 대화형, demo: 데모 시나리오)")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        run_demo_scenarios()
    else:
        run_interactive_chat() 