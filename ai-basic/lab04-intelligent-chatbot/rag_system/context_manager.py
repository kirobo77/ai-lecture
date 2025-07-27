"""
Lab 4 - RAG 컨텍스트 관리자
대화 컨텍스트, 세션 관리, 메모리 최적화를 담당
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import threading
import time

@dataclass
class ConversationTurn:
    """단일 대화 턴"""
    turn_id: str
    user_input: str
    assistant_response: str
    intent: str
    confidence: float
    apis_used: List[str]
    success: bool
    timestamp: str
    processing_time: float
    metadata: Dict = None

@dataclass
class SessionContext:
    """세션 컨텍스트"""
    session_id: str
    user_id: Optional[str]
    created_at: str
    last_activity: str
    conversation_turns: List[ConversationTurn]
    user_preferences: Dict
    active_topics: List[str]
    session_metadata: Dict = None

class ContextManager:
    """대화 컨텍스트 관리자"""
    
    def __init__(self, max_history_length: int = 20, session_timeout_hours: int = 24):
        """컨텍스트 관리자 초기화"""
        self.max_history_length = max_history_length
        self.session_timeout_hours = session_timeout_hours
        
        # 활성 세션들
        self.active_sessions: Dict[str, SessionContext] = {}
        
        # 컨텍스트 윈도우 관리
        self.context_windows: Dict[str, deque] = {}
        
        # 스레드 안전성을 위한 락
        self.session_lock = threading.RLock()
        
        # 세션 정리 스레드
        self.cleanup_thread = None
        self.start_cleanup_service()
        
        print(f"컨텍스트 관리자 초기화: 최대 {max_history_length}턴, 세션 타임아웃 {session_timeout_hours}시간")
    
    def create_session(self, user_id: Optional[str] = None, 
                      session_metadata: Dict = None) -> str:
        """새 세션 생성"""
        with self.session_lock:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            session = SessionContext(
                session_id=session_id,
                user_id=user_id,
                created_at=datetime.now().isoformat(),
                last_activity=datetime.now().isoformat(),
                conversation_turns=[],
                user_preferences={},
                active_topics=[],
                session_metadata=session_metadata or {}
            )
            
            self.active_sessions[session_id] = session
            self.context_windows[session_id] = deque(maxlen=self.max_history_length)
            
            print(f"새 세션 생성: {session_id}")
            return session_id
    
    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """세션 조회"""
        with self.session_lock:
            return self.active_sessions.get(session_id)
    
    def update_session_activity(self, session_id: str):
        """세션 활동 시간 업데이트"""
        with self.session_lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id].last_activity = datetime.now().isoformat()
    
    def add_conversation_turn(self, session_id: str, user_input: str, 
                            assistant_response: str, intent: str, 
                            confidence: float, apis_used: List[str], 
                            success: bool, processing_time: float,
                            metadata: Dict = None) -> str:
        """대화 턴 추가"""
        with self.session_lock:
            if session_id not in self.active_sessions:
                # 세션이 없으면 새로 생성
                self.create_session()
            
            session = self.active_sessions[session_id]
            
            turn_id = f"turn_{len(session.conversation_turns)}_{uuid.uuid4().hex[:6]}"
            
            turn = ConversationTurn(
                turn_id=turn_id,
                user_input=user_input,
                assistant_response=assistant_response,
                intent=intent,
                confidence=confidence,
                apis_used=apis_used,
                success=success,
                timestamp=datetime.now().isoformat(),
                processing_time=processing_time,
                metadata=metadata or {}
            )
            
            # 세션에 턴 추가
            session.conversation_turns.append(turn)
            
            # 컨텍스트 윈도우에 추가
            context_window = self.context_windows[session_id]
            context_window.append(turn)
            
            # 세션 활동 시간 업데이트
            self.update_session_activity(session_id)
            
            # 활성 토픽 업데이트
            self.update_active_topics(session_id, intent, user_input)
            
            return turn_id
    
    def update_active_topics(self, session_id: str, intent: str, user_input: str):
        """활성 토픽 업데이트"""
        session = self.active_sessions.get(session_id)
        if not session:
            return
        
        # 토픽 키워드 추출 (간단한 휴리스틱)
        topic_keywords = {
            "weather": ["날씨", "기온", "비", "눈", "바람"],
            "calendar": ["일정", "회의", "약속", "미팅", "스케줄"],
            "file": ["파일", "문서", "자료", "보고서", "데이터"],
            "notification": ["알림", "메시지", "이메일", "슬랙", "공지"]
        }
        
        # 현재 의도를 활성 토픽에 추가
        if intent not in session.active_topics:
            session.active_topics.append(intent)
        
        # 키워드 기반 토픽 감지
        user_input_lower = user_input.lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                if topic not in session.active_topics:
                    session.active_topics.append(topic)
        
        # 최대 5개 토픽만 유지
        if len(session.active_topics) > 5:
            session.active_topics = session.active_topics[-5:]
    
    def get_conversation_context(self, session_id: str, 
                                include_metadata: bool = False) -> Dict:
        """대화 컨텍스트 조회"""
        with self.session_lock:
            session = self.active_sessions.get(session_id)
            if not session:
                return {}
            
            context_window = self.context_windows.get(session_id, deque())
            
            # 최근 대화 턴들
            recent_turns = []
            for turn in list(context_window):
                turn_data = {
                    "user_input": turn.user_input,
                    "assistant_response": turn.assistant_response,
                    "intent": turn.intent,
                    "success": turn.success,
                    "timestamp": turn.timestamp
                }
                
                if include_metadata:
                    turn_data["metadata"] = turn.metadata
                    turn_data["apis_used"] = turn.apis_used
                    turn_data["processing_time"] = turn.processing_time
                
                recent_turns.append(turn_data)
            
            return {
                "session_id": session_id,
                "user_id": session.user_id,
                "session_created": session.created_at,
                "last_activity": session.last_activity,
                "total_turns": len(session.conversation_turns),
                "recent_turns": recent_turns,
                "active_topics": session.active_topics,
                "user_preferences": session.user_preferences
            }
    
    def get_contextual_summary(self, session_id: str, 
                              last_n_turns: int = 5) -> str:
        """대화 컨텍스트 요약 생성"""
        context = self.get_conversation_context(session_id)
        if not context:
            return "새로운 대화입니다."
        
        recent_turns = context["recent_turns"][-last_n_turns:]
        active_topics = context["active_topics"]
        
        summary_parts = []
        
        # 활성 토픽
        if active_topics:
            topics_str = ", ".join(active_topics)
            summary_parts.append(f"현재 활성 토픽: {topics_str}")
        
        # 최근 대화 패턴
        if recent_turns:
            successful_turns = sum(1 for turn in recent_turns if turn["success"])
            success_rate = successful_turns / len(recent_turns) * 100
            
            summary_parts.append(f"최근 {len(recent_turns)}턴 중 {successful_turns}개 성공 ({success_rate:.0f}%)")
            
            # 최근 의도들
            recent_intents = [turn["intent"] for turn in recent_turns[-3:]]
            if recent_intents:
                summary_parts.append(f"최근 의도: {' → '.join(recent_intents)}")
        
        return "\n".join(summary_parts) if summary_parts else "대화 컨텍스트 없음"
    
    def update_user_preference(self, session_id: str, key: str, value: Any):
        """사용자 선호도 업데이트"""
        with self.session_lock:
            session = self.active_sessions.get(session_id)
            if session:
                session.user_preferences[key] = value
                self.update_session_activity(session_id)
    
    def get_user_preference(self, session_id: str, key: str, default: Any = None) -> Any:
        """사용자 선호도 조회"""
        session = self.active_sessions.get(session_id)
        if session:
            return session.user_preferences.get(key, default)
        return default
    
    def analyze_conversation_patterns(self, session_id: str) -> Dict:
        """대화 패턴 분석"""
        session = self.active_sessions.get(session_id)
        if not session:
            return {}
        
        turns = session.conversation_turns
        if not turns:
            return {}
        
        # 통계 계산
        total_turns = len(turns)
        successful_turns = sum(1 for turn in turns if turn.success)
        success_rate = successful_turns / total_turns * 100
        
        # 의도별 통계
        intent_counts = {}
        for turn in turns:
            intent_counts[turn.intent] = intent_counts.get(turn.intent, 0) + 1
        
        # API 사용 통계
        api_usage = {}
        for turn in turns:
            for api in turn.apis_used:
                api_usage[api] = api_usage.get(api, 0) + 1
        
        # 응답 시간 통계
        processing_times = [turn.processing_time for turn in turns]
        avg_processing_time = sum(processing_times) / len(processing_times)
        
        # 대화 길이 패턴
        user_input_lengths = [len(turn.user_input) for turn in turns]
        avg_input_length = sum(user_input_lengths) / len(user_input_lengths)
        
        return {
            "session_stats": {
                "total_turns": total_turns,
                "successful_turns": successful_turns,
                "success_rate": round(success_rate, 2),
                "avg_processing_time": round(avg_processing_time, 3),
                "avg_input_length": round(avg_input_length, 1)
            },
            "intent_distribution": intent_counts,
            "api_usage": api_usage,
            "active_topics": session.active_topics,
            "session_duration": self.calculate_session_duration(session)
        }
    
    def calculate_session_duration(self, session: SessionContext) -> str:
        """세션 지속 시간 계산"""
        created = datetime.fromisoformat(session.created_at)
        last_activity = datetime.fromisoformat(session.last_activity)
        duration = last_activity - created
        
        hours = duration.total_seconds() / 3600
        if hours < 1:
            minutes = duration.total_seconds() / 60
            return f"{minutes:.0f}분"
        else:
            return f"{hours:.1f}시간"
    
    def export_session_data(self, session_id: str) -> Dict:
        """세션 데이터 내보내기"""
        with self.session_lock:
            session = self.active_sessions.get(session_id)
            if not session:
                return {}
            
            # 세션 데이터를 직렬화 가능한 형태로 변환
            return {
                "session_info": asdict(session),
                "conversation_analysis": self.analyze_conversation_patterns(session_id),
                "context_summary": self.get_contextual_summary(session_id),
                "export_timestamp": datetime.now().isoformat()
            }
    
    def cleanup_expired_sessions(self):
        """만료된 세션 정리"""
        with self.session_lock:
            current_time = datetime.now()
            expired_sessions = []
            
            for session_id, session in self.active_sessions.items():
                last_activity = datetime.fromisoformat(session.last_activity)
                if (current_time - last_activity).total_seconds() > (self.session_timeout_hours * 3600):
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                print(f"만료된 세션 정리: {session_id}")
                del self.active_sessions[session_id]
                if session_id in self.context_windows:
                    del self.context_windows[session_id]
    
    def start_cleanup_service(self):
        """자동 세션 정리 서비스 시작"""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(3600)  # 1시간마다 실행
                    self.cleanup_expired_sessions()
                except Exception as e:
                    print(f"세션 정리 중 오류: {e}")
        
        self.cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        print("자동 세션 정리 서비스 시작")
    
    def get_system_stats(self) -> Dict:
        """시스템 통계 정보"""
        with self.session_lock:
            total_sessions = len(self.active_sessions)
            total_turns = sum(len(session.conversation_turns) for session in self.active_sessions.values())
            
            # 활성 세션 통계
            current_time = datetime.now()
            active_sessions = 0
            
            for session in self.active_sessions.values():
                last_activity = datetime.fromisoformat(session.last_activity)
                if (current_time - last_activity).total_seconds() < 3600:  # 1시간 이내
                    active_sessions += 1
            
            return {
                "total_sessions": total_sessions,
                "active_sessions_1h": active_sessions,
                "total_conversation_turns": total_turns,
                "avg_turns_per_session": round(total_turns / total_sessions, 1) if total_sessions > 0 else 0,
                "max_history_length": self.max_history_length,
                "session_timeout_hours": self.session_timeout_hours
            }

# 전역 컨텍스트 관리자 인스턴스
context_manager = ContextManager()

def get_or_create_session(session_id: Optional[str] = None, 
                         user_id: Optional[str] = None) -> str:
    """세션 조회 또는 생성"""
    if session_id and context_manager.get_session(session_id):
        context_manager.update_session_activity(session_id)
        return session_id
    else:
        return context_manager.create_session(user_id)

def add_conversation(session_id: str, user_input: str, assistant_response: str,
                    intent: str, confidence: float, apis_used: List[str],
                    success: bool, processing_time: float, metadata: Dict = None) -> str:
    """대화 기록 추가"""
    return context_manager.add_conversation_turn(
        session_id, user_input, assistant_response, intent,
        confidence, apis_used, success, processing_time, metadata
    )

def get_conversation_context(session_id: str) -> Dict:
    """대화 컨텍스트 조회"""
    return context_manager.get_conversation_context(session_id)

def get_contextual_summary(session_id: str) -> str:
    """컨텍스트 요약 조회"""
    return context_manager.get_contextual_summary(session_id)

def test_context_manager():
    """컨텍스트 관리자 테스트"""
    print("=" * 60)
    print("RAG 컨텍스트 관리자 테스트")
    print("=" * 60)
    
    # 테스트 세션 생성
    session_id = context_manager.create_session(user_id="test_user")
    print(f"테스트 세션 생성: {session_id}")
    
    # 테스트 대화 추가
    test_conversations = [
        {
            "user_input": "오늘 날씨 어때?",
            "assistant_response": "서울의 현재 날씨는 맑음, 21도입니다.",
            "intent": "weather_query",
            "confidence": 0.95,
            "apis_used": ["weather"],
            "success": True,
            "processing_time": 1.2
        },
        {
            "user_input": "내일 일정 확인해줘",
            "assistant_response": "내일 오전 10시에 팀 미팅이 있습니다.",
            "intent": "calendar_query",
            "confidence": 0.9,
            "apis_used": ["calendar"],
            "success": True,
            "processing_time": 0.8
        },
        {
            "user_input": "날씨 정보를 팀에게 알려줘",
            "assistant_response": "날씨 정보를 슬랙으로 전송했습니다.",
            "intent": "weather_and_notify",
            "confidence": 0.85,
            "apis_used": ["weather", "notification"],
            "success": True,
            "processing_time": 2.1
        }
    ]
    
    print(f"\n💬 테스트 대화 {len(test_conversations)}개 추가:")
    for i, conv in enumerate(test_conversations, 1):
        turn_id = context_manager.add_conversation_turn(session_id, **conv)
        print(f"  {i}. {conv['user_input'][:30]}... → {turn_id}")
    
    # 컨텍스트 조회
    print(f"\n📋 세션 컨텍스트:")
    context = context_manager.get_conversation_context(session_id, include_metadata=True)
    print(f"  세션 ID: {context['session_id']}")
    print(f"  총 대화 턴: {context['total_turns']}")
    print(f"  활성 토픽: {context['active_topics']}")
    
    # 컨텍스트 요약
    print(f"\n📝 컨텍스트 요약:")
    summary = context_manager.get_contextual_summary(session_id)
    print(f"  {summary}")
    
    # 대화 패턴 분석
    print(f"\n📊 대화 패턴 분석:")
    analysis = context_manager.analyze_conversation_patterns(session_id)
    stats = analysis["session_stats"]
    print(f"  성공률: {stats['success_rate']}%")
    print(f"  평균 처리시간: {stats['avg_processing_time']}초")
    print(f"  API 사용: {analysis['api_usage']}")
    
    # 시스템 통계
    print(f"\n⚙️ 시스템 통계:")
    system_stats = context_manager.get_system_stats()
    for key, value in system_stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n" + "=" * 60)
    print("컨텍스트 관리자 테스트 완료!")

if __name__ == "__main__":
    test_context_manager() 