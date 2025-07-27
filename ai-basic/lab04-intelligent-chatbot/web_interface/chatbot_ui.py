"""
Lab 4 - Streamlit 웹 인터페이스
지능형 API 라우팅 챗봇의 웹 UI
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
import json
import time
from datetime import datetime
from typing import Dict, List

# 메인 챗봇 시스템 import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from main_chatbot import IntelligentChatbot

def initialize_chatbot():
    """챗봇 초기화 (중복 방지)"""
    # 이미 초기화된 경우 재사용
    if 'chatbot' in st.session_state and st.session_state.get('initialized', False):
        return True
    
    # 초기화 중인 경우 대기
    if st.session_state.get('initializing', False):
        st.info("🔄 챗봇 초기화 중입니다... 잠시 기다려주세요.")
        return False
    
    try:
        st.session_state.initializing = True
        print("🚀 Streamlit: 챗봇 초기화 시작")
        
        st.session_state.chatbot = IntelligentChatbot()
        st.session_state.initialized = True
        st.session_state.initializing = False
        
        print("✅ Streamlit: 챗봇 초기화 완료")
        return True
        
    except Exception as e:
        st.error(f"챗봇 초기화 실패: {str(e)}")
        st.session_state.initialized = False
        st.session_state.initializing = False
        print(f"❌ Streamlit: 챗봇 초기화 실패 - {e}")
        return False

def initialize_session_state():
    """세션 상태 초기화"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'system_stats' not in st.session_state:
        st.session_state.system_stats = {
            "total_conversations": 0,
            "successful_conversations": 0,
            "agent_usage": {
                "weather": 0,
                "calendar": 0,
                "file": 0,
                "notification": 0
            }
        }

def display_header():
    """헤더 섹션 표시"""
    st.set_page_config(
        page_title="지능형 API 라우팅 챗봇",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 지능형 API 라우팅 챗봇")
    st.markdown("**RAG + Multi-Agent + MCP 통합 시스템**")
    
    # 시스템 상태 표시
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🧠 Intent Classifier", "준비됨", delta="RAG 기반")
    
    with col2:
        st.metric("🤖 Multi-Agent", "4개 활성", delta="Weather, Calendar, File, Notification")
    
    with col3:
        total_conversations = st.session_state.system_stats["total_conversations"]
        st.metric("💬 총 대화", total_conversations)
    
    with col4:
        if total_conversations > 0:
            success_rate = (st.session_state.system_stats["successful_conversations"] / total_conversations) * 100
            st.metric("✅ 성공률", f"{success_rate:.1f}%")
        else:
            st.metric("✅ 성공률", "0%")

def display_sidebar():
    """사이드바 표시"""
    with st.sidebar:
        st.header("🔧 시스템 설정")
        
        # API 상태 체크
        st.subheader("📡 API 서버 상태")
        api_status = check_api_servers()
        
        for api_name, status in api_status.items():
            if status:
                st.success(f"✅ {api_name} API")
            else:
                st.error(f"❌ {api_name} API")
        
        st.divider()
        
        # 에이전트 사용 통계
        st.subheader("📊 에이전트 사용 통계")
        agent_usage = st.session_state.system_stats["agent_usage"]
        
        for agent, count in agent_usage.items():
            st.metric(f"{agent.title()} Agent", count)
        
        st.divider()
        
        # 설정 옵션
        st.subheader("⚙️ 설정")
        
        show_debug = st.checkbox("디버그 모드", help="상세한 로그 표시")
        auto_clear = st.checkbox("자동 대화 정리", help="10개 이상시 자동 정리")
        
        if st.button("🗑️ 대화 기록 삭제"):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.success("대화 기록이 삭제되었습니다.")
            st.rerun()
        
        # 대화 내보내기
        if st.session_state.conversation_history:
            if st.button("📥 대화 기록 내보내기"):
                export_conversation_history()

def check_api_servers():
    """API 서버 상태 확인"""
    import requests
    
    api_endpoints = {
        "Weather": "http://localhost:8001",
        "Calendar": "http://localhost:8002", 
        "File Manager": "http://localhost:8003",
        "Notification": "http://localhost:8004"
    }
    
    status = {}
    for name, url in api_endpoints.items():
        try:
            response = requests.get(url, timeout=2)
            status[name] = response.status_code == 200
        except:
            status[name] = False
    
    return status

def display_chat_interface():
    """채팅 인터페이스 표시"""
    # 디버그 모드 체크박스 (한 번만 생성)
    debug_mode = st.sidebar.checkbox("디버그 모드", key="debug_mode_checkbox")
    
    # 채팅 메시지 표시
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # 메타데이터 표시 (디버그 모드일 때)
            if "metadata" in message and debug_mode:
                with st.expander(f"🔍 상세 정보 (메시지 {i+1})", key=f"metadata_expander_{i}"):
                    st.json(message["metadata"])

def process_user_input(user_input: str):
    """사용자 입력 처리"""
    if not user_input.strip():
        return
    
    # 사용자 메시지 추가
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().isoformat()
    })
    
    # 챗봇 응답 생성
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("🤔 처리 중..."):
            try:
                # 챗봇에서 응답 받기
                response_data = st.session_state.chatbot.process_message(user_input)
                
                response_text = response_data.get("response", "응답을 생성할 수 없습니다.")
                
                # 응답 표시
                st.markdown(response_text)
                
                # 처리 시간 표시
                processing_time = response_data.get("processing_time", 0)
                st.caption(f"⏱️ 처리 시간: {processing_time:.2f}초")
                
                # 어시스턴트 메시지 추가
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {
                        "processing_time": processing_time,
                        "success": response_data.get("success", False),
                        "agents_used": response_data.get("agents_used", []),
                        "intent": response_data.get("intent", "unknown")
                    }
                })
                
                # 대화 기록 저장
                st.session_state.conversation_history.append(response_data)
                
                # 통계 업데이트
                st.session_state.system_stats["total_conversations"] += 1
                if response_data.get("success", False):
                    st.session_state.system_stats["successful_conversations"] += 1
                
                # 에이전트 사용 통계 업데이트
                for agent in response_data.get("agents_used", []):
                    if agent in st.session_state.system_stats["agent_usage"]:
                        st.session_state.system_stats["agent_usage"][agent] += 1
                
            except Exception as e:
                error_message = f"❌ 오류가 발생했습니다: {str(e)}"
                st.error(error_message)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_message,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {"error": True}
                })

def display_example_queries():
    """예시 쿼리 표시"""
    st.subheader("💡 사용 예시")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🌤️ 날씨 관련**")
        if st.button("오늘 서울 날씨 어때?", key="weather1"):
            process_user_input("오늘 서울 날씨 어때?")
            st.rerun()
            
        if st.button("날씨 확인하고 팀에게 알려줘", key="weather2"):
            process_user_input("날씨 확인하고 팀에게 알려줘")
            st.rerun()
        
        st.markdown("**📅 일정 관리**")
        if st.button("오늘 일정 확인해줘", key="calendar1"):
            process_user_input("오늘 일정 확인해줘")
            st.rerun()
            
        if st.button("내일 3시에 회의 잡아줘", key="calendar2"):
            process_user_input("내일 3시에 회의 잡아줘")
            st.rerun()
    
    with col2:
        st.markdown("**📁 파일 관리**")
        if st.button("프로젝트 문서 찾아줘", key="file1"):
            process_user_input("프로젝트 문서 찾아줘")
            st.rerun()
            
        if st.button("API 명세서 보여줘", key="file2"):
            process_user_input("API 명세서 보여줘")
            st.rerun()
        
        st.markdown("**🔔 알림 발송**")
        if st.button("팀에게 알려줘", key="notification1"):
            process_user_input("팀에게 알려줘")
            st.rerun()
            
        if st.button("회의 일정을 이메일로 보내줘", key="notification2"):
            process_user_input("회의 일정을 이메일로 보내줘")
            st.rerun()

def export_conversation_history():
    """대화 기록 내보내기"""
    if st.session_state.conversation_history:
        export_data = {
            "export_time": datetime.now().isoformat(),
            "total_conversations": len(st.session_state.conversation_history),
            "conversations": st.session_state.conversation_history,
            "system_stats": st.session_state.system_stats
        }
        
        json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
        
        st.download_button(
            label="📥 JSON 파일로 다운로드",
            data=json_str,
            file_name=f"chatbot_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def main():
    """메인 함수"""
    # 세션 상태 초기화
    initialize_session_state()
    
    # 헤더 표시
    display_header()
    
    # 챗봇 초기화
    if not initialize_chatbot():
        st.error("❌ 챗봇을 초기화할 수 없습니다. API 키 설정을 확인해주세요.")
        return
    
    # 사이드바 표시
    display_sidebar()
    
    # 메인 컨텐츠
    main_tab, example_tab, stats_tab = st.tabs(["💬 채팅", "💡 예시", "📊 통계"])
    
    with main_tab:
        # 채팅 인터페이스
        display_chat_interface()
        
        # 채팅 입력
        if prompt := st.chat_input("메시지를 입력하세요..."):
            process_user_input(prompt)
            st.rerun()
    
    with example_tab:
        display_example_queries()
    
    with stats_tab:
        st.subheader("📊 시스템 통계")
        
        # 대화 통계
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("총 대화 수", st.session_state.system_stats["total_conversations"])
        
        with col2:
            st.metric("성공한 대화", st.session_state.system_stats["successful_conversations"])
        
        with col3:
            if st.session_state.system_stats["total_conversations"] > 0:
                success_rate = (st.session_state.system_stats["successful_conversations"] / 
                              st.session_state.system_stats["total_conversations"]) * 100
                st.metric("성공률", f"{success_rate:.1f}%")
            else:
                st.metric("성공률", "0%")
        
        # 에이전트 사용 차트
        if any(count > 0 for count in st.session_state.system_stats["agent_usage"].values()):
            st.subheader("🤖 에이전트 사용 분포")
            agent_data = st.session_state.system_stats["agent_usage"]
            st.bar_chart(agent_data)
        
        # 최근 대화 기록
        if st.session_state.conversation_history:
            st.subheader("📝 최근 대화 기록")
            
            # 최근 5개 대화만 표시
            recent_conversations = st.session_state.conversation_history[-5:]
            
            for i, conv in enumerate(reversed(recent_conversations), 1):
                with st.expander(f"대화 {len(st.session_state.conversation_history) - i + 1}: {conv.get('user_input', 'N/A')[:50]}..."):
                    st.json(conv)

if __name__ == "__main__":
    main() 