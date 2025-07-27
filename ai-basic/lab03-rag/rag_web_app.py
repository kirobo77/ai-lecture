"""
Lab 3 - Step 4: RAG 웹 인터페이스
Streamlit을 활용한 사용자 친화적 RAG 시스템 구현
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from shared.config import validate_api_keys, CHROMA_PERSIST_DIRECTORY, OPENAI_API_KEY, CHAT_MODEL
from shared.utils import EmbeddingUtils, ChatUtils
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
import io
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import uuid

# 기존 모듈들 import
try:
    from basic_rag import BasicRAGSystem, RAGResponse, RetrievalResult
    from advanced_retrieval import HybridRetriever, SearchResult
    from context_management import ContextManager, ContextWindow
except ImportError as e:
    st.error(f"모듈 import 오류: {e}")
    st.stop()

@dataclass
class UserFeedback:
    """사용자 피드백을 담는 데이터 클래스"""
    question: str
    answer: str
    rating: int  # 1-5
    feedback_text: str
    timestamp: str
    session_id: str

class RAGWebInterface:
    """RAG 웹 인터페이스 메인 클래스"""
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_page_config()
        
    def initialize_session_state(self):
        """세션 상태 초기화"""
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = None
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'feedback_data' not in st.session_state:
            st.session_state.feedback_data = []
        
        if 'uploaded_docs' not in st.session_state:
            st.session_state.uploaded_docs = []
        
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        
        if 'system_stats' not in st.session_state:
            st.session_state.system_stats = {}
    
    def setup_page_config(self):
        """페이지 설정"""
        st.set_page_config(
            page_title="AI 기초 실습 - RAG 시스템",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )

class DocumentManager:
    """문서 관리를 위한 클래스"""
    
    @staticmethod
    def upload_documents():
        """문서 업로드 인터페이스"""
        st.header("📄 문서 관리")
        
        # 파일 업로드
        uploaded_files = st.file_uploader(
            "문서 파일을 업로드하세요",
            type=['txt', 'md', 'json'],
            accept_multiple_files=True,
            help="텍스트 파일(.txt), 마크다운 파일(.md), JSON 파일(.json)을 지원합니다."
        )
        
        if uploaded_files:
            documents = []
            
            for uploaded_file in uploaded_files:
                try:
                    # 파일 내용 읽기
                    if uploaded_file.type == "text/plain":
                        content = str(uploaded_file.read(), "utf-8")
                    elif uploaded_file.type == "application/json":
                        content = json.loads(uploaded_file.read())
                        if isinstance(content, dict) and 'content' in content:
                            content = content['content']
                        else:
                            content = json.dumps(content, ensure_ascii=False, indent=2)
                    else:
                        content = str(uploaded_file.read(), "utf-8")
                    
                    document = {
                        "id": f"upload_{int(time.time())}_{uploaded_file.name}",
                        "content": content,
                        "metadata": {
                            "filename": uploaded_file.name,
                            "file_type": uploaded_file.type,
                            "upload_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "file_size": uploaded_file.size
                        }
                    }
                    
                    documents.append(document)
                    
                except Exception as e:
                    st.error(f"파일 '{uploaded_file.name}' 처리 중 오류: {e}")
            
            if documents:
                st.success(f"{len(documents)}개 문서가 업로드되었습니다.")
                
                # 문서 미리보기
                if st.checkbox("업로드된 문서 미리보기"):
                    for doc in documents:
                        with st.expander(f"📄 {doc['metadata']['filename']}"):
                            st.write(f"**크기:** {doc['metadata']['file_size']} bytes")
                            st.write(f"**업로드 시간:** {doc['metadata']['upload_time']}")
                            st.write("**내용 미리보기:**")
                            st.text(doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'])
                
                return documents
        
        return []
    
    @staticmethod
    def load_sample_documents():
        """샘플 문서 로드"""
        if st.button("📚 샘플 문서 로드"):
            sample_docs = [
                {
                    "id": "sample_ai_intro",
                    "content": """인공지능(AI)은 인간의 지능을 모방하여 기계가 학습하고 추론할 수 있게 하는 기술입니다. 
                    AI는 머신러닝, 딥러닝, 자연어 처리, 컴퓨터 비전 등 다양한 분야를 포함합니다. 
                    현재 AI는 의료진단, 자율주행, 언어번역, 이미지 인식 등 많은 영역에서 활용되고 있습니다.""",
                    "metadata": {"category": "AI_기초", "author": "AI연구팀", "date": "2024-01-01"}
                },
                {
                    "id": "sample_rag_system",
                    "content": """RAG(Retrieval-Augmented Generation)는 검색과 생성을 결합한 AI 시스템입니다.
                    먼저 관련 문서를 검색하여 컨텍스트를 구성하고, 이를 바탕으로 답변을 생성합니다.
                    RAG는 할루시네이션을 줄이고 정확한 정보 기반 답변을 제공할 수 있습니다.""",
                    "metadata": {"category": "RAG", "author": "RAG연구팀", "date": "2024-01-04"}
                },
                {
                    "id": "sample_vector_db",
                    "content": """벡터 데이터베이스는 고차원 벡터 데이터를 효율적으로 저장하고 검색하는 데이터베이스입니다.
                    임베딩 벡터 간의 유사도를 계산하여 의미적으로 관련된 정보를 찾을 수 있습니다.
                    ChromaDB, Pinecone, Weaviate 등이 대표적인 벡터 데이터베이스입니다.""",
                    "metadata": {"category": "벡터DB", "author": "데이터팀", "date": "2024-01-05"}
                }
            ]
            
            st.success("샘플 문서가 로드되었습니다!")
            return sample_docs
        
        return []

class RAGInterface:
    """RAG 시스템 인터페이스"""
    
    @staticmethod
    def setup_rag_system():
        """RAG 시스템 설정"""
        st.header("⚙️ RAG 시스템 설정")
        
        col1, col2 = st.columns(2)
        
        with col1:
            collection_name = st.text_input(
                "컬렉션 이름",
                value="rag-web-demo",
                help="벡터 데이터베이스 컬렉션 이름"
            )
            
            max_results = st.slider(
                "최대 검색 결과 수",
                min_value=1,
                max_value=20,
                value=5,
                help="검색할 문서의 최대 개수"
            )
        
        with col2:
            relevance_threshold = st.slider(
                "관련성 임계값",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="답변에 포함할 문서의 최소 관련성 점수"
            )
            
            semantic_weight = st.slider(
                "시맨틱 검색 가중치",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="하이브리드 검색에서 시맨틱 검색의 가중치"
            )
        
        # 시스템 초기화
        if st.button("🚀 RAG 시스템 초기화"):
            if not validate_api_keys():
                st.error("API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
                return None
            
            try:
                with st.spinner("RAG 시스템 초기화 중..."):
                    rag_system = BasicRAGSystem(collection_name)
                    st.session_state.rag_system = rag_system
                    st.session_state.max_results = max_results
                    st.session_state.relevance_threshold = relevance_threshold
                    st.session_state.semantic_weight = semantic_weight
                
                st.success("RAG 시스템이 성공적으로 초기화되었습니다!")
                
                # 시스템 통계 표시
                stats = rag_system.get_system_stats()
                st.session_state.system_stats = stats
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("총 문서 수", stats.get('total_documents', 0))
                with col2:
                    st.metric("임베딩 모델", stats.get('embedding_model', 'N/A'))
                with col3:
                    st.metric("채팅 모델", stats.get('chat_model', 'N/A'))
                
                return rag_system
                
            except Exception as e:
                st.error(f"RAG 시스템 초기화 실패: {e}")
                return None
        
        return st.session_state.rag_system
    
    @staticmethod
    def chat_interface():
        """채팅 인터페이스"""
        st.header("💬 RAG 채팅")
        
        if not st.session_state.rag_system:
            st.warning("먼저 RAG 시스템을 초기화해주세요.")
            return
        
        # 채팅 히스토리 표시
        chat_container = st.container()
        
        with chat_container:
            for i, chat in enumerate(st.session_state.chat_history):
                # 사용자 질문
                with st.chat_message("user"):
                    st.write(chat['question'])
                
                # AI 답변
                with st.chat_message("assistant"):
                    st.write(chat['answer'])
                    
                    # 참조 문서 표시
                    if chat.get('sources'):
                        with st.expander("📚 참조 문서"):
                            for j, source in enumerate(chat['sources'][:3]):
                                st.write(f"**문서 {j+1}** (관련성: {source.relevance_score:.3f})")
                                st.write(f"카테고리: {source.metadata.get('category', 'N/A')}")
                                st.write(f"내용: {source.document[:200]}...")
                                st.divider()
                    
                    # 처리 정보
                    if chat.get('processing_time'):
                        st.caption(f"처리 시간: {chat['processing_time']:.2f}초")
        
        # 새 질문 입력
        user_question = st.chat_input("질문을 입력하세요...")
        
        if user_question:
            RAGInterface.process_question(user_question)
    
    @staticmethod
    def process_question(question: str):
        """질문 처리 및 답변 생성"""
        rag_system = st.session_state.rag_system
        
        try:
            # 답변 생성 (진행률 표시와 함께)
            with st.spinner("답변 생성 중..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 단계별 진행률 표시
                status_text.text("1/4: 쿼리 처리 중...")
                progress_bar.progress(25)
                time.sleep(0.5)
                
                status_text.text("2/4: 관련 문서 검색 중...")
                progress_bar.progress(50)
                time.sleep(0.5)
                
                status_text.text("3/4: 컨텍스트 구성 중...")
                progress_bar.progress(75)
                time.sleep(0.5)
                
                status_text.text("4/4: 답변 생성 중...")
                progress_bar.progress(100)
                
                # 실제 RAG 처리
                response = rag_system.query(
                    question,
                    top_k=st.session_state.get('max_results', 5),
                    relevance_threshold=st.session_state.get('relevance_threshold', 0.3)
                )
                
                # 진행률 표시 제거
                progress_bar.empty()
                status_text.empty()
            
            # 채팅 히스토리에 추가
            chat_entry = {
                'question': question,
                'answer': response.answer,
                'sources': response.sources,
                'processing_time': response.processing_time,
                'token_usage': response.token_usage,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            st.session_state.chat_history.append(chat_entry)
            
            # 페이지 새로고침 (채팅 히스토리 업데이트)
            st.rerun()
            
        except Exception as e:
            st.error(f"답변 생성 중 오류: {e}")

class AnalyticsInterface:
    """분석 인터페이스"""
    
    @staticmethod
    def show_analytics():
        """분석 대시보드"""
        st.header("📊 시스템 분석")
        
        if not st.session_state.chat_history:
            st.info("채팅 기록이 없습니다. 먼저 질문을 해보세요!")
            return
        
        # 기본 통계
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("총 질문 수", len(st.session_state.chat_history))
        
        with col2:
            avg_time = sum(chat.get('processing_time', 0) for chat in st.session_state.chat_history) / len(st.session_state.chat_history)
            st.metric("평균 응답 시간", f"{avg_time:.2f}초")
        
        with col3:
            total_tokens = sum(chat.get('token_usage', {}).get('total_tokens', 0) for chat in st.session_state.chat_history)
            st.metric("총 토큰 사용량", total_tokens)
        
        with col4:
            if st.session_state.feedback_data:
                avg_rating = sum(fb.rating for fb in st.session_state.feedback_data) / len(st.session_state.feedback_data)
                st.metric("평균 만족도", f"{avg_rating:.1f}/5")
            else:
                st.metric("평균 만족도", "N/A")
        
        # 시간별 응답 시간 차트
        if len(st.session_state.chat_history) > 1:
            st.subheader("응답 시간 추이")
            
            df = pd.DataFrame([
                {
                    'question_num': i+1,
                    'processing_time': chat.get('processing_time', 0),
                    'timestamp': chat.get('timestamp', ''),
                    'question': chat['question'][:50] + "..." if len(chat['question']) > 50 else chat['question']
                }
                for i, chat in enumerate(st.session_state.chat_history)
            ])
            
            fig = px.line(df, x='question_num', y='processing_time', 
                         title='질문별 응답 시간',
                         labels={'question_num': '질문 순서', 'processing_time': '응답 시간 (초)'},
                         hover_data=['question'])
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 토큰 사용량 분석
        st.subheader("토큰 사용량 분석")
        
        token_data = []
        for i, chat in enumerate(st.session_state.chat_history):
            token_usage = chat.get('token_usage', {})
            if token_usage and not token_usage.get('error'):
                token_data.append({
                    'question_num': i+1,
                    'prompt_tokens': token_usage.get('prompt_tokens', 0),
                    'completion_tokens': token_usage.get('completion_tokens', 0),
                    'total_tokens': token_usage.get('total_tokens', 0)
                })
        
        if token_data:
            df_tokens = pd.DataFrame(token_data)
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Bar(x=df_tokens['question_num'], y=df_tokens['prompt_tokens'], 
                      name='프롬프트 토큰', marker_color='lightblue'),
                secondary_y=False,
            )
            
            fig.add_trace(
                go.Bar(x=df_tokens['question_num'], y=df_tokens['completion_tokens'], 
                      name='완성 토큰', marker_color='lightgreen'),
                secondary_y=False,
            )
            
            fig.add_trace(
                go.Scatter(x=df_tokens['question_num'], y=df_tokens['total_tokens'], 
                          mode='lines+markers', name='총 토큰', marker_color='red'),
                secondary_y=True,
            )
            
            fig.update_layout(title='질문별 토큰 사용량')
            fig.update_xaxes(title_text="질문 순서")
            fig.update_yaxes(title_text="토큰 수", secondary_y=False)
            fig.update_yaxes(title_text="총 토큰 수", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def show_feedback_analysis():
        """피드백 분석"""
        st.subheader("사용자 피드백 분석")
        
        if not st.session_state.feedback_data:
            st.info("피드백 데이터가 없습니다.")
            return
        
        # 만족도 분포
        ratings = [fb.rating for fb in st.session_state.feedback_data]
        rating_counts = pd.Series(ratings).value_counts().sort_index()
        
        fig = px.bar(x=rating_counts.index, y=rating_counts.values,
                    title='만족도 점수 분포',
                    labels={'x': '만족도 점수', 'y': '응답 수'})
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 피드백 텍스트 분석
        st.subheader("피드백 내용")
        
        for fb in st.session_state.feedback_data[-5:]:  # 최근 5개만 표시
            with st.expander(f"만족도 {fb.rating}/5 - {fb.timestamp}"):
                st.write(f"**질문:** {fb.question}")
                st.write(f"**피드백:** {fb.feedback_text}")

class FeedbackInterface:
    """피드백 인터페이스"""
    
    @staticmethod
    def collect_feedback():
        """피드백 수집"""
        st.header("📝 피드백")
        
        if not st.session_state.chat_history:
            st.info("피드백을 남기려면 먼저 질문을 해보세요.")
            return
        
        # 최근 질문 선택
        recent_questions = [
            f"Q{i+1}: {chat['question'][:100]}{'...' if len(chat['question']) > 100 else ''}"
            for i, chat in enumerate(st.session_state.chat_history[-10:])  # 최근 10개
        ]
        
        selected_question = st.selectbox(
            "피드백을 남길 질문을 선택하세요:",
            recent_questions
        )
        
        if selected_question:
            question_idx = int(selected_question.split(':')[0][1:]) - 1
            
            # 해당 질문과 답변 표시
            chat = st.session_state.chat_history[question_idx]
            
            with st.expander("선택된 질문과 답변"):
                st.write(f"**질문:** {chat['question']}")
                st.write(f"**답변:** {chat['answer']}")
            
            # 피드백 폼
            with st.form("feedback_form"):
                st.subheader("답변에 대한 평가를 해주세요")
                
                rating = st.radio(
                    "만족도 (1: 매우 불만족, 5: 매우 만족)",
                    options=[1, 2, 3, 4, 5],
                    index=4,
                    horizontal=True
                )
                
                feedback_text = st.text_area(
                    "구체적인 피드백을 남겨주세요:",
                    placeholder="답변의 정확성, 유용성, 개선사항 등을 자유롭게 작성해주세요."
                )
                
                submitted = st.form_submit_button("피드백 제출")
                
                if submitted:
                    feedback = UserFeedback(
                        question=chat['question'],
                        answer=chat['answer'],
                        rating=rating,
                        feedback_text=feedback_text,
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        session_id=st.session_state.session_id
                    )
                    
                    st.session_state.feedback_data.append(feedback)
                    st.success("피드백이 성공적으로 제출되었습니다!")
                    
                    # 피드백 저장 (실제 환경에서는 데이터베이스에 저장)
                    FeedbackInterface.save_feedback(feedback)
    
    @staticmethod
    def save_feedback(feedback: UserFeedback):
        """피드백 저장 (파일 또는 데이터베이스)"""
        try:
            # JSON 파일로 저장
            feedback_file = "feedback_data.json"
            
            if os.path.exists(feedback_file):
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []
            
            existing_data.append(asdict(feedback))
            
            with open(feedback_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            st.error(f"피드백 저장 실패: {e}")

def main():
    """메인 애플리케이션"""
    
    # 웹 인터페이스 초기화
    web_interface = RAGWebInterface()
    
    # 제목 및 설명
    st.title("🤖 AI 기초 실습 - RAG 시스템")
    st.markdown("""
    **Retrieval-Augmented Generation (RAG) 시스템 웹 인터페이스**
    
    이 웹 애플리케이션을 통해 RAG 시스템의 다양한 기능을 체험해보세요:
    - 📄 문서 업로드 및 관리
    - 💬 실시간 질의응답
    - 📊 시스템 성능 분석
    - 📝 사용자 피드백 수집
    """)
    
    # 사이드바 메뉴
    with st.sidebar:
        st.header("🗂️ 메뉴")
        
        menu_option = st.radio(
            "선택하세요:",
            [
                "🏠 홈",
                "📄 문서 관리", 
                "⚙️ RAG 설정",
                "💬 채팅",
                "📊 분석",
                "📝 피드백"
            ]
        )
        
        # API 키 상태 표시
        st.divider()
        st.subheader("🔐 시스템 상태")
        
        if validate_api_keys():
            st.success("✅ API 키 설정됨")
        else:
            st.error("❌ API 키 미설정")
            st.warning("`.env` 파일에 `OPENAI_API_KEY`를 설정해주세요.")
        
        # 세션 정보
        if st.session_state.rag_system:
            st.success("✅ RAG 시스템 활성화")
            stats = st.session_state.system_stats
            st.metric("문서 수", stats.get('total_documents', 0))
        else:
            st.warning("⚠️ RAG 시스템 비활성화")
    
    # 메인 컨텐츠
    if menu_option == "🏠 홈":
        st.header("🏠 RAG 시스템 홈")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚀 시작하기")
            st.markdown("""
            1. **문서 관리**에서 문서를 업로드하거나 샘플 문서를 로드하세요
            2. **RAG 설정**에서 시스템을 초기화하세요
            3. **채팅**에서 질문을 해보세요
            4. **분석**에서 시스템 성능을 확인하세요
            5. **피드백**으로 시스템 개선에 도움을 주세요
            """)
        
        with col2:
            st.subheader("📈 시스템 현황")
            
            if st.session_state.chat_history:
                st.metric("총 질문 수", len(st.session_state.chat_history))
                
                latest_chat = st.session_state.chat_history[-1]
                st.write("**최근 질문:**")
                st.write(f"Q: {latest_chat['question'][:100]}...")
                st.write(f"처리 시간: {latest_chat.get('processing_time', 0):.2f}초")
            else:
                st.info("아직 질문이 없습니다.")
    
    elif menu_option == "📄 문서 관리":
        # 문서 업로드
        uploaded_docs = DocumentManager.upload_documents()
        
        if uploaded_docs:
            st.session_state.uploaded_docs.extend(uploaded_docs)
        
        # 샘플 문서 로드
        sample_docs = DocumentManager.load_sample_documents()
        
        if sample_docs:
            st.session_state.uploaded_docs.extend(sample_docs)
        
        # 문서 목록 표시
        if st.session_state.uploaded_docs:
            st.subheader("📚 업로드된 문서 목록")
            
            docs_df = pd.DataFrame([
                {
                    "ID": doc["id"],
                    "파일명": doc["metadata"].get("filename", "N/A"),
                    "카테고리": doc["metadata"].get("category", "N/A"),
                    "크기": f"{len(doc['content'])}자",
                    "업로드 시간": doc["metadata"].get("upload_time", doc["metadata"].get("date", "N/A"))
                }
                for doc in st.session_state.uploaded_docs
            ])
            
            st.dataframe(docs_df, use_container_width=True)
    
    elif menu_option == "⚙️ RAG 설정":
        rag_system = RAGInterface.setup_rag_system()
        
        # 문서 추가
        if rag_system and st.session_state.uploaded_docs:
            if st.button("📥 문서를 RAG 시스템에 추가"):
                try:
                    with st.spinner("문서 추가 중..."):
                        rag_system.add_documents(st.session_state.uploaded_docs)
                    
                    st.success(f"{len(st.session_state.uploaded_docs)}개 문서가 RAG 시스템에 추가되었습니다!")
                    
                    # 시스템 통계 업데이트
                    stats = rag_system.get_system_stats()
                    st.session_state.system_stats = stats
                    
                except Exception as e:
                    st.error(f"문서 추가 실패: {e}")
    
    elif menu_option == "💬 채팅":
        RAGInterface.chat_interface()
    
    elif menu_option == "📊 분석":
        AnalyticsInterface.show_analytics()
        AnalyticsInterface.show_feedback_analysis()
    
    elif menu_option == "📝 피드백":
        FeedbackInterface.collect_feedback()
    
    # 하단 정보
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <small>AI 기초 실습 - Lab 3: RAG 시스템 | Powered by Streamlit & OpenAI</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 