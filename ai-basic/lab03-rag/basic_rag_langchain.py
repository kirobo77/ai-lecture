"""
Lab 3 - Step 1: 기본 RAG 파이프라인 (LangChain 버전)
LangChain을 활용한 간결하고 효율적인 RAG 시스템 구현
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import validate_api_keys, CHROMA_PERSIST_DIRECTORY, OPENAI_API_KEY, CHAT_MODEL
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import httpx

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.schema import Document

# SSL 검증 비활성화 HTTP 클라이언트
no_ssl_httpx = httpx.Client(verify=False)


@dataclass
class LangChainRAGResponse:
    """LangChain RAG 시스템의 응답을 담는 데이터 클래스"""
    question: str
    answer: str
    source_documents: List[Document]
    processing_time: float
    retrieval_time: float
    generation_time: float

class LangChainRAGSystem:
    """LangChain 기반 RAG 시스템"""
    
    def __init__(self, collection_name: str = "langchain-rag-demo", 
                 chunk_size: int = 500, chunk_overlap: int = 50):
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # LangChain 컴포넌트 초기화
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model="text-embedding-ada-002",
            http_client=no_ssl_httpx
        )
        
        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model=CHAT_MODEL,
            temperature=0.1,
            http_client=no_ssl_httpx
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Vector store 초기화
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        
        self._setup_vectorstore()
        self._setup_rag_chain()
        
        print(f"LangChain RAG System 초기화 완료")
        print(f"컬렉션: {collection_name}")
        print(f"청크 크기: {chunk_size}, 오버랩: {chunk_overlap}")
    
    def _setup_vectorstore(self):
        """벡터 스토어 설정"""
        try:
            # ChromaDB 벡터 스토어 초기화 (기존 데이터 로드 시도)
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=CHROMA_PERSIST_DIRECTORY
            )
            
            # Retriever 설정
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            print(f"벡터 스토어 설정 완료")
            
        except Exception as e:
            print(f"벡터 스토어 설정 실패: {e}")
            raise
    
    def _setup_rag_chain(self):
        """RAG 체인 설정 (LCEL 사용)"""
        
        # 프롬프트 템플릿 정의
        rag_prompt = ChatPromptTemplate.from_template("""
다음 맥락 정보를 바탕으로 질문에 답해주세요.

맥락 정보:
{context}

질문: {question}

답변할 때 다음 사항을 지켜주세요:
1. 제공된 맥락 정보를 바탕으로 정확하고 구체적으로 답변하세요
2. 맥락에 없는 정보는 추측하지 마세요
3. 출처가 명확한 정보를 우선적으로 활용하세요
4. 답변을 명확하고 이해하기 쉽게 구성하세요

답변:""")
        
        # 문서를 문자열로 포맷하는 함수
        def format_docs(docs):
            formatted_docs = []
            for i, doc in enumerate(docs):
                content = doc.page_content
                metadata = doc.metadata
                
                doc_text = f"[문서 {i+1}]\n{content}\n"
                if metadata:
                    meta_info = []
                    for key, value in metadata.items():
                        if key in ['category', 'author', 'date', 'source']:
                            meta_info.append(f"{key}: {value}")
                    if meta_info:
                        doc_text += f"출처: {', '.join(meta_info)}\n"
                
                formatted_docs.append(doc_text)
            
            return "\n".join(formatted_docs)
        
        # RAG 체인 구성 (LCEL)
        self.rag_chain = (
            RunnableParallel({
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            })
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("RAG 체인 설정 완료")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """문서를 벡터 스토어에 추가"""
        if not documents:
            print("추가할 문서가 없습니다.")
            return
        
        try:
            # 문서를 LangChain Document 객체로 변환
            langchain_docs = []
            
            for doc in documents:
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
                
                # 문서 ID 메타데이터에 추가
                if 'id' in doc:
                    metadata['source'] = doc['id']
                
                # Document 객체 생성
                document = Document(
                    page_content=content,
                    metadata=metadata
                )
                
                langchain_docs.append(document)
            
            # 텍스트 분할
            print(f"문서 분할 중... (청크 크기: {self.chunk_size})")
            split_docs = self.text_splitter.split_documents(langchain_docs)
            
            print(f"분할된 청크 수: {len(split_docs)}")
            
            # 벡터 스토어에 추가
            print("벡터 스토어에 문서 추가 중...")
            self.vectorstore.add_documents(split_docs)
            
            print(f"{len(documents)}개 문서 ({len(split_docs)}개 청크)가 성공적으로 추가되었습니다.")
            
        except Exception as e:
            print(f"문서 추가 중 오류: {e}")
            raise
    
    def query(self, question: str, return_source_documents: bool = True) -> LangChainRAGResponse:
        """RAG 시스템을 통한 질의응답"""
        
        start_time = time.time()
        
        print(f"\n질문: {question}")
        print("=" * 50)
        
        try:
            # 1. 문서 검색 (시간 측정)
            print("관련 문서 검색 중...")
            retrieval_start = time.time()
            
            retrieved_docs = self.retriever.get_relevant_documents(question)
            
            retrieval_time = time.time() - retrieval_start
            print(f"검색된 문서 수: {len(retrieved_docs)}")
            print(f"검색 시간: {retrieval_time:.3f}초")
            
            # 검색 결과 미리보기
            if retrieved_docs:
                print("검색 결과 미리보기:")
                for i, doc in enumerate(retrieved_docs[:3]):
                    print(f"  {i+1}. {doc.page_content[:100]}...")
                    if doc.metadata:
                        print(f"     메타데이터: {doc.metadata}")
            
            # 2. 답변 생성 (시간 측정)
            print(f"\n답변 생성 중...")
            generation_start = time.time()
            
            answer = self.rag_chain.invoke(question)
            
            generation_time = time.time() - generation_start
            print(f"생성 시간: {generation_time:.3f}초")
            
            # 3. 총 처리 시간
            total_time = time.time() - start_time
            
            # 4. 응답 구성
            response = LangChainRAGResponse(
                question=question,
                answer=answer,
                source_documents=retrieved_docs if return_source_documents else [],
                processing_time=total_time,
                retrieval_time=retrieval_time,
                generation_time=generation_time
            )
            
            return response
            
        except Exception as e:
            print(f"질의응답 중 오류: {e}")
            # 오류 시 기본 응답 반환
            return LangChainRAGResponse(
                question=question,
                answer=f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {e}",
                source_documents=[],
                processing_time=time.time() - start_time,
                retrieval_time=0,
                generation_time=0
            )
    
    def update_retriever_config(self, search_type: str = "similarity", k: int = 5, 
                               fetch_k: int = 20, score_threshold: float = 0.5):
        """Retriever 설정 업데이트"""
        search_kwargs = {"k": k}
        
        if search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = score_threshold
        elif search_type == "mmr":
            search_kwargs["fetch_k"] = fetch_k
        
        self.retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        print(f"Retriever 설정 업데이트: {search_type}, k={k}")
    
    def get_vectorstore_stats(self) -> Dict[str, Any]:
        """벡터 스토어 통계 정보"""
        try:
            # ChromaDB 컬렉션 정보
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "embedding_model": "text-embedding-ada-002",
                "chat_model": self.llm.model_name,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            }
        except Exception as e:
            return {"error": f"통계 정보를 가져올 수 없습니다: {e}"}

def create_sample_documents() -> List[Dict[str, Any]]:
    """테스트용 샘플 문서 생성 (동일한 데이터)"""
    documents = [
        {
            "id": "ai_intro_lc_001",
            "content": """인공지능(AI)은 인간의 지능을 모방하여 기계가 학습하고 추론할 수 있게 하는 기술입니다. 
            AI는 머신러닝, 딥러닝, 자연어 처리, 컴퓨터 비전 등 다양한 분야를 포함합니다. 
            현재 AI는 의료진단, 자율주행, 언어번역, 이미지 인식 등 많은 영역에서 활용되고 있습니다.""",
            "metadata": {"category": "AI_기초", "author": "AI연구팀", "date": "2024-01-01"}
        },
        {
            "id": "ml_basics_lc_001", 
            "content": """머신러닝은 데이터를 통해 컴퓨터가 자동으로 학습하는 인공지능의 한 분야입니다.
            지도학습, 비지도학습, 강화학습의 세 가지 주요 접근법이 있습니다.
            지도학습은 라벨이 있는 데이터로 훈련하며, 분류와 회귀 문제를 해결합니다.
            비지도학습은 라벨 없는 데이터에서 패턴을 찾는 방법입니다.""",
            "metadata": {"category": "머신러닝", "author": "ML연구팀", "date": "2024-01-02"}
        },
        {
            "id": "dl_concepts_lc_001",
            "content": """딥러닝은 신경망을 여러 층으로 쌓아 복잡한 패턴을 학습하는 머신러닝 기법입니다.
            CNN(Convolutional Neural Network)은 이미지 처리에 특화되어 있고,
            RNN(Recurrent Neural Network)은 순차 데이터 처리에 적합합니다.
            트랜스포머 모델은 자연어 처리 분야에서 혁신적인 성과를 거두었습니다.""",
            "metadata": {"category": "딥러닝", "author": "DL연구팀", "date": "2024-01-03"}
        },
        {
            "id": "rag_system_lc_001",
            "content": """RAG(Retrieval-Augmented Generation)는 검색과 생성을 결합한 AI 시스템입니다.
            먼저 관련 문서를 검색하여 컨텍스트를 구성하고, 이를 바탕으로 답변을 생성합니다.
            RAG는 할루시네이션을 줄이고 정확한 정보 기반 답변을 제공할 수 있습니다.
            벡터 데이터베이스와 대화형 AI 모델의 조합으로 구현됩니다.""",
            "metadata": {"category": "RAG", "author": "RAG연구팀", "date": "2024-01-04"}
        },
        {
            "id": "langchain_intro_001",
            "content": """LangChain은 언어 모델을 활용한 애플리케이션 개발을 위한 프레임워크입니다.
            RAG, 에이전트, 체인 구성 등 복잡한 AI 워크플로우를 간단하게 구현할 수 있습니다.
            LCEL(LangChain Expression Language)을 통해 모듈식 체인 구성이 가능합니다.
            다양한 벡터 스토어, LLM, 도구와의 통합을 지원합니다.""",
            "metadata": {"category": "LangChain", "author": "프레임워크팀", "date": "2024-01-06"}
        }
    ]
    
    return documents

def demonstrate_langchain_rag():
    """LangChain RAG 시스템 시연"""
    print("Lab 3 - Step 1: 기본 RAG 파이프라인 (LangChain 버전)")
    print("LangChain을 활용한 간결한 RAG 시스템 구현\n")
    
    # API 키 검증
    if not validate_api_keys():
        return
    
    # LangChain RAG 시스템 초기화
    print("LangChain RAG 시스템 초기화")
    print("=" * 40)
    rag_system = LangChainRAGSystem("langchain-rag-demo")
    
    # 샘플 문서 추가
    print(f"\n샘플 문서 추가")
    print("=" * 40)
    sample_docs = create_sample_documents()
    rag_system.add_documents(sample_docs)
    
    # 시스템 통계
    stats = rag_system.get_vectorstore_stats()
    print(f"\n시스템 통계:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 테스트 질문들
    test_questions = [
        "RAG 시스템이 무엇인가요?",
        "머신러닝의 주요 접근법은 무엇인가요?", 
        "딥러닝에서 사용되는 신경망 종류를 알려주세요",
        "LangChain의 장점은 무엇인가요?",
        "AI의 실제 응용 분야는 어떤 것들이 있나요?"
    ]
    
    # 각 질문에 대해 RAG 시스템 테스트
    for i, question in enumerate(test_questions, 1):
        print(f"\n테스트 {i}/{len(test_questions)}")
        print("=" * 60)
        
        # RAG 쿼리 실행
        response = rag_system.query(question)
        
        # 결과 출력
        print(f"\n답변:")
        print("-" * 30)
        print(response.answer)
        
        print(f"\n처리 정보:")
        print(f"  총 처리 시간: {response.processing_time:.2f}초")
        print(f"  검색 시간: {response.retrieval_time:.2f}초")
        print(f"  생성 시간: {response.generation_time:.2f}초")
        print(f"  참조 문서 수: {len(response.source_documents)}")
        
        if response.source_documents:
            print(f"\n참조 문서:")
            for j, doc in enumerate(response.source_documents[:3]):
                print(f"    {j+1}. 카테고리: {doc.metadata.get('category', 'N/A')}")
                print(f"       내용: {doc.page_content[:100]}...")
        
        print("\n" + "=" * 60)

def compare_implementations():
    """순수 구현 vs LangChain 구현 비교"""
    print(f"\n구현 방식 비교")
    print("=" * 50)
    
    print("1. 순수 구현 (basic_rag.py)")
    print("   장점: 내부 동작 원리 명확히 이해")
    print("   단점: 코드 복잡성, 개발 시간")
    print("   코드 라인 수: ~600줄")
    
    print("\n2. LangChain 구현 (basic_rag_langchain.py)")
    print("   장점: 간결함, 검증된 컴포넌트")
    print("   단점: 프레임워크 의존성")
    print("   코드 라인 수: ~300줄")
    
    print(f"\n성능 특성:")
    print("- 검색 정확도: 유사함 (동일한 임베딩 모델)")
    print("- 개발 속도: LangChain 우세")
    print("- 커스터마이징: 순수 구현 우세")
    print("- 유지보수: LangChain 우세")

def demonstrate_different_search_strategies():
    """다양한 검색 전략 시연"""
    print(f"\n다양한 검색 전략 비교")
    print("=" * 50)
    
    rag_system = LangChainRAGSystem("langchain-rag-demo")
    test_question = "딥러닝과 머신러닝의 차이점은?"
    
    search_strategies = [
        ("similarity", "유사도 기반"),
        ("similarity_score_threshold", "임계값 기반"),
        ("mmr", "다양성 고려 (MMR)")
    ]
    
    for strategy, description in search_strategies:
        print(f"\n{description} 검색:")
        print("-" * 30)
        
        # 검색 전략 변경
        if strategy == "similarity_score_threshold":
            rag_system.update_retriever_config(strategy, k=5, score_threshold=0.3)
        elif strategy == "mmr":
            rag_system.update_retriever_config(strategy, k=5, fetch_k=10)
        else:
            rag_system.update_retriever_config(strategy, k=5)
        
        # 문서 검색만 실행 (빠른 비교를 위해)
        docs = rag_system.retriever.get_relevant_documents(test_question)
        
        print(f"검색된 문서 수: {len(docs)}")
        for i, doc in enumerate(docs[:3]):
            print(f"  {i+1}. {doc.metadata.get('category', 'N/A')}")
            print(f"     {doc.page_content[:80]}...")

def main():
    """메인 실행 함수"""
    try:
        # LangChain RAG 시스템 시연
        demonstrate_langchain_rag()
        
        # 구현 방식 비교
        compare_implementations()
        
        # 다양한 검색 전략
        demonstrate_different_search_strategies()
        
        print(f"\nLangChain RAG 실습 완료!")
        print("순수 구현(basic_rag.py)과 비교해보세요.")
        
    except Exception as e:
        print(f"실습 중 오류 발생: {e}")
        print("LangChain 설치 상태를 확인해주세요: pip install langchain langchain-openai langchain-community")

if __name__ == "__main__":
    main() 