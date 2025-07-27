"""
Lab 3 - Step 1: 기본 RAG 파이프라인
RAG(Retrieval-Augmented Generation) 시스템의 핵심 컴포넌트 구현
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.utils import embedding_functions
from shared.config import validate_api_keys, CHROMA_PERSIST_DIRECTORY, OPENAI_API_KEY, CHAT_MODEL, TEMPERATURE
from shared.utils import EmbeddingUtils, ChatUtils
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    """검색 결과를 담는 데이터 클래스"""
    document: str
    metadata: Dict[str, Any]
    distance: float
    relevance_score: float

@dataclass
class RAGResponse:
    """RAG 시스템의 최종 응답을 담는 데이터 클래스"""
    question: str
    answer: str
    sources: List[RetrievalResult]
    processing_time: float
    token_usage: Dict[str, int]

class QueryProcessor:
    """사용자 질문을 검색 쿼리로 변환하는 클래스"""
    
    def __init__(self):
        self.query_expansion_templates = [
            "핵심 키워드 추출: {query}",
            "관련 개념 확장: {query}",
            "동의어 포함: {query}"
        ]
    
    def process_query(self, user_query: str) -> str:
        """
        사용자 질문을 검색에 최적화된 쿼리로 변환
        """
        # 기본적으로는 원본 쿼리 사용 (향후 확장 가능)
        processed_query = user_query.strip()
        
        # 질문 형태를 서술형으로 변환 (선택적)
        if processed_query.endswith('?'):
            processed_query = processed_query[:-1]
        
        return processed_query
    
    def expand_query(self, query: str) -> List[str]:
        """
        쿼리 확장을 통해 다양한 검색 쿼리 생성
        """
        expanded_queries = [query]
        
        # 키워드 기반 확장
        keywords = self._extract_keywords(query)
        if keywords:
            expanded_queries.append(" ".join(keywords))
        
        return expanded_queries
    
    def _extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 핵심 키워드 추출 (간단한 구현)"""
        # 실제로는 더 정교한 키워드 추출 알고리즘 사용 가능
        words = text.split()
        # 간단한 불용어 제거
        stop_words = {'은', '는', '이', '가', '을', '를', '에', '의', '와', '과', '에서', '로', '으로', '에게', '한테'}
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        return keywords

class DocumentRetriever:
    """벡터 데이터베이스에서 관련 문서를 검색하는 클래스"""
    
    def __init__(self, collection_name: str = "ai-basic-rag"):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """ChromaDB 클라이언트 및 컬렉션 초기화"""
        try:
            self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
            
            # OpenAI 임베딩 함수 설정
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name="text-embedding-ada-002"
            )
            
            # 컬렉션 가져오기 또는 생성
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=openai_ef
                )
                print(f"기존 컬렉션 '{self.collection_name}' 로드됨")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=openai_ef,
                    metadata={"description": "RAG 시스템용 문서 컬렉션"}
                )
                print(f"새 컬렉션 '{self.collection_name}' 생성됨")
                
        except Exception as e:
            print(f"ChromaDB 초기화 오류: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """문서를 벡터 데이터베이스에 추가"""
        if not documents:
            return
        
        try:
            texts = [doc['content'] for doc in documents]
            ids = [doc.get('id', f"doc_{i}") for i, doc in enumerate(documents)]
            metadatas = [doc.get('metadata', {}) for doc in documents]
            
            self.collection.add(
                documents=texts,
                ids=ids,
                metadatas=metadatas
            )
            
            print(f"{len(documents)}개 문서가 추가되었습니다.")
            
        except Exception as e:
            print(f"문서 추가 중 오류: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[RetrievalResult]:
        """유사도 기반 문서 검색"""
        try:
            # ChromaDB 검색 실행
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=filters,
                include=["documents", "metadatas", "distances"]
            )
            
            # 결과를 RetrievalResult 객체로 변환
            retrieval_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # 거리를 관련성 점수로 변환 (0-1, 높을수록 관련성 높음)
                    relevance_score = max(0, 1 - distance)
                    
                    retrieval_results.append(RetrievalResult(
                        document=doc,
                        metadata=metadata or {},
                        distance=distance,
                        relevance_score=relevance_score
                    ))
            
            return retrieval_results
            
        except Exception as e:
            print(f"문서 검색 중 오류: {e}")
            return []

class ContextBuilder:
    """검색된 문서들을 컨텍스트로 구성하는 클래스"""
    
    def __init__(self, max_tokens: int = 3000):
        self.max_tokens = max_tokens
    
    def build_context(self, retrieval_results: List[RetrievalResult], 
                     relevance_threshold: float = 0.3) -> str:
        """
        검색 결과를 바탕으로 컨텍스트 구성
        """
        if not retrieval_results:
            return ""
        
        # 관련성 점수로 필터링
        filtered_results = [
            result for result in retrieval_results 
            if result.relevance_score >= relevance_threshold
        ]
        
        if not filtered_results:
            # 임계값을 통과한 결과가 없으면 가장 관련성 높은 결과 사용
            filtered_results = [max(retrieval_results, key=lambda x: x.relevance_score)]
        
        # 토큰 제한을 고려하여 컨텍스트 구성
        context_parts = []
        current_tokens = 0
        
        for i, result in enumerate(filtered_results):
            # 문서 내용과 메타데이터 포함
            doc_text = f"[문서 {i+1}]\n{result.document}\n"
            if result.metadata:
                doc_text += f"출처: {result.metadata}\n"
            
            # 대략적인 토큰 수 계산 (1토큰 ≈ 4자)
            estimated_tokens = len(doc_text) // 4
            
            if current_tokens + estimated_tokens > self.max_tokens:
                # 토큰 제한 초과 시 현재 문서를 잘라서 포함
                remaining_chars = (self.max_tokens - current_tokens) * 4
                if remaining_chars > 100:  # 최소 길이 확보
                    truncated_text = doc_text[:remaining_chars] + "...\n"
                    context_parts.append(truncated_text)
                break
            
            context_parts.append(doc_text)
            current_tokens += estimated_tokens
        
        return "\n".join(context_parts)
    
    def get_context_summary(self, context: str) -> Dict[str, Any]:
        """컨텍스트의 요약 정보 제공"""
        return {
            "character_count": len(context),
            "estimated_tokens": len(context) // 4,
            "document_count": context.count("[문서")
        }

class ResponseGenerator:
    """LLM을 사용해 최종 답변을 생성하는 클래스"""
    
    def __init__(self):
        self.default_template = """다음 맥락 정보를 바탕으로 질문에 답해주세요.

맥락 정보:
{context}

질문: {question}

답변할 때 다음 사항을 지켜주세요:
1. 제공된 맥락 정보를 바탕으로 정확하고 구체적으로 답변하세요
2. 맥락에 없는 정보는 추측하지 마세요
3. 출처가 명확한 정보를 우선적으로 활용하세요
4. 답변을 명확하고 이해하기 쉽게 구성하세요

답변:"""
    
    def generate_response(self, question: str, context: str, 
                         template: Optional[str] = None) -> Tuple[str, Dict[str, int]]:
        """
        질문과 컨텍스트를 바탕으로 답변 생성
        """
        if template is None:
            template = self.default_template
        
        # 프롬프트 구성
        prompt = template.format(context=context, question=question)
        
        # ChatGPT API 호출
        messages = [
            {"role": "system", "content": "당신은 정확하고 도움이 되는 AI 어시스턴트입니다."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = ChatUtils.get_chat_response(messages)
            
            # 토큰 사용량 추정 (실제 API에서는 정확한 값 제공)
            token_usage = {
                "prompt_tokens": len(prompt) // 4,
                "completion_tokens": len(response) // 4,
                "total_tokens": (len(prompt) + len(response)) // 4
            }
            
            return response, token_usage
            
        except Exception as e:
            print(f"답변 생성 중 오류: {e}")
            return f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {e}", {"error": True}

class BasicRAGSystem:
    """기본 RAG 시스템을 통합하는 메인 클래스"""
    
    def __init__(self, collection_name: str = "ai-basic-rag"):
        self.query_processor = QueryProcessor()
        self.retriever = DocumentRetriever(collection_name)
        self.context_builder = ContextBuilder()
        self.response_generator = ResponseGenerator()
        
        print(f"Basic RAG System 초기화 완료")
        print(f"컬렉션: {collection_name}")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """시스템에 문서 추가"""
        self.retriever.add_documents(documents)
    
    def query(self, question: str, top_k: int = 5, 
              relevance_threshold: float = 0.3) -> RAGResponse:
        """
        RAG 시스템을 통한 질의응답
        """
        start_time = time.time()
        
        print(f"\n질문: {question}")
        print("=" * 50)
        
        # 1. 쿼리 처리
        processed_query = self.query_processor.process_query(question)
        print(f"처리된 쿼리: {processed_query}")
        
        # 2. 문서 검색
        print(f"\n관련 문서 검색 중...")
        retrieval_results = self.retriever.search(processed_query, top_k=top_k)
        print(f"검색된 문서 수: {len(retrieval_results)}")
        
        if retrieval_results:
            print("검색 결과:")
            for i, result in enumerate(retrieval_results):
                print(f"  {i+1}. 관련성: {result.relevance_score:.3f}, "
                      f"거리: {result.distance:.3f}")
                print(f"     내용 미리보기: {result.document[:100]}...")
        
        # 3. 컨텍스트 구성
        print(f"\n컨텍스트 구성 중...")
        context = self.context_builder.build_context(retrieval_results, relevance_threshold)
        context_info = self.context_builder.get_context_summary(context)
        print(f"컨텍스트 정보: {context_info}")
        
        # 4. 답변 생성
        print(f"\n답변 생성 중...")
        answer, token_usage = self.response_generator.generate_response(question, context)
        
        # 5. 응답 시간 계산
        processing_time = time.time() - start_time
        
        # 6. 결과 구성
        rag_response = RAGResponse(
            question=question,
            answer=answer,
            sources=retrieval_results,
            processing_time=processing_time,
            token_usage=token_usage
        )
        
        return rag_response
    
    def get_system_stats(self) -> Dict[str, Any]:
        """시스템 통계 정보 반환"""
        try:
            doc_count = self.retriever.collection.count()
            return {
                "total_documents": doc_count,
                "collection_name": self.retriever.collection_name,
                "embedding_model": "text-embedding-ada-002",
                "chat_model": CHAT_MODEL
            }
        except:
            return {"error": "통계 정보를 가져올 수 없습니다"}

def create_sample_documents() -> List[Dict[str, Any]]:
    """테스트용 샘플 문서 생성"""
    documents = [
        {
            "id": "ai_intro_001",
            "content": """인공지능(AI)은 인간의 지능을 모방하여 기계가 학습하고 추론할 수 있게 하는 기술입니다. 
            AI는 머신러닝, 딥러닝, 자연어 처리, 컴퓨터 비전 등 다양한 분야를 포함합니다. 
            현재 AI는 의료진단, 자율주행, 언어번역, 이미지 인식 등 많은 영역에서 활용되고 있습니다.""",
            "metadata": {"category": "AI_기초", "author": "AI연구팀", "date": "2024-01-01"}
        },
        {
            "id": "ml_basics_001", 
            "content": """머신러닝은 데이터를 통해 컴퓨터가 자동으로 학습하는 인공지능의 한 분야입니다.
            지도학습, 비지도학습, 강화학습의 세 가지 주요 접근법이 있습니다.
            지도학습은 라벨이 있는 데이터로 훈련하며, 분류와 회귀 문제를 해결합니다.
            비지도학습은 라벨 없는 데이터에서 패턴을 찾는 방법입니다.""",
            "metadata": {"category": "머신러닝", "author": "ML연구팀", "date": "2024-01-02"}
        },
        {
            "id": "dl_concepts_001",
            "content": """딥러닝은 신경망을 여러 층으로 쌓아 복잡한 패턴을 학습하는 머신러닝 기법입니다.
            CNN(Convolutional Neural Network)은 이미지 처리에 특화되어 있고,
            RNN(Recurrent Neural Network)은 순차 데이터 처리에 적합합니다.
            트랜스포머 모델은 자연어 처리 분야에서 혁신적인 성과를 거두었습니다.""",
            "metadata": {"category": "딥러닝", "author": "DL연구팀", "date": "2024-01-03"}
        },
        {
            "id": "rag_system_001",
            "content": """RAG(Retrieval-Augmented Generation)는 검색과 생성을 결합한 AI 시스템입니다.
            먼저 관련 문서를 검색하여 컨텍스트를 구성하고, 이를 바탕으로 답변을 생성합니다.
            RAG는 할루시네이션을 줄이고 정확한 정보 기반 답변을 제공할 수 있습니다.
            벡터 데이터베이스와 대화형 AI 모델의 조합으로 구현됩니다.""",
            "metadata": {"category": "RAG", "author": "RAG연구팀", "date": "2024-01-04"}
        },
        {
            "id": "vector_db_001",
            "content": """벡터 데이터베이스는 고차원 벡터 데이터를 효율적으로 저장하고 검색하는 데이터베이스입니다.
            임베딩 벡터 간의 유사도를 계산하여 의미적으로 관련된 정보를 찾을 수 있습니다.
            코사인 유사도, 유클리드 거리 등의 메트릭을 사용하여 검색을 수행합니다.
            ChromaDB, Pinecone, Weaviate 등이 대표적인 벡터 데이터베이스입니다.""",
            "metadata": {"category": "벡터DB", "author": "데이터팀", "date": "2024-01-05"}
        }
    ]
    
    return documents

def demonstrate_basic_rag():
    """기본 RAG 시스템 시연"""
    print("Lab 3 - Step 1: 기본 RAG 파이프라인")
    print("RAG 시스템 핵심 기능 시연\n")
    
    # API 키 검증
    if not validate_api_keys():
        return
    
    # RAG 시스템 초기화
    print("RAG 시스템 초기화")
    print("=" * 40)
    rag_system = BasicRAGSystem("rag-demo")
    
    # 샘플 문서 추가
    print(f"\n샘플 문서 추가")
    print("=" * 40)
    sample_docs = create_sample_documents()
    rag_system.add_documents(sample_docs)
    
    # 시스템 통계
    stats = rag_system.get_system_stats()
    print(f"시스템 통계: {stats}")
    
    # 테스트 질문들
    test_questions = [
        "RAG 시스템이 무엇인가요?",
        "머신러닝의 주요 접근법은 무엇인가요?", 
        "딥러닝에서 사용되는 신경망 종류를 알려주세요",
        "벡터 데이터베이스는 어떻게 작동하나요?",
        "AI의 실제 응용 분야는 어떤 것들이 있나요?"
    ]
    
    # 각 질문에 대해 RAG 시스템 테스트
    for i, question in enumerate(test_questions, 1):
        print(f"\n테스트 {i}/{len(test_questions)}")
        print("=" * 60)
        
        # RAG 쿼리 실행
        response = rag_system.query(question, top_k=3)
        
        # 결과 출력
        print(f"\n답변:")
        print("-" * 30)
        print(response.answer)
        
        print(f"\n처리 정보:")
        print(f"  처리 시간: {response.processing_time:.2f}초")
        print(f"  토큰 사용량: {response.token_usage}")
        print(f"  참조 문서 수: {len(response.sources)}")
        
        if response.sources:
            print(f"\n참조 문서:")
            for j, source in enumerate(response.sources[:3]):
                print(f"    {j+1}. 관련성: {source.relevance_score:.3f}")
                print(f"       카테고리: {source.metadata.get('category', 'N/A')}")
        
        print("\n" + "=" * 60)

def analyze_rag_performance():
    """RAG 시스템 성능 분석"""
    print(f"\nRAG 시스템 성능 분석")
    print("=" * 40)
    
    rag_system = BasicRAGSystem("rag-demo")
    
    performance_questions = [
        "인공지능이란 무엇인가요?",
        "머신러닝과 딥러닝의 차이점은?",
        "RAG의 장점은 무엇인가요?"
    ]
    
    total_time = 0
    total_tokens = 0
    
    for question in performance_questions:
        response = rag_system.query(question, top_k=5)
        total_time += response.processing_time
        
        if isinstance(response.token_usage, dict) and 'total_tokens' in response.token_usage:
            total_tokens += response.token_usage['total_tokens']
    
    print(f"성능 분석 결과:")
    print(f"  평균 처리 시간: {total_time/len(performance_questions):.2f}초")
    print(f"  평균 토큰 사용량: {total_tokens/len(performance_questions):.0f}토큰")
    print(f"  총 질문 수: {len(performance_questions)}개")

def main():
    """메인 실행 함수"""
    try:
        # 기본 RAG 시스템 시연
        demonstrate_basic_rag()
        
        # 성능 분석
        analyze_rag_performance()
        
        print(f"\n실습 완료!")
        print("다음 단계에서는 고급 검색 기법을 학습합니다.")
        
    except Exception as e:
        print(f"실습 중 오류 발생: {e}")
        print("ChromaDB 설치 상태를 확인해주세요: pip install chromadb")

if __name__ == "__main__":
    main() 