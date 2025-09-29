"""
Lab 3 - Step 2: 고급 검색 기법 (LangChain 버전)
LangChain의 고급 검색 컴포넌트를 활용한 효율적인 구현
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

# 고급 검색 컴포넌트
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter, LLMChainExtractor
from langchain_community.document_transformers import EmbeddingsRedundantFilter

# SSL 검증 비활성화 HTTP 클라이언트
no_ssl_httpx = httpx.Client(verify=False)


@dataclass
class AdvancedSearchResult:
    """고급 검색 결과를 담는 데이터 클래스"""
    question: str
    answer: str
    source_documents: List[Document]
    processing_time: float
    retrieval_time: float
    generation_time: float
    retrieval_method: str
    num_retrieved_docs: int
    compression_ratio: float

class LangChainAdvancedRetriever:
    """LangChain 고급 검색 기법을 활용한 시스템"""
    
    def __init__(self, collection_name: str = "langchain-advanced-demo"):
        self.collection_name = collection_name
        
        # 기본 컴포넌트 초기화
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
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        
        # 벡터 스토어 설정
        self.vectorstore = None
        self.documents = []  # BM25를 위한 문서 저장
        
        self._setup_vectorstore()
        self._setup_retrievers()
        
        print(f"LangChain Advanced Retriever 초기화 완료")
    
    def _setup_vectorstore(self):
        """벡터 스토어 설정"""
        try:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=CHROMA_PERSIST_DIRECTORY
            )
            print("벡터 스토어 설정 완료")
        except Exception as e:
            print(f"벡터 스토어 설정 실패: {e}")
            raise
    
    def _setup_retrievers(self):
        """다양한 검색기 설정"""
        if self.vectorstore is None:
            return
        
        # 1. 기본 유사도 검색기
        self.similarity_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # 2. MMR 검색기 (다양성 고려)
        self.mmr_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.7}
        )
        
        # 3. 임계값 기반 검색기
        self.threshold_retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.3, "k": 5}
        )
        
        print("기본 검색기들 설정 완료")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """문서 추가 및 모든 검색기 업데이트"""
        if not documents:
            return
        
        try:
            # LangChain Document 변환
            langchain_docs = []
            for doc in documents:
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
                if 'id' in doc:
                    metadata['source'] = doc['id']
                
                document = Document(page_content=content, metadata=metadata)
                langchain_docs.append(document)
            
            # 텍스트 분할
            split_docs = self.text_splitter.split_documents(langchain_docs)
            
            # 벡터 스토어에 추가
            self.vectorstore.add_documents(split_docs)
            
            # BM25를 위한 문서 저장
            self.documents.extend(split_docs)
            
            print(f"{len(documents)}개 문서 ({len(split_docs)}개 청크) 추가 완료")
            
        except Exception as e:
            print(f"문서 추가 실패: {e}")
            raise
    
    def get_multi_query_retriever(self) -> MultiQueryRetriever:
        """다중 쿼리 검색기 생성"""
        return MultiQueryRetriever.from_llm(
            retriever=self.similarity_retriever,
            llm=self.llm,
            parser_key="lines"  # 여러 쿼리를 줄바꿈으로 구분
        )
    
    def get_ensemble_retriever(self) -> EnsembleRetriever:
        """앙상블 검색기 생성 (벡터 + BM25)"""
        if not self.documents:
            print("문서가 없어 BM25 검색기를 생성할 수 없습니다.")
            return None
        
        try:
            # BM25 검색기 생성
            bm25_retriever = BM25Retriever.from_documents(self.documents)
            bm25_retriever.k = 5
            
            # 앙상블 검색기 (벡터 검색 70% + BM25 30%)
            ensemble_retriever = EnsembleRetriever(
                retrievers=[self.similarity_retriever, bm25_retriever],
                weights=[0.7, 0.3]
            )
            
            return ensemble_retriever
            
        except Exception as e:
            print(f"앙상블 검색기 생성 실패: {e}")
            return None
    
    def get_compression_retriever(self, compression_type: str = "embeddings") -> ContextualCompressionRetriever:
        """압축 검색기 생성"""
        
        if compression_type == "embeddings":
            # 임베딩 기반 필터
            embeddings_filter = EmbeddingsFilter(
                embeddings=self.embeddings,
                similarity_threshold=0.3
            )
            
            # 중복 제거 필터
            redundant_filter = EmbeddingsRedundantFilter(
                embeddings=self.embeddings,
                similarity_threshold=0.8
            )
            
            # 압축 파이프라인
            pipeline_compressor = DocumentCompressorPipeline(
                transformers=[embeddings_filter, redundant_filter]
            )
            
        elif compression_type == "llm":
            # LLM 기반 추출
            pipeline_compressor = LLMChainExtractor.from_llm(self.llm)
        
        else:
            # 기본: 임베딩 필터만
            pipeline_compressor = EmbeddingsFilter(
                embeddings=self.embeddings,
                similarity_threshold=0.3
            )
        
        return ContextualCompressionRetriever(
            base_compressor=pipeline_compressor,
            base_retriever=self.similarity_retriever
        )
    
    def search_with_method(self, question: str, method: str = "similarity") -> AdvancedSearchResult:
        """지정된 방법으로 검색 및 답변 생성"""
        
        start_time = time.time()
        
        print(f"\n질문: {question}")
        print(f"검색 방법: {method}")
        print("=" * 50)
        
        try:
            # 검색기 선택
            retrieval_start = time.time()
            
            if method == "similarity":
                retriever = self.similarity_retriever
                retrieved_docs = retriever.get_relevant_documents(question)
                
            elif method == "mmr":
                retriever = self.mmr_retriever
                retrieved_docs = retriever.get_relevant_documents(question)
                
            elif method == "threshold":
                retriever = self.threshold_retriever
                retrieved_docs = retriever.get_relevant_documents(question)
                
            elif method == "multi_query":
                retriever = self.get_multi_query_retriever()
                retrieved_docs = retriever.get_relevant_documents(question)
                
            elif method == "ensemble":
                retriever = self.get_ensemble_retriever()
                if retriever is None:
                    retrieved_docs = []
                else:
                    retrieved_docs = retriever.get_relevant_documents(question)
                
            elif method == "compression":
                retriever = self.get_compression_retriever("embeddings")
                retrieved_docs = retriever.get_relevant_documents(question)
                
            elif method == "llm_compression":
                retriever = self.get_compression_retriever("llm")
                retrieved_docs = retriever.get_relevant_documents(question)
                
            else:
                retrieved_docs = []
            
            retrieval_time = time.time() - retrieval_start
            
            print(f"검색 시간: {retrieval_time:.3f}초")
            print(f"검색된 문서 수: {len(retrieved_docs)}")
            
            # 압축 비율 계산 (압축 방법인 경우)
            compression_ratio = 1.0
            if "compression" in method and len(retrieved_docs) > 0:
                # 원본 대비 압축된 문서 비율 추정
                original_length = sum(len(doc.page_content) for doc in retrieved_docs)
                # 실제로는 원본 검색 결과와 비교해야 하지만, 여기서는 근사치 사용
                compression_ratio = min(1.0, original_length / (original_length + 100))
            
            # 답변 생성
            generation_start = time.time()
            
            if retrieved_docs:
                # 컨텍스트 구성
                context = "\n\n".join([
                    f"[문서 {i+1}]\n{doc.page_content}\n출처: {doc.metadata}"
                    for i, doc in enumerate(retrieved_docs[:5])
                ])
                
                # 프롬프트 생성
                prompt = f"""
다음 맥락 정보를 바탕으로 질문에 답해주세요.

맥락 정보:
{context}

질문: {question}

답변:"""
                
                messages = [
                    {"role": "system", "content": "당신은 정확하고 도움이 되는 AI 어시스턴트입니다."},
                    {"role": "user", "content": prompt}
                ]
                
                # ChatOpenAI 직접 사용
                chat_response = self.llm.invoke(messages)
                answer = chat_response.content if hasattr(chat_response, 'content') else str(chat_response)
                
            else:
                answer = "관련 문서를 찾을 수 없어 답변을 생성할 수 없습니다."
            
            generation_time = time.time() - generation_start
            total_time = time.time() - start_time
            
            print(f"생성 시간: {generation_time:.3f}초")
            print(f"총 처리 시간: {total_time:.3f}초")
            
            return AdvancedSearchResult(
                question=question,
                answer=answer,
                source_documents=retrieved_docs,
                processing_time=total_time,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                retrieval_method=method,
                num_retrieved_docs=len(retrieved_docs),
                compression_ratio=compression_ratio
            )
            
        except Exception as e:
            print(f"검색 실패: {e}")
            return AdvancedSearchResult(
                question=question,
                answer=f"검색 중 오류 발생: {e}",
                source_documents=[],
                processing_time=time.time() - start_time,
                retrieval_time=0,
                generation_time=0,
                retrieval_method=method,
                num_retrieved_docs=0,
                compression_ratio=1.0
            )

def create_advanced_sample_documents() -> List[Dict[str, Any]]:
    """고급 검색 테스트용 문서 생성"""
    documents = [
        {
            "id": "advanced_rag_001",
            "content": """고급 RAG 기법에는 여러 접근법이 있습니다. 
            다중 쿼리 검색은 하나의 질문을 여러 관점에서 재작성하여 더 풍부한 정보를 찾습니다.
            앙상블 검색은 벡터 검색과 키워드 검색을 결합하여 정확도를 높입니다.
            컨텍스트 압축은 관련성 높은 정보만 선별하여 토큰 효율성을 개선합니다.""",
            "metadata": {"category": "고급RAG", "author": "RAG전문가", "date": "2024-01-15"}
        },
        {
            "id": "langchain_features_001",
            "content": """LangChain은 다양한 고급 검색 기능을 제공합니다.
            MultiQueryRetriever는 LLM을 사용해 쿼리를 확장합니다.
            EnsembleRetriever는 여러 검색 방식을 조합합니다.
            ContextualCompressionRetriever는 관련성 기반으로 문서를 압축합니다.
            이러한 기능들을 조합하여 더 정확한 RAG 시스템을 구축할 수 있습니다.""",
            "metadata": {"category": "LangChain", "author": "개발팀", "date": "2024-01-18"}
        },
        {
            "id": "retrieval_strategies_001",
            "content": """효과적인 검색 전략에는 다음이 포함됩니다:
            1. 유사도 검색: 임베딩 벡터 간 코사인 유사도 활용
            2. MMR 검색: 관련성과 다양성의 균형 고려
            3. 임계값 검색: 최소 유사도 기준 적용
            4. 하이브리드 검색: 여러 방법의 장점 결합
            각 방법은 데이터와 사용 사례에 따라 성능이 달라집니다.""",
            "metadata": {"category": "검색전략", "author": "연구팀", "date": "2024-01-20"}
        },
        {
            "id": "performance_optimization_001",
            "content": """RAG 성능 최적화를 위한 핵심 원칙:
            검색 정확도와 속도의 균형 유지
            컨텍스트 윈도우 효율적 활용
            중복 정보 제거 및 압축
            실시간 처리를 위한 캐싱 전략
            사용자 피드백을 통한 지속적 개선
            이러한 원칙들을 통해 실용적인 RAG 시스템을 구축할 수 있습니다.""",
            "metadata": {"category": "성능최적화", "author": "엔지니어링팀", "date": "2024-01-22"}
        }
    ]
    
    return documents

def demonstrate_advanced_retrieval_methods():
    """다양한 고급 검색 방법 시연"""
    print("Lab 3 - Step 2: 고급 검색 기법 (LangChain 버전)")
    print("LangChain 고급 검색 컴포넌트 활용\n")
    
    if not validate_api_keys():
        return
    
    # 고급 검색 시스템 초기화
    retriever_system = LangChainAdvancedRetriever("langchain-advanced-demo")
    
    # 샘플 문서 추가
    sample_docs = create_advanced_sample_documents()
    retriever_system.add_documents(sample_docs)
    
    # 테스트 질문
    test_question = "RAG 시스템의 성능을 어떻게 향상시킬 수 있나요?"
    
    # 다양한 검색 방법 테스트
    methods = [
        ("similarity", "기본 유사도 검색"),
        ("mmr", "MMR 검색 (다양성 고려)"),
        ("threshold", "임계값 기반 검색"),
        ("multi_query", "다중 쿼리 검색"),
        ("ensemble", "앙상블 검색 (벡터 + BM25)"),
        ("compression", "압축 검색 (임베딩 기반)"),
    ]
    
    results = []
    
    for method, description in methods:
        print(f"\n{description}")
        print("=" * 50)
        
        try:
            result = retriever_system.search_with_method(test_question, method)
            results.append((method, result))
            
            # 결과 요약 출력
            print(f"\n답변 (처음 200자):")
            print(result.answer[:200] + "..." if len(result.answer) > 200 else result.answer)
            
            print(f"\n성능 지표:")
            print(f"  검색된 문서 수: {result.num_retrieved_docs}")
            print(f"  검색 시간: {result.retrieval_time:.3f}초")
            print(f"  생성 시간: {result.generation_time:.3f}초")
            print(f"  총 처리 시간: {result.processing_time:.3f}초")
            
            if result.compression_ratio < 1.0:
                print(f"  압축 비율: {result.compression_ratio:.3f}")
            
        except Exception as e:
            print(f"오류 발생: {e}")
            continue
        
        print("\n" + "=" * 50)
    
    # 성능 비교 요약
    if results:
        print(f"\n성능 비교 요약")
        print("=" * 60)
        
        for method, result in results:
            print(f"{method:15s}: "
                  f"검색 {result.retrieval_time:.3f}s, "
                  f"생성 {result.generation_time:.3f}s, "
                  f"문서 {result.num_retrieved_docs}개")

def demonstrate_query_expansion():
    """쿼리 확장 기법 시연"""
    print(f"\n쿼리 확장 기법 시연")
    print("=" * 50)
    
    retriever_system = LangChainAdvancedRetriever("langchain-advanced-demo")
    
    # 간단한 질문으로 다중 쿼리 검색 테스트
    simple_question = "LangChain이 뭐야?"
    
    print(f"원본 질문: '{simple_question}'")
    print()
    
    try:
        # 다중 쿼리 검색기 생성
        multi_query_retriever = retriever_system.get_multi_query_retriever()
        
        # 검색 실행 (내부적으로 쿼리 확장됨)
        docs = multi_query_retriever.get_relevant_documents(simple_question)
        
        print(f"다중 쿼리 검색 결과:")
        print(f"검색된 문서 수: {len(docs)}")
        
        for i, doc in enumerate(docs[:3]):
            print(f"\n문서 {i+1}:")
            print(f"카테고리: {doc.metadata.get('category', 'N/A')}")
            print(f"내용: {doc.page_content[:150]}...")
            
    except Exception as e:
        print(f"다중 쿼리 검색 실패: {e}")

def compare_with_basic_implementation():
    """기본 구현과 LangChain 고급 기능 비교"""
    print(f"\n기본 구현 vs LangChain 고급 기능 비교")
    print("=" * 60)
    
    comparison_data = [
        ("기능", "기본 구현", "LangChain 고급"),
        ("코드 복잡도", "높음 (직접 구현)", "낮음 (내장 기능)"),
        ("다중 쿼리", "수동 구현 필요", "MultiQueryRetriever"),
        ("앙상블 검색", "복잡한 로직", "EnsembleRetriever"),
        ("문서 압축", "커스텀 필터", "ContextualCompressionRetriever"),
        ("유지보수", "직접 관리", "프레임워크 지원"),
        ("확장성", "높음 (맞춤형)", "중간 (표준화)"),
        ("학습 곡선", "가파름", "완만함"),
        ("디버깅", "어려움", "쉬움 (추상화)")
    ]
    
    for row in comparison_data:
        print(f"{row[0]:12s} | {row[1]:20s} | {row[2]:25s}")
    
    print(f"\n권장사항:")
    print("- 학습 목적: 기본 구현으로 원리 이해")
    print("- 실무 프로젝트: LangChain 고급 기능 활용")
    print("- 특수 요구사항: 기본 구현 + LangChain 하이브리드")

def main():
    """메인 실행 함수"""
    try:
        # 고급 검색 방법 시연
        demonstrate_advanced_retrieval_methods()
        
        # 쿼리 확장 기법 시연
        demonstrate_query_expansion()
        
        # 구현 방식 비교
        compare_with_basic_implementation()
        
        print(f"\nLangChain 고급 검색 기법 실습 완료!")
        print("기본 구현(advanced_retrieval.py)과 비교해보세요.")
        
    except Exception as e:
        print(f"실습 중 오류 발생: {e}")
        print("필요한 패키지를 설치해주세요:")
        print("pip install langchain langchain-openai langchain-community rank_bm25")

if __name__ == "__main__":
    main() 