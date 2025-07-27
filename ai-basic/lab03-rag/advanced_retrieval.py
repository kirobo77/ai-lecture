"""
Lab 3 - Step 2: 고급 검색 기법
다양한 검색 전략과 최적화 기법을 구현
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.utils import embedding_functions
from shared.config import validate_api_keys, CHROMA_PERSIST_DIRECTORY, OPENAI_API_KEY, CHAT_MODEL
from shared.utils import EmbeddingUtils, ChatUtils
import numpy as np
import time
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from collections import Counter
import math

@dataclass
class SearchResult:
    """검색 결과를 담는 데이터 클래스"""
    document_id: str
    content: str
    metadata: Dict[str, Any]
    semantic_score: float
    keyword_score: float
    combined_score: float
    rank: int

class AdvancedQueryProcessor:
    """고급 쿼리 처리를 위한 클래스"""
    
    def __init__(self):
        self.stop_words = {
            '이', '가', '은', '는', '을', '를', '에', '의', '와', '과', '도', '에서', 
            '로', '으로', '에게', '한테', '께', '부터', '까지', '마다', '보다', '처럼',
            '같이', '하고', '하지만', '그리고', '또는', '그런데', '그러나', '따라서'
        }
        
    def expand_query(self, query: str, method: str = "synonym") -> List[str]:
        """
        쿼리 확장을 통한 검색 성능 향상
        """
        expanded_queries = [query]
        
        if method == "synonym":
            # 동의어 확장 (실제로는 더 정교한 동의어 사전 사용)
            synonym_map = {
                "AI": ["인공지능", "artificial intelligence", "머신러닝"],
                "머신러닝": ["ML", "기계학습", "machine learning"],
                "딥러닝": ["deep learning", "심층학습", "신경망"],
                "RAG": ["검색증강생성", "retrieval augmented generation"],
                "벡터": ["vector", "임베딩", "embedding"]
            }
            
            for term, synonyms in synonym_map.items():
                if term.lower() in query.lower():
                    for synonym in synonyms:
                        expanded_query = query.replace(term, synonym)
                        expanded_queries.append(expanded_query)
        
        elif method == "keyword":
            # 키워드 기반 확장
            keywords = self._extract_keywords(query)
            if len(keywords) > 1:
                expanded_queries.append(" ".join(keywords))
                
        elif method == "contextual":
            # 문맥적 확장 (GPT를 활용한 쿼리 확장)
            expanded_queries.extend(self._generate_contextual_queries(query))
        
        return list(set(expanded_queries))  # 중복 제거
    
    def _extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 핵심 키워드 추출"""
        # 한글, 영문, 숫자만 추출
        words = re.findall(r'[가-힣a-zA-Z0-9]+', text)
        
        # 불용어 제거 및 길이 필터링
        keywords = [
            word for word in words 
            if word.lower() not in self.stop_words and len(word) > 1
        ]
        
        return keywords
    
    def _generate_contextual_queries(self, query: str) -> List[str]:
        """GPT를 활용한 문맥적 쿼리 확장"""
        try:
            prompt = f"""
다음 질문과 관련된 검색 쿼리를 3개 생성해주세요. 
원래 질문과 같은 의미지만 다른 표현으로 작성해주세요.

원래 질문: {query}

검색 쿼리 (한 줄에 하나씩):
"""
            
            messages = [
                {"role": "system", "content": "당신은 검색 쿼리 최적화 전문가입니다."},
                {"role": "user", "content": prompt}
            ]
            
            response = ChatUtils.get_chat_response(messages)
            
            # 응답에서 쿼리 추출
            queries = [
                line.strip() 
                for line in response.split('\n') 
                if line.strip() and not line.startswith(('1.', '2.', '3.', '-', '*'))
            ]
            
            return queries[:3]  # 최대 3개만 반환
            
        except Exception as e:
            print(f"문맥적 쿼리 확장 실패: {e}")
            return []

class HybridRetriever:
    """시맨틱 검색과 키워드 검색을 결합한 하이브리드 검색"""
    
    def __init__(self, collection_name: str = "ai-basic-rag"):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.documents_cache = {}  # 문서 캐싱
        self._initialize_client()
        self._build_keyword_index()
    
    def _initialize_client(self):
        """ChromaDB 클라이언트 초기화"""
        try:
            self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
            
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name="text-embedding-ada-002"
            )
            
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=openai_ef
                )
                print(f"컬렉션 '{self.collection_name}' 로드됨")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=openai_ef
                )
                print(f"새 컬렉션 '{self.collection_name}' 생성됨")
                
        except Exception as e:
            print(f"ChromaDB 초기화 오류: {e}")
            raise
    
    def _build_keyword_index(self):
        """키워드 기반 검색을 위한 인덱스 구축"""
        try:
            # 컬렉션의 모든 문서 가져오기
            all_docs = self.collection.get(include=["documents", "metadatas"])
            
            self.keyword_index = {}
            self.documents_cache = {}
            
            for i, (doc_id, content, metadata) in enumerate(zip(
                all_docs['ids'],
                all_docs['documents'], 
                all_docs['metadatas']
            )):
                # 문서 캐싱
                self.documents_cache[doc_id] = {
                    'content': content,
                    'metadata': metadata or {}
                }
                
                # 키워드 인덱스 구축
                words = re.findall(r'[가-힣a-zA-Z0-9]+', content.lower())
                word_counts = Counter(words)
                
                for word, count in word_counts.items():
                    if word not in self.keyword_index:
                        self.keyword_index[word] = {}
                    self.keyword_index[word][doc_id] = count
            
            print(f"키워드 인덱스 구축 완료: {len(self.keyword_index)}개 단어")
            
        except Exception as e:
            print(f"키워드 인덱스 구축 실패: {e}")
            self.keyword_index = {}
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """시맨틱(의미적) 검색"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            semantic_results = []
            if results['documents'] and results['documents'][0]:
                for doc_id, content, metadata, distance in zip(
                    results['ids'][0],
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                ):
                    semantic_score = max(0, 1 - distance)  # 거리를 점수로 변환
                    semantic_results.append({
                        'id': doc_id,
                        'content': content,
                        'metadata': metadata or {},
                        'semantic_score': semantic_score
                    })
            
            return semantic_results
            
        except Exception as e:
            print(f"시맨틱 검색 오류: {e}")
            return []
    
    def keyword_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """키워드 기반 검색 (BM25 유사 알고리즘)"""
        if not self.keyword_index:
            return []
        
        # 쿼리에서 키워드 추출
        query_words = re.findall(r'[가-힣a-zA-Z0-9]+', query.lower())
        
        # 문서별 점수 계산
        doc_scores = {}
        total_docs = len(self.documents_cache)
        
        for word in query_words:
            if word in self.keyword_index:
                # IDF 계산 (문서 빈도의 역수)
                docs_with_word = len(self.keyword_index[word])
                idf = math.log(total_docs / docs_with_word)
                
                for doc_id, term_freq in self.keyword_index[word].items():
                    # TF-IDF 점수 계산
                    doc_length = len(self.documents_cache[doc_id]['content'].split())
                    tf = term_freq / doc_length
                    score = tf * idf
                    
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = 0
                    doc_scores[doc_id] += score
        
        # 점수 순으로 정렬
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        keyword_results = []
        for doc_id, score in sorted_docs[:top_k]:
            doc_info = self.documents_cache[doc_id]
            keyword_results.append({
                'id': doc_id,
                'content': doc_info['content'],
                'metadata': doc_info['metadata'],
                'keyword_score': score
            })
        
        return keyword_results
    
    def hybrid_search(self, query: str, top_k: int = 10, 
                     semantic_weight: float = 0.7) -> List[SearchResult]:
        """하이브리드 검색: 시맨틱 + 키워드 검색 결합"""
        
        print(f"하이브리드 검색 실행: '{query}'")
        print(f"시맨틱 가중치: {semantic_weight}, 키워드 가중치: {1-semantic_weight}")
        
        # 각각의 검색 결과 얻기
        semantic_results = self.semantic_search(query, top_k * 2)
        keyword_results = self.keyword_search(query, top_k * 2)
        
        print(f"시맨틱 검색 결과: {len(semantic_results)}개")
        print(f"키워드 검색 결과: {len(keyword_results)}개")
        
        # 결과 통합
        combined_results = {}
        
        # 시맨틱 검색 결과 추가
        for result in semantic_results:
            doc_id = result['id']
            combined_results[doc_id] = {
                'id': doc_id,
                'content': result['content'],
                'metadata': result['metadata'],
                'semantic_score': result['semantic_score'],
                'keyword_score': 0.0
            }
        
        # 키워드 검색 결과 통합
        for result in keyword_results:
            doc_id = result['id']
            if doc_id in combined_results:
                combined_results[doc_id]['keyword_score'] = result['keyword_score']
            else:
                combined_results[doc_id] = {
                    'id': doc_id,
                    'content': result['content'],
                    'metadata': result['metadata'],
                    'semantic_score': 0.0,
                    'keyword_score': result['keyword_score']
                }
        
        # 정규화 및 최종 점수 계산
        semantic_scores = [r['semantic_score'] for r in combined_results.values()]
        keyword_scores = [r['keyword_score'] for r in combined_results.values()]
        
        # 점수 정규화 (0-1 범위)
        if semantic_scores and max(semantic_scores) > 0:
            semantic_max = max(semantic_scores)
            for result in combined_results.values():
                result['semantic_score'] = result['semantic_score'] / semantic_max
        
        if keyword_scores and max(keyword_scores) > 0:
            keyword_max = max(keyword_scores)
            for result in combined_results.values():
                result['keyword_score'] = result['keyword_score'] / keyword_max
        
        # 최종 결합 점수 계산
        final_results = []
        for result in combined_results.values():
            combined_score = (
                semantic_weight * result['semantic_score'] + 
                (1 - semantic_weight) * result['keyword_score']
            )
            
            final_results.append(SearchResult(
                document_id=result['id'],
                content=result['content'],
                metadata=result['metadata'],
                semantic_score=result['semantic_score'],
                keyword_score=result['keyword_score'],
                combined_score=combined_score,
                rank=0  # 나중에 설정
            ))
        
        # 최종 점수로 정렬
        final_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # 순위 설정
        for i, result in enumerate(final_results):
            result.rank = i + 1
        
        return final_results[:top_k]

class QueryRewriter:
    """쿼리 재작성을 통한 검색 성능 향상"""
    
    def __init__(self):
        pass
    
    def rewrite_query(self, original_query: str, method: str = "clarification") -> str:
        """
        쿼리 재작성
        """
        if method == "clarification":
            return self._clarify_query(original_query)
        elif method == "expansion":
            return self._expand_query(original_query)
        elif method == "simplification":
            return self._simplify_query(original_query)
        else:
            return original_query
    
    def _clarify_query(self, query: str) -> str:
        """쿼리 명확화"""
        try:
            prompt = f"""
다음 질문을 더 명확하고 구체적으로 다시 작성해주세요.
검색에 최적화된 형태로 변환하되, 원래 의도는 유지해주세요.

원본 질문: {query}

개선된 질문:"""
            
            messages = [
                {"role": "system", "content": "당신은 검색 쿼리 최적화 전문가입니다."},
                {"role": "user", "content": prompt}
            ]
            
            return ChatUtils.get_chat_response(messages).strip()
            
        except Exception as e:
            print(f"쿼리 명확화 실패: {e}")
            return query
    
    def _expand_query(self, query: str) -> str:
        """쿼리 확장"""
        # 간단한 키워드 기반 확장
        keywords = re.findall(r'[가-힣a-zA-Z0-9]+', query)
        if len(keywords) > 1:
            return f"{query} {' '.join(keywords)}"
        return query
    
    def _simplify_query(self, query: str) -> str:
        """쿼리 단순화"""
        # 핵심 키워드만 추출
        keywords = re.findall(r'[가-힣a-zA-Z0-9]+', query)
        important_words = [word for word in keywords if len(word) > 2]
        return ' '.join(important_words[:3])  # 최대 3개 키워드만

class ReRanker:
    """검색 결과 재순위 매기기"""
    
    def __init__(self):
        pass
    
    def rerank_results(self, query: str, results: List[SearchResult], 
                      method: str = "relevance") -> List[SearchResult]:
        """
        검색 결과 재순위
        """
        if method == "relevance":
            return self._rerank_by_relevance(query, results)
        elif method == "diversity":
            return self._rerank_by_diversity(results)
        elif method == "recency":
            return self._rerank_by_recency(results)
        else:
            return results
    
    def _rerank_by_relevance(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """관련성 기반 재순위"""
        query_words = set(re.findall(r'[가-힣a-zA-Z0-9]+', query.lower()))
        
        for result in results:
            content_words = set(re.findall(r'[가-힣a-zA-Z0-9]+', result.content.lower()))
            
            # 쿼리와 문서 간 단어 겹침 비율 계산
            overlap = len(query_words.intersection(content_words))
            total_words = len(query_words.union(content_words))
            
            if total_words > 0:
                overlap_score = overlap / total_words
                # 기존 점수와 겹침 점수 결합
                result.combined_score = 0.7 * result.combined_score + 0.3 * overlap_score
        
        # 새로운 점수로 재정렬
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # 순위 재설정
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results
    
    def _rerank_by_diversity(self, results: List[SearchResult]) -> List[SearchResult]:
        """다양성 기반 재순위 (중복 내용 제거)"""
        diverse_results = []
        used_categories = set()
        
        # 높은 점수 순으로 정렬된 결과에서 다양한 카테고리 선택
        for result in results:
            category = result.metadata.get('category', 'unknown')
            
            if category not in used_categories or len(diverse_results) < 3:
                diverse_results.append(result)
                used_categories.add(category)
        
        # 나머지 결과 추가 (점수 순)
        remaining = [r for r in results if r not in diverse_results]
        diverse_results.extend(remaining)
        
        # 순위 재설정
        for i, result in enumerate(diverse_results):
            result.rank = i + 1
        
        return diverse_results
    
    def _rerank_by_recency(self, results: List[SearchResult]) -> List[SearchResult]:
        """최신성 기반 재순위"""
        # 날짜 정보가 있는 경우에만 적용
        dated_results = []
        undated_results = []
        
        for result in results:
            if 'date' in result.metadata:
                dated_results.append(result)
            else:
                undated_results.append(result)
        
        # 날짜순 정렬 (최신순)
        dated_results.sort(
            key=lambda x: x.metadata.get('date', '1900-01-01'), 
            reverse=True
        )
        
        # 최신 문서에 보너스 점수 부여
        for i, result in enumerate(dated_results):
            recency_bonus = (len(dated_results) - i) / len(dated_results) * 0.1
            result.combined_score += recency_bonus
        
        # 전체 결과 결합 및 재정렬
        all_results = dated_results + undated_results
        all_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # 순위 재설정
        for i, result in enumerate(all_results):
            result.rank = i + 1
        
        return all_results

def create_advanced_sample_documents() -> List[Dict[str, Any]]:
    """고급 검색 테스트용 문서 생성"""
    documents = [
        {
            "id": "ai_overview_2024",
            "content": """2024년 인공지능 기술 동향: 생성형 AI와 대규모 언어모델(LLM)이 주목받고 있습니다. 
            ChatGPT, GPT-4, Claude와 같은 모델들이 다양한 산업에 혁신을 가져오고 있습니다. 
            특히 자연어 처리, 코드 생성, 창작 활동에서 뛰어난 성능을 보이고 있습니다.""",
            "metadata": {"category": "AI_동향", "author": "테크리뷰", "date": "2024-01-15", "type": "article"}
        },
        {
            "id": "rag_implementation_guide",
            "content": """RAG 시스템 구현 가이드: 검색 증강 생성(Retrieval-Augmented Generation)은 
            외부 지식 베이스를 활용하여 LLM의 답변 품질을 향상시키는 기법입니다. 
            벡터 데이터베이스, 임베딩 모델, 재순위 알고리즘이 핵심 구성 요소입니다.""",
            "metadata": {"category": "RAG", "author": "AI개발팀", "date": "2024-01-10", "type": "tutorial"}
        },
        {
            "id": "vector_search_optimization",
            "content": """벡터 검색 최적화 기법: 고차원 임베딩 벡터의 유사도 검색 성능을 향상시키는 방법들을 소개합니다. 
            HNSW, IVF, PQ 등의 인덱싱 알고리즘과 거리 메트릭 선택의 중요성을 다룹니다. 
            실시간 검색과 정확도 사이의 트레이드오프를 고려한 최적화 전략이 필요합니다.""",
            "metadata": {"category": "벡터검색", "author": "검색엔진팀", "date": "2024-01-08", "type": "technical"}
        },
        {
            "id": "embedding_models_comparison",
            "content": """임베딩 모델 성능 비교: OpenAI의 text-embedding-ada-002, Sentence-BERT, E5 등 
            다양한 임베딩 모델의 특성과 성능을 비교 분석합니다. 
            한국어 처리 성능, 계산 비용, 차원 수에 따른 장단점을 살펴봅니다.""",
            "metadata": {"category": "임베딩", "author": "NLP연구소", "date": "2024-01-05", "type": "research"}
        },
        {
            "id": "hybrid_search_strategies",
            "content": """하이브리드 검색 전략: 시맨틱 검색과 키워드 검색을 효과적으로 결합하는 방법론입니다. 
            BM25와 벡터 검색의 조합, 가중치 조절, 결과 통합 알고리즘을 다룹니다. 
            사용자 쿼리 유형에 따른 적응형 검색 전략 구현이 핵심입니다.""",
            "metadata": {"category": "검색전략", "author": "검색최적화팀", "date": "2024-01-12", "type": "methodology"}
        },
        {
            "id": "llm_fine_tuning_guide",
            "content": """대규모 언어모델 파인튜닝 가이드: LoRA, QLoRA를 활용한 효율적인 모델 커스터마이징 방법을 설명합니다. 
            도메인 특화 데이터셋 구축, 학습 하이퍼파라미터 튜닝, 성능 평가 메트릭을 포함합니다. 
            리소스 제약 환경에서의 실용적인 접근법을 제시합니다.""",
            "metadata": {"category": "모델튜닝", "author": "ML엔지니어링팀", "date": "2024-01-20", "type": "guide"}
        }
    ]
    
    return documents

def demonstrate_search_strategies():
    """다양한 검색 전략 시연"""
    print("Lab 3 - Step 2: 고급 검색 기법")
    print("다양한 검색 전략 비교 분석\n")
    
    if not validate_api_keys():
        return
    
    # 하이브리드 검색 시스템 초기화
    print("하이브리드 검색 시스템 초기화")
    print("=" * 50)
    
    retriever = HybridRetriever("advanced-search-demo")
    
    # 샘플 문서 추가
    sample_docs = create_advanced_sample_documents()
    
    # 기존 문서가 있는지 확인
    try:
        existing_count = retriever.collection.count()
        if existing_count == 0:
            print("샘플 문서 추가 중...")
            for doc in sample_docs:
                retriever.collection.add(
                    documents=[doc['content']],
                    ids=[doc['id']],
                    metadatas=[doc['metadata']]
                )
            print(f"{len(sample_docs)}개 문서 추가 완료")
            retriever._build_keyword_index()  # 인덱스 재구축
        else:
            print(f"기존 {existing_count}개 문서 사용")
    except Exception as e:
        print(f"문서 추가 오류: {e}")
    
    # 테스트 쿼리들
    test_queries = [
        "RAG 시스템 구현 방법",
        "벡터 검색 최적화",
        "임베딩 모델 성능 비교",
        "하이브리드 검색 전략"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n테스트 {i}: '{query}'")
        print("=" * 60)
        
        # 1. 시맨틱 검색만
        print("\n1. 시맨틱 검색")
        print("-" * 30)
        semantic_results = retriever.semantic_search(query, top_k=3)
        for j, result in enumerate(semantic_results):
            print(f"  {j+1}. 점수: {result['semantic_score']:.3f}")
            print(f"     내용: {result['content'][:80]}...")
        
        # 2. 키워드 검색만  
        print("\n2. 키워드 검색")
        print("-" * 30)
        keyword_results = retriever.keyword_search(query, top_k=3)
        for j, result in enumerate(keyword_results):
            print(f"  {j+1}. 점수: {result['keyword_score']:.3f}")
            print(f"     내용: {result['content'][:80]}...")
        
        # 3. 하이브리드 검색
        print("\n3. 하이브리드 검색 (시맨틱 70% + 키워드 30%)")
        print("-" * 30)
        hybrid_results = retriever.hybrid_search(query, top_k=3, semantic_weight=0.7)
        for result in hybrid_results:
            print(f"  {result.rank}. 종합점수: {result.combined_score:.3f}")
            print(f"     시맨틱: {result.semantic_score:.3f}, 키워드: {result.keyword_score:.3f}")
            print(f"     카테고리: {result.metadata.get('category', 'N/A')}")
            print(f"     내용: {result.content[:80]}...")
        
        print("\n" + "=" * 60)

def demonstrate_query_processing():
    """쿼리 처리 및 재작성 시연"""
    print(f"\n고급 쿼리 처리 기법")
    print("=" * 50)
    
    query_processor = AdvancedQueryProcessor()
    query_rewriter = QueryRewriter()
    
    test_query = "AI가 뭐야?"
    
    print(f"원본 쿼리: '{test_query}'")
    print()
    
    # 1. 쿼리 확장
    print("1. 쿼리 확장")
    print("-" * 20)
    
    expanded_synonym = query_processor.expand_query(test_query, "synonym")
    print(f"동의어 확장: {expanded_synonym}")
    
    expanded_keyword = query_processor.expand_query(test_query, "keyword") 
    print(f"키워드 확장: {expanded_keyword}")
    
    expanded_contextual = query_processor.expand_query(test_query, "contextual")
    print(f"문맥적 확장: {expanded_contextual}")
    
    # 2. 쿼리 재작성
    print(f"\n2. 쿼리 재작성")
    print("-" * 20)
    
    clarified = query_rewriter.rewrite_query(test_query, "clarification")
    print(f"명확화: '{clarified}'")
    
    simplified = query_rewriter.rewrite_query(test_query, "simplification")
    print(f"단순화: '{simplified}'")

def demonstrate_reranking():
    """재순위 기법 시연"""
    print(f"\n재순위 기법 시연")
    print("=" * 50)
    
    retriever = HybridRetriever("advanced-search-demo")
    reranker = ReRanker()
    
    query = "RAG 시스템 구현"
    
    # 초기 검색 결과
    initial_results = retriever.hybrid_search(query, top_k=5)
    
    print(f"쿼리: '{query}'")
    print(f"\n초기 검색 결과:")
    print("-" * 30)
    for result in initial_results:
        print(f"  {result.rank}. 점수: {result.combined_score:.3f}")
        print(f"     카테고리: {result.metadata.get('category', 'N/A')}")
        print(f"     날짜: {result.metadata.get('date', 'N/A')}")
    
    # 관련성 기반 재순위
    print(f"\n관련성 기반 재순위:")
    print("-" * 30)
    relevance_reranked = reranker.rerank_results(query, initial_results.copy(), "relevance")
    for result in relevance_reranked:
        print(f"  {result.rank}. 점수: {result.combined_score:.3f}")
        print(f"     카테고리: {result.metadata.get('category', 'N/A')}")
    
    # 다양성 기반 재순위
    print(f"\n다양성 기반 재순위:")
    print("-" * 30)
    diversity_reranked = reranker.rerank_results(query, initial_results.copy(), "diversity")
    for result in diversity_reranked:
        print(f"  {result.rank}. 점수: {result.combined_score:.3f}")
        print(f"     카테고리: {result.metadata.get('category', 'N/A')}")
    
    # 최신성 기반 재순위
    print(f"\n최신성 기반 재순위:")
    print("-" * 30)
    recency_reranked = reranker.rerank_results(query, initial_results.copy(), "recency")
    for result in recency_reranked:
        print(f"  {result.rank}. 점수: {result.combined_score:.3f}")
        print(f"     날짜: {result.metadata.get('date', 'N/A')}")

def analyze_search_performance():
    """검색 성능 분석"""
    print(f"\n검색 성능 분석")
    print("=" * 50)
    
    retriever = HybridRetriever("advanced-search-demo")
    
    test_queries = [
        "인공지능 최신 동향",
        "RAG 구현 가이드", 
        "벡터 검색 최적화 방법"
    ]
    
    methods = [
        ("시맨틱", "semantic"),
        ("키워드", "keyword"), 
        ("하이브리드", "hybrid")
    ]
    
    performance_data = {}
    
    for method_name, method_type in methods:
        print(f"\n{method_name} 검색 성능:")
        print("-" * 30)
        
        total_time = 0
        total_results = 0
        
        for query in test_queries:
            start_time = time.time()
            
            if method_type == "semantic":
                results = retriever.semantic_search(query, top_k=5)
            elif method_type == "keyword":
                results = retriever.keyword_search(query, top_k=5)
            else:  # hybrid
                results = retriever.hybrid_search(query, top_k=5)
            
            search_time = time.time() - start_time
            total_time += search_time
            total_results += len(results)
            
            print(f"  '{query}': {search_time:.3f}초, {len(results)}개 결과")
        
        avg_time = total_time / len(test_queries)
        avg_results = total_results / len(test_queries)
        
        performance_data[method_name] = {
            'avg_time': avg_time,
            'avg_results': avg_results
        }
        
        print(f"  평균 검색 시간: {avg_time:.3f}초")
        print(f"  평균 결과 수: {avg_results:.1f}개")
    
    # 성능 비교 요약
    print(f"\n성능 비교 요약:")
    print("-" * 30)
    for method, data in performance_data.items():
        print(f"{method:8s}: {data['avg_time']:.3f}초")

def main():
    """메인 실행 함수"""
    try:
        # 검색 전략 비교
        demonstrate_search_strategies()
        
        # 쿼리 처리 기법
        demonstrate_query_processing()
        
        # 재순위 기법
        demonstrate_reranking()
        
        # 성능 분석
        analyze_search_performance()
        
        print(f"\n고급 검색 기법 실습 완료!")
        print("다음 단계에서는 컨텍스트 관리 기법을 학습합니다.")
        
    except Exception as e:
        print(f"실습 중 오류 발생: {e}")
        print("시스템 설정을 확인해주세요.")

if __name__ == "__main__":
    main() 