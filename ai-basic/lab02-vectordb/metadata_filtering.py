"""
Lab 2 - Step 3: 메타데이터 활용
복합 메타데이터를 활용한 필터링 검색 및 하이브리드 검색
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from shared.config import validate_api_keys, CHROMA_PERSIST_DIRECTORY
from shared.utils import EmbeddingUtils
import time
import random
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

class AdvancedSearchEngine:
    """고급 메타데이터 검색 엔진"""
    
    def __init__(self, collection_name="advanced_search"):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        self.collection = None
        
    def initialize(self, reset=False):
        """검색 엔진 초기화"""
        print("고급 검색 엔진 초기화")
        print("=" * 40)
        
        if reset:
            try:
                self.client.delete_collection(self.collection_name)
                print(f"기존 컬렉션 '{self.collection_name}' 삭제됨")
            except Exception as e:
                print(f"컬렉션 삭제 실패: {e}")
        
        try:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "고급 메타데이터 검색용 컬렉션"}
            )
            print(f"새 컬렉션 '{self.collection_name}' 생성됨")
        except Exception:
            self.collection = self.client.get_collection(self.collection_name)
            print(f"기존 컬렉션 '{self.collection_name}' 로드됨")
        
        print(f"컬렉션 문서 수: {self.collection.count()}")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """문서들을 컬렉션에 추가"""
        print(f"\n문서 {len(documents)}개 추가")
        
        docs = [doc["content"] for doc in documents]
        metadatas = [self._convert_metadata_for_chroma(doc["metadata"]) for doc in documents]
        ids = [doc["id"] for doc in documents]
        
        self.collection.add(
            documents=docs,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"추가 완료. 총 문서 수: {self.collection.count()}")
    
    def _convert_metadata_for_chroma(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """ChromaDB 호환을 위해 메타데이터 변환"""
        converted = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                # 리스트를 쉼표로 구분된 문자열로 변환
                converted[key] = ", ".join(str(item) for item in value)
            else:
                converted[key] = value
        return converted
    
    def search_by_category(self, query: str, categories: List[str], top_k: int = 5) -> List[Dict]:
        """카테고리별 검색"""
        print(f"\n카테고리별 검색")
        print(f"쿼리: '{query}'")
        print(f"카테고리: {categories}")
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"category": {"$in": categories}}
        )
        
        return self._format_results(results)
    
    def search_by_author(self, query: str, authors: List[str], top_k: int = 5) -> List[Dict]:
        """작성자별 검색"""
        print(f"\n작성자별 검색")
        print(f"쿼리: '{query}'")
        print(f"작성자: {authors}")
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"author": {"$in": authors}}
        )
        
        return self._format_results(results)
    
    def search_by_date_range(self, query: str, start_date: str, end_date: str, top_k: int = 5) -> List[Dict]:
        """날짜 범위별 검색"""
        print(f"\n날짜 범위별 검색")
        print(f"쿼리: '{query}'")
        print(f"날짜 범위: {start_date} ~ {end_date}")
        
        # ChromaDB는 날짜 문자열 비교를 지원하지 않으므로 모든 문서를 가져온 후 필터링
        all_results = self.collection.query(
            query_texts=[query],
            n_results=100  # 충분히 큰 수로 설정
        )
        
        # 날짜 범위로 필터링
        filtered_results = []
        for i, metadata in enumerate(all_results['metadatas'][0]):
            doc_date = metadata.get('date', '')
            if start_date <= doc_date <= end_date:
                filtered_results.append({
                    'id': all_results['ids'][0][i],
                    'document': all_results['documents'][0][i],
                    'metadata': metadata,
                    'distance': all_results['distances'][0][i] if 'distances' in all_results else 0.0
                })
        
        # 상위 k개 반환
        filtered_results.sort(key=lambda x: x['distance'])
        return filtered_results[:top_k]
    
    def complex_search(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Dict]:
        """복합 조건 검색"""
        print(f"\n복합 조건 검색")
        print(f"쿼리: '{query}'")
        print(f"필터: {filters}")
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filters
        )
        
        return self._format_results(results)
    
    def _normalize_filters_for_chroma(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        # 이미 $and/$or/$not로 시작하면 그대로 반환
        if any(k in filters for k in ["$and", "$or", "$not"]):
            return filters
        # 복합 조건이면 $and로 감싸기
        if len(filters) > 1:
            return {"$and": [{k: v} for k, v in filters.items()]}
        return filters

    def hybrid_search(self, query: str, filters: Dict[str, Any], 
                     semantic_weight: float = 0.7, top_k: int = 5) -> List[Dict]:
        """하이브리드 검색 (의미적 유사도 + 메타데이터 필터링)"""
        print(f"\n하이브리드 검색")
        print(f"쿼리: '{query}'")
        print(f"의미적 가중치: {semantic_weight}")
        print(f"필터: {filters}")
        
        # 1. 필터 조건에 맞는 모든 문서 조회
        where_condition = self._normalize_filters_for_chroma(filters)
        filtered_docs = self.collection.get(
            where=where_condition,
            include=["documents", "metadatas"]
        )
        
        if not filtered_docs['documents']:
            print("필터 조건에 맞는 문서가 없습니다.")
            return []
        
        print(f"필터링된 문서 수: {len(filtered_docs['documents'])}")
        
        # 2. 필터링된 문서 중에서 의미적 검색
        results = self.collection.query(
            query_texts=[query],
            n_results=5,  # top_k
            where=where_condition
        )
        
        # 결과 조립 (ids 대신 enumerate 인덱스 사용)
        formatted_results = []
        for i, doc in enumerate(filtered_docs['documents']):
            meta = filtered_docs['metadatas'][i]
            formatted_results.append({
                'document': doc,
                'metadata': meta,
                'distance': results['distances'][0][i] if 'distances' in results and len(results['distances'][0]) > i else 0.0
            })
        
        return formatted_results
    
    def faceted_search(self, query: str, top_k: int = 10) -> Dict[str, List[Dict]]:
        """패싯 검색 (카테고리별로 결과 그룹화)"""
        print(f"\n패싯 검색")
        print(f"쿼리: '{query}'")
        
        # 전체 검색
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]  # 'ids' 제거
        )
        
        # 결과 포맷팅 (enumerate 인덱스 사용)
        formatted_results = []
        for i, (document, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            formatted_results.append({
                'id': i + 1,
                'content': document,
                'metadata': metadata,
                'distance': distance,
                'rank': i + 1
            })
        
        # 카테고리별로 그룹화
        faceted_results = {}
        for result in formatted_results:
            category = result['metadata'].get('category', 'unknown')
            if category not in faceted_results:
                faceted_results[category] = []
            faceted_results[category].append(result)
        
        return faceted_results
    
    def _format_results(self, results) -> List[Dict]:
        """검색 결과 포맷팅"""
        formatted = []
        
        if not results['ids'] or not results['ids'][0]:
            return formatted
        
        for i, (doc_id, document, metadata, distance) in enumerate(zip(
            results['ids'][0],
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            formatted.append({
                'id': doc_id,
                'content': document,
                'metadata': metadata,
                'distance': distance,
                'rank': i + 1
            })
        
        return formatted
    
    def _calculate_metadata_score(self, metadata: Dict, query: str) -> float:
        """메타데이터 기반 스코어 계산 (간단한 예시)"""
        score = 0.0
        
        # 제목에 쿼리 키워드가 포함되어 있으면 가점
        title = metadata.get('title', '').lower()
        if any(word.lower() in title for word in query.split()):
            score += 0.3
        
        # 최신 문서일수록 가점
        try:
            doc_date = datetime.strptime(metadata.get('date', '2020-01-01'), '%Y-%m-%d')
            days_ago = (datetime.now() - doc_date).days
            recency_score = max(0, 1 - days_ago / 365)  # 1년 기준
            score += 0.2 * recency_score
        except:
            pass
        
        # 문서 길이 정규화 (너무 짧지도 길지도 않은 문서 선호)
        word_count = metadata.get('word_count', 0)
        if 50 <= word_count <= 500:
            score += 0.1
        
        return min(score, 1.0)  # 최대 1.0으로 제한

def create_rich_sample_data() -> List[Dict[str, Any]]:
    """풍부한 메타데이터를 가진 샘플 데이터 생성"""
    print("풍부한 메타데이터 샘플 데이터 생성")
    print("=" * 40)
    
    categories = ["AI/ML", "클라우드", "데이터사이언스", "웹개발", "모바일", "보안"]
    authors = ["김AI", "박클라우드", "이데이터", "최웹", "정모바일", "한보안"]
    companies = ["테크코", "데이터랩", "AI솔루션", "클라우드웍스", "스마트시스템"]
    tags_pool = ["python", "tensorflow", "aws", "docker", "react", "vue", "android", "ios", "security", "blockchain"]
    
    documents = []
    
    # 기술 문서들
    tech_contents = [
        "머신러닝 모델 배포를 위한 Docker 컨테이너 최적화 전략에 대해 알아보겠습니다. GPU 자원 활용과 메모리 관리가 핵심입니다.",
        "AWS Lambda를 활용한 서버리스 아키텍처 구현 방법을 설명합니다. 비용 효율성과 확장성을 동시에 달성할 수 있습니다.",
        "React와 TypeScript를 결합한 현대적인 웹 애플리케이션 개발 패턴을 소개합니다. 타입 안정성과 개발 생산성이 향상됩니다.",
        "TensorFlow를 사용한 컴퓨터 비전 모델 개발 과정을 단계별로 설명합니다. 데이터 전처리부터 모델 평가까지 포함합니다.",
        "Kubernetes 클러스터에서의 CI/CD 파이프라인 구축 방법을 다룹니다. GitOps 패턴을 적용한 자동화된 배포 시스템입니다.",
        "Vue.js 3의 Composition API를 활용한 상태 관리 패턴을 설명합니다. 대규모 애플리케이션에서의 코드 재사용성이 핵심입니다.",
        "PyTorch를 이용한 자연어 처리 모델 개발 가이드입니다. BERT 기반 트랜스포머 모델의 파인튜닝 과정을 포함합니다.",
        "마이크로서비스 아키텍처에서의 API 게이트웨이 설계 원칙을 다룹니다. 인증, 로드밸런싱, 모니터링이 주요 고려사항입니다.",
        "Android Jetpack Compose를 활용한 선언적 UI 개발 방법론을 소개합니다. 기존 XML 방식 대비 생산성이 크게 향상됩니다.",
        "사이버 보안 관점에서의 DevSecOps 구현 전략을 설명합니다. 보안을 개발 프로세스에 통합하는 것이 핵심입니다."
    ]
    
    for i, content in enumerate(tech_contents):
        # 날짜 생성 (최근 2년 내)
        days_ago = random.randint(0, 730)
        doc_date = datetime.now() - timedelta(days=days_ago)
        
        # 메타데이터 생성
        category = random.choice(categories)
        author = random.choice(authors)
        company = random.choice(companies)
        
        # 카테고리에 따른 태그 선택
        category_tags = {
            "AI/ML": ["python", "tensorflow", "pytorch"],
            "클라우드": ["aws", "docker", "kubernetes"],
            "데이터사이언스": ["python", "pandas", "numpy"],
            "웹개발": ["react", "vue", "javascript"],
            "모바일": ["android", "ios", "flutter"],
            "보안": ["security", "encryption", "authentication"]
        }
        
        available_tags = category_tags.get(category, tags_pool)
        sample_size = min(random.randint(2, 4), len(available_tags))
        tags = random.sample(available_tags, sample_size)
        
        # 조회수, 좋아요 등 추가 메트릭
        views = random.randint(100, 10000)
        likes = random.randint(10, views // 10)
        
        doc = {
            "id": f"tech_doc_{i+1:03d}",
            "content": content,
            "metadata": {
                "title": f"{category} 기술 가이드 - {i+1:03d}",
                "category": category,
                "author": author,
                "company": company,
                "date": doc_date.strftime("%Y-%m-%d"),
                "tags": tags,
                "difficulty": random.choice(["초급", "중급", "고급"]),
                "estimated_read_time": random.randint(5, 30),
                "views": views,
                "likes": likes,
                "word_count": len(content.split()),
                "language": "Korean",
                "type": "technical_guide",
                "status": random.choice(["published", "draft", "review"]),
                "last_updated": (doc_date + timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d")
            }
        }
        
        documents.append(doc)
    
    # 통계 출력
    stats = {
        "categories": {},
        "authors": {},
        "companies": {},
        "difficulties": {}
    }
    
    for doc in documents:
        meta = doc["metadata"]
        stats["categories"][meta["category"]] = stats["categories"].get(meta["category"], 0) + 1
        stats["authors"][meta["author"]] = stats["authors"].get(meta["author"], 0) + 1
        stats["companies"][meta["company"]] = stats["companies"].get(meta["company"], 0) + 1
        stats["difficulties"][meta["difficulty"]] = stats["difficulties"].get(meta["difficulty"], 0) + 1
    
    print(f"총 문서 수: {len(documents)}")
    print(f"카테고리 분포: {stats['categories']}")
    print(f"난이도 분포: {stats['difficulties']}")
    
    return documents

def demonstrate_basic_filtering():
    """기본 필터링 기능 시연"""
    print("\n기본 필터링 기능 시연")
    print("=" * 50)
    
    # 검색 엔진 초기화
    search_engine = AdvancedSearchEngine("filtering_demo")
    search_engine.initialize(reset=True)
    
    # 샘플 데이터 추가
    documents = create_rich_sample_data()
    search_engine.add_documents(documents)
    
    # 1. 카테고리별 검색
    print("\n1. 카테고리별 검색")
    print("-" * 30)
    
    results = search_engine.search_by_category(
        query="딥러닝 모델 개발",
        categories=["AI/ML", "데이터사이언스"],
        top_k=3
    )
    
    print_search_results(results, "카테고리 필터링")
    
    # 2. 작성자별 검색
    print("\n2. 작성자별 검색")
    print("-" * 30)
    
    results = search_engine.search_by_author(
        query="클라우드 배포",
        authors=["박클라우드", "김AI"],
        top_k=3
    )
    
    print_search_results(results, "작성자 필터링")
    
    # 3. 날짜 범위별 검색
    print("\n3. 날짜 범위별 검색")
    print("-" * 30)
    
    # 최근 6개월 문서만 검색
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    
    results = search_engine.search_by_date_range(
        query="웹 애플리케이션 개발",
        start_date=start_date,
        end_date=end_date,
        top_k=3
    )
    
    print_search_results(results, "날짜 범위 필터링")

def demonstrate_complex_filtering():
    """복합 조건 필터링 시연"""
    print("\n복합 조건 필터링 시연")
    print("=" * 50)
    
    search_engine = AdvancedSearchEngine("filtering_demo")
    search_engine.initialize(reset=False)  # 기존 데이터 사용
    
    # 1. AND 조건
    print("\n1. AND 조건 검색")
    print("-" * 30)
    print("조건: AI/ML 카테고리 AND 고급 난이도")
    
    and_filter = {
        "$and": [
            {"category": "AI/ML"},
            {"difficulty": "고급"}
        ]
    }
    
    results = search_engine.complex_search(
        query="머신러닝 최적화",
        filters=and_filter,
        top_k=3
    )
    
    print_search_results(results, "AND 조건")
    
    # 2. OR 조건
    print("\n2. OR 조건 검색")
    print("-" * 30)
    print("조건: 웹개발 OR 모바일 카테고리")
    
    or_filter = {
        "category": {"$in": ["웹개발", "모바일"]}
    }
    
    results = search_engine.complex_search(
        query="사용자 인터페이스",
        filters=or_filter,
        top_k=3
    )
    
    print_search_results(results, "OR 조건")
    
    # 3. NOT 조건
    print("\n3. NOT 조건 검색")
    print("-" * 30)
    print("조건: 초급이 아닌 문서")
    
    not_filter = {
        "difficulty": {"$ne": "초급"}
    }
    
    results = search_engine.complex_search(
        query="고급 기술",
        filters=not_filter,
        top_k=3
    )
    
    print_search_results(results, "NOT 조건")
    
    # 4. 복합 조건
    print("\n4. 복합 조건 검색")
    print("-" * 30)
    print("조건: (AI/ML OR 클라우드) AND 중급 이상 AND 조회수 1000 이상")
    
    complex_filter = {
        "$and": [
            {"category": {"$in": ["AI/ML", "클라우드"]}},
            {"difficulty": {"$in": ["중급", "고급"]}},
            {"views": {"$gte": 1000}}
        ]
    }
    
    results = search_engine.complex_search(
        query="아키텍처 설계",
        filters=complex_filter,
        top_k=3
    )
    
    print_search_results(results, "복합 조건")

def demonstrate_hybrid_search():
    """하이브리드 검색 시연"""
    print("\n하이브리드 검색 시연")
    print("=" * 50)
    
    search_engine = AdvancedSearchEngine("filtering_demo")
    search_engine.initialize(reset=False)
    
    # 1. 기본 하이브리드 검색
    print("\n1. 기본 하이브리드 검색")
    print("-" * 30)
    
    hybrid_filter = {
        "category": {"$in": ["AI/ML", "데이터사이언스"]},
        "difficulty": {"$ne": "초급"}
    }
    
    results = search_engine.hybrid_search(
        query="딥러닝 모델 배포",
        filters=hybrid_filter,
        semantic_weight=0.7,
        top_k=5
    )
    
    print_hybrid_results(results, "하이브리드 검색")
    
    # 2. 의미적 가중치 비교
    print("\n2. 의미적 가중치 비교")
    print("-" * 30)
    
    weights = [0.3, 0.5, 0.7, 0.9]
    query = "클라우드 보안"
    filter_condition = {"category": {"$in": ["클라우드", "보안"]}}
    
    for weight in weights:
        print(f"\n의미적 가중치: {weight}")
        results = search_engine.hybrid_search(
            query=query,
            filters=filter_condition,
            semantic_weight=weight,
            top_k=3
        )
        
        if results:
            top_result = results[0]
            print(f"  최상위 결과: {top_result['metadata']['title']}")
            print(f"  하이브리드 스코어: {top_result.get('hybrid_score', 0.0):.3f}")

def demonstrate_faceted_search():
    """패싯 검색 시연"""
    print("\n패싯 검색 시연")
    print("=" * 50)
    
    search_engine = AdvancedSearchEngine("filtering_demo")
    search_engine.initialize(reset=False)
    
    query = "개발 방법론"
    
    faceted_results = search_engine.faceted_search(query, top_k=12)
    
    print(f"쿼리: '{query}'")
    print(f"카테고리별 검색 결과:")
    
    for category, results in faceted_results.items():
        print(f"\n[{category}] ({len(results)}개)")
        for i, result in enumerate(results[:3]):  # 각 카테고리별로 최대 3개만 표시
            print(f"  {i+1}. {result['metadata']['title']}")
            print(f"     거리: {result['distance']:.3f}")

def print_search_results(results: List[Dict], title: str):
    """검색 결과 출력"""
    print(f"\n{title} 결과 ({len(results)}개):")
    
    for i, result in enumerate(results, 1):
        meta = result['metadata']
        rank = result.get('rank', i)
        content = result.get('content', result.get('document', ''))
        print(f"\n  {rank}. {meta['title']}")
        print(f"     카테고리: {meta['category']} | 작성자: {meta['author']}")
        print(f"     날짜: {meta['date']} | 난이도: {meta['difficulty']}")
        print(f"     거리: {result['distance']:.3f}")
        print(f"     내용: {content[:80]}...")

def print_hybrid_results(results: List[Dict], title: str):
    """하이브리드 검색 결과 출력"""
    print(f"\n{title} 결과 ({len(results)}개):")
    
    for i, result in enumerate(results, 1):
        meta = result['metadata']
        rank = result.get('rank', i)
        content = result.get('content', result.get('document', ''))
        print(f"\n  {rank}. {meta['title']}")
        print(f"     카테고리: {meta['category']} | 난이도: {meta['difficulty']}")
        print(f"     의미적 거리: {result['distance']:.3f}")
        if 'hybrid_score' in result:
            print(f"     하이브리드 스코어: {result['hybrid_score']:.3f}")
        print(f"     내용: {content[:80]}...")

def analyze_filtering_performance():
    """필터링 성능 분석"""
    print("\n필터링 성능 분석")
    print("=" * 50)
    
    search_engine = AdvancedSearchEngine("filtering_demo")
    search_engine.initialize(reset=False)
    
    test_cases = [
        {
            "name": "전체 검색",
            "query": "개발 가이드",
            "filter": None
        },
        {
            "name": "단일 조건",
            "query": "개발 가이드", 
            "filter": {"category": "AI/ML"}
        },
        {
            "name": "복합 조건",
            "query": "개발 가이드",
            "filter": {
                "$and": [
                    {"category": {"$in": ["AI/ML", "웹개발"]}},
                    {"difficulty": {"$ne": "초급"}}
                ]
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        
        start_time = time.time()
        
        if test_case['filter']:
            results = search_engine.complex_search(
                query=test_case['query'],
                filters=test_case['filter'],
                top_k=10
            )
        else:
            results = search_engine.collection.query(
                query_texts=[test_case['query']],
                n_results=10
            )
            results = search_engine._format_results(results)
        
        end_time = time.time()
        
        print(f"  검색 시간: {(end_time - start_time)*1000:.2f}ms")
        print(f"  결과 수: {len(results)}")

def main():
    """메인 실행 함수"""
    print("Lab 2 - Step 3: 메타데이터 활용")
    print("복합 메타데이터를 활용한 필터링 검색\n")
    
    # API 키 확인
    if not validate_api_keys():
        print("API 키 설정이 필요합니다.")
        return
    
    try:
        # 1. 기본 필터링 기능
        demonstrate_basic_filtering()
        
        # 2. 복합 조건 필터링
        demonstrate_complex_filtering()
        
        # 3. 하이브리드 검색
        demonstrate_hybrid_search()
        
        # 4. 패싯 검색
        demonstrate_faceted_search()
        
        # 5. 성능 분석
        analyze_filtering_performance()
        
        print("\n" + "=" * 50)
        print("메타데이터 활용 학습 완료!")
        print("=" * 50)
        print("\n학습한 내용:")
        print("• 카테고리, 작성자, 날짜별 필터링 검색")
        print("• AND, OR, NOT 등 복합 조건 검색")
        print("• 하이브리드 검색 (의미적 + 메타데이터)")
        print("• 패싯 검색을 통한 결과 그룹화")
        print("• 메타데이터 기반 스코어링 전략")
        
        print("\n다음 단계:")
        print("performance_comparison.py를 실행하여 성능 최적화를 학습해보세요!")
        
    except Exception as e:
        print(f"실습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 