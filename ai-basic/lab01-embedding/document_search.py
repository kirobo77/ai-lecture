"""
Lab 1 - Step 3: 문서 검색 엔진 구현
임베딩을 활용한 의미 기반 문서 검색 시스템을 구현합니다
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.utils import EmbeddingUtils, SimilarityUtils, TextUtils, print_progress, format_results
from shared.config import validate_api_keys
import json
import time
from typing import List, Dict, Any
import pickle

class DocumentSearchEngine:
    """문서 검색 엔진 클래스"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadata = []
        
    def add_document(self, text: str, metadata: Dict[str, Any] = None):
        """문서 추가"""
        if metadata is None:
            metadata = {}
            
        doc_id = len(self.documents)
        metadata['doc_id'] = doc_id
        metadata['length'] = len(text)
        metadata['tokens'] = TextUtils.count_tokens(text)
        
        self.documents.append(text)
        self.metadata.append(metadata)
        
        print(f"문서 추가: ID {doc_id}, 길이 {len(text)}자")
        
    def build_index(self, batch_size: int = 10):
        """문서 임베딩 인덱스 구축"""
        print(f"\n인덱스 구축 시작 (총 {len(self.documents)}개 문서)")
        
        start_time = time.time()
        
        # 배치 단위로 임베딩 생성
        all_embeddings = []
        for i in range(0, len(self.documents), batch_size):
            batch_docs = self.documents[i:i+batch_size]
            batch_embeddings = EmbeddingUtils.get_embeddings_batch(batch_docs)
            all_embeddings.extend(batch_embeddings)
            
            print_progress(
                min(i + batch_size, len(self.documents)), 
                len(self.documents), 
                "임베딩 생성"
            )
        
        self.embeddings = all_embeddings
        
        end_time = time.time()
        print(f"\n인덱스 구축 완료: {end_time - start_time:.2f}초")
        print(f"평균 처리 시간/문서: {(end_time - start_time) / len(self.documents):.3f}초")
        
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """쿼리 기반 문서 검색"""
        if not self.embeddings:
            raise ValueError("인덱스가 구축되지 않았습니다. build_index()를 먼저 실행해주세요.")
        
        print(f"검색 쿼리: '{query}'")
        
        # 쿼리 임베딩 생성
        query_embedding = EmbeddingUtils.get_embedding(query)
        
        # 유사도 계산
        results = SimilarityUtils.find_most_similar(
            query_embedding, self.embeddings, self.documents, top_k
        )
        
        # 메타데이터 추가
        for result in results:
            doc_index = result['index']
            result['metadata'] = self.metadata[doc_index]
            
        return results
        
    def save_index(self, filepath: str):
        """인덱스를 파일로 저장"""
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'metadata': self.metadata
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"인덱스 저장 완료: {filepath}")
        
    def load_index(self, filepath: str):
        """파일에서 인덱스 로드"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        self.documents = data['documents']
        self.embeddings = data['embeddings']
        self.metadata = data['metadata']
        
        print(f"인덱스 로드 완료: {len(self.documents)}개 문서")

def create_sample_documents():
    """샘플 문서 컬렉션 생성"""
    print("=" * 50)
    print("샘플 문서 컬렉션 생성")
    print("=" * 50)
    
    documents = [
        # AI/기술 관련
        {
            "text": "인공지능(AI)은 인간의 지능을 기계로 구현하는 기술입니다. 머신러닝, 딥러닝, 자연어처리 등의 분야가 포함됩니다.",
            "category": "기술",
            "title": "인공지능 개요",
            "author": "김AI"
        },
        {
            "text": "머신러닝은 데이터로부터 패턴을 학습하여 예측하는 AI 기술입니다. 지도학습, 비지도학습, 강화학습으로 분류됩니다.",
            "category": "기술",
            "title": "머신러닝 기초",
            "author": "박ML"
        },
        {
            "text": "딥러닝은 인공신경망을 여러 층으로 쌓아서 복잡한 패턴을 학습하는 방법입니다. CNN, RNN, Transformer 등이 대표적입니다.",
            "category": "기술",
            "title": "딥러닝 심화",
            "author": "이DL"
        },
        
        # 음식 관련
        {
            "text": "김치찌개는 한국의 대표적인 음식 중 하나입니다. 신김치, 돼지고기, 두부를 넣고 끓인 얼큰한 찌개입니다.",
            "category": "음식",
            "title": "김치찌개 레시피",
            "author": "요리왕"
        },
        {
            "text": "비빔밥은 여러 가지 나물과 고기를 밥 위에 올리고 고추장과 함께 비벼 먹는 한국 전통 음식입니다.",
            "category": "음식",
            "title": "비빔밥 만들기",
            "author": "한식셰프"
        },
        {
            "text": "파스타는 이탈리아의 대표 음식으로, 밀가루로 만든 면에 다양한 소스를 곁들여 먹습니다. 스파게티, 펜네, 라자냐 등이 있습니다.",
            "category": "음식",
            "title": "이탈리아 파스타",
            "author": "이탈리아셰프"
        },
        
        # 스포츠 관련
        {
            "text": "축구는 전 세계에서 가장 인기 있는 스포츠입니다. 11명의 선수가 팀을 이루어 공을 상대 골대에 넣는 경기입니다.",
            "category": "스포츠",
            "title": "축구 기초 규칙",
            "author": "축구해설가"
        },
        {
            "text": "야구는 9명이 한 팀을 이루어 9회까지 공격과 수비를 번갈아 하며 더 많은 점수를 내는 스포츠입니다.",
            "category": "스포츠",
            "title": "야구 게임 룰",
            "author": "야구전문가"
        },
        {
            "text": "농구는 5명이 한 팀을 이루어 상대편 바스켓에 공을 넣어 점수를 내는 실내 스포츠입니다. NBA가 가장 유명한 리그입니다.",
            "category": "스포츠",
            "title": "농구와 NBA",
            "author": "농구코치"
        },
        
        # 여행 관련
        {
            "text": "파리는 프랑스의 수도로 에펠탑, 루브르 박물관, 샹젤리제 거리 등 유명한 관광지가 많습니다.",
            "category": "여행",
            "title": "파리 여행 가이드",
            "author": "여행작가"
        },
        {
            "text": "도쿄는 일본의 수도로 전통과 현대가 조화를 이루는 도시입니다. 아사쿠사, 시부야, 하라주쿠 등이 인기 관광지입니다.",
            "category": "여행",
            "title": "도쿄 관광 명소",
            "author": "일본통"
        },
        {
            "text": "뉴욕은 미국의 대표 도시로 자유의 여신상, 타임스퀘어, 센트럴파크 등 볼거리가 가득합니다.",
            "category": "여행",
            "title": "뉴욕 여행 코스",
            "author": "미국여행가"
        }
    ]
    
    return documents

def demonstrate_basic_search():
    """기본 검색 기능 시연"""
    print("\n" + "=" * 50)
    print("기본 검색 기능 시연")
    print("=" * 50)
    
    # 검색 엔진 초기화
    search_engine = DocumentSearchEngine()
    
    # 샘플 문서 추가
    documents = create_sample_documents()
    
    print(f"\n문서 추가 중...")
    for doc in documents:
        search_engine.add_document(doc['text'], {
            'category': doc['category'],
            'title': doc['title'],
            'author': doc['author']
        })
    
    # 인덱스 구축
    search_engine.build_index()
    
    # 다양한 검색 쿼리 테스트
    test_queries = [
        "딥러닝과 신경망",
        "한국 전통 음식",
        "축구 경기 규칙",
        "유럽 여행지",
        "머신러닝 알고리즘"
    ]
    
    for query in test_queries:
        print(f"\n" + "-" * 40)
        results = search_engine.search(query, top_k=3)
        
        print(f"상위 3개 결과:")
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            print(f"  {i}. [{metadata['category']}] {metadata['title']}")
            print(f"     유사도: {result['similarity']:.3f}")
            print(f"     내용: {result['text'][:60]}...")
            print(f"     작성자: {metadata['author']}")
    
    return search_engine

def analyze_search_performance(search_engine: DocumentSearchEngine):
    """검색 성능 분석"""
    print("\n" + "=" * 50)
    print("검색 성능 분석")
    print("=" * 50)
    
    # 성능 측정용 쿼리
    performance_queries = [
        "인공지능 기술",
        "맛있는 요리",
        "인기 스포츠",
        "해외 여행"
    ] * 5  # 5번 반복으로 평균 계산
    
    search_times = []
    
    print("검색 속도 측정 중...")
    for i, query in enumerate(performance_queries):
        start_time = time.time()
        results = search_engine.search(query, top_k=5)
        end_time = time.time()
        
        search_time = end_time - start_time
        search_times.append(search_time)
        
        print_progress(i + 1, len(performance_queries), "검색 테스트")
    
    # 성능 통계
    avg_time = sum(search_times) / len(search_times)
    min_time = min(search_times)
    max_time = max(search_times)
    
    print(f"\n검색 성능 통계:")
    print(f"  평균 검색 시간: {avg_time:.3f}초")
    print(f"  최소 검색 시간: {min_time:.3f}초")
    print(f"  최대 검색 시간: {max_time:.3f}초")
    print(f"  초당 검색 가능 수: {1/avg_time:.1f}회")

def compare_keyword_vs_semantic():
    """키워드 검색 vs 의미 기반 검색 비교"""
    print("\n" + "=" * 50)
    print("키워드 vs 의미 기반 검색 비교")
    print("=" * 50)
    
    # 검색 엔진 초기화
    search_engine = DocumentSearchEngine()
    documents = create_sample_documents()
    
    for doc in documents:
        search_engine.add_document(doc['text'], {
            'category': doc['category'],
            'title': doc['title']
        })
    
    search_engine.build_index()
    
    # 비교 테스트 케이스
    test_cases = [
        {
            "query": "AI 기술",
            "keyword_matches": ["인공지능", "머신러닝"],
            "description": "AI 관련 용어로 검색"
        },
        {
            "query": "맛있는 식사",
            "keyword_matches": ["음식", "요리"],
            "description": "음식 관련 의미로 검색"
        },
        {
            "query": "공 스포츠",
            "keyword_matches": ["축구", "야구", "농구"],
            "description": "공을 사용하는 스포츠 검색"
        }
    ]
    
    for test_case in test_cases:
        query = test_case["query"]
        keyword_matches = test_case["keyword_matches"]
        
        print(f"\n테스트: {test_case['description']}")
        print(f"   쿼리: '{query}'")
        
        # 의미 기반 검색
        semantic_results = search_engine.search(query, top_k=3)
        
        # 키워드 기반 검색 시뮬레이션
        keyword_results = []
        for i, doc in enumerate(search_engine.documents):
            score = 0
            for keyword in keyword_matches:
                if keyword in doc:
                    score += 1
            
            if score > 0:
                keyword_results.append({
                    'index': i,
                    'text': doc,
                    'score': score,
                    'metadata': search_engine.metadata[i]
                })
        
        keyword_results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n   의미 기반 검색 결과:")
        for i, result in enumerate(semantic_results[:3], 1):
            metadata = result['metadata']
            print(f"     {i}. [{metadata['category']}] {metadata['title']} (유사도: {result['similarity']:.3f})")
        
        print(f"\n   키워드 기반 검색 결과:")
        for i, result in enumerate(keyword_results[:3], 1):
            metadata = result['metadata']
            print(f"     {i}. [{metadata['category']}] {metadata['title']} (매칭: {result['score']}개)")

def interactive_search():
    """대화형 검색 인터페이스"""
    print("\n" + "=" * 50)
    print("대화형 검색 인터페이스")
    print("=" * 50)
    
    # 검색 엔진 준비
    search_engine = DocumentSearchEngine()
    documents = create_sample_documents()
    
    for doc in documents:
        search_engine.add_document(doc['text'], {
            'category': doc['category'],
            'title': doc['title'],
            'author': doc['author']
        })
    
    search_engine.build_index()
    
    print("자유롭게 검색해보세요! (종료: 'quit')")
    print("사용 가능한 카테고리: 기술, 음식, 스포츠, 여행")
    
    while True:
        print("\n" + "-" * 30)
        query = input("검색어를 입력하세요: ").strip()
        
        if query.lower() == 'quit':
            break
            
        if not query:
            continue
        
        try:
            results = search_engine.search(query, top_k=5)
            
            print(f"\n검색 결과 ({len(results)}개):")
            for i, result in enumerate(results, 1):
                metadata = result['metadata']
                print(f"\n  {i}. {metadata['title']}")
                print(f"     카테고리: {metadata['category']}")
                print(f"     작성자: {metadata['author']}")
                print(f"     유사도: {result['similarity']:.3f}")
                print(f"     내용: {result['text'][:100]}...")
                
        except Exception as e:
            print(f"검색 중 오류: {e}")

def main():
    """메인 실행 함수"""
    print("Lab 1 - Step 3: 문서 검색 엔진 구현")
    print("의미 기반 문서 검색 시스템 구축\n")
    
    # API 키 확인
    if not validate_api_keys():
        print("API 키 설정이 필요합니다.")
        return
    
    try:
        # 1. 기본 검색 기능 시연
        search_engine = demonstrate_basic_search()
        
        # 2. 검색 성능 분석
        analyze_search_performance(search_engine)
        
        # 3. 키워드 vs 의미 기반 검색 비교
        compare_keyword_vs_semantic()
        
        # 4. 대화형 검색 (선택사항)
        print("\n대화형 검색을 체험하시겠습니까? (y/n): ", end="")
        if input().lower().startswith('y'):
            interactive_search()
        
        print("\n" + "=" * 50)
        print("문서 검색 엔진 구현 완료!")
        print("=" * 50)
        print("\n학습한 내용:")
        print("• 문서 컬렉션 구축 및 인덱싱")
        print("• 의미 기반 유사도 검색")
        print("• 검색 결과 랭킹 및 메타데이터 활용")
        print("• 키워드 검색 대비 의미 검색의 장점")
        
        print("\n다음 단계:")
        print("embedding_visualization.py를 실행하여 임베딩을 시각화해보세요!")
        
    except Exception as e:
        print(f"실습 중 오류 발생: {e}")

if __name__ == "__main__":
    main() 