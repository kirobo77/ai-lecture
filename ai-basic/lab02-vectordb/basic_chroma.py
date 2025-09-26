"""
Lab 2 - Step 1: ChromaDB 기초
ChromaDB를 활용한 벡터 데이터베이스 기본 사용법 학습
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from shared.config import validate_api_keys, CHROMA_PERSIST_DIRECTORY, CHROMA_COLLECTION_NAME, OPENAI_API_KEY
from shared.utils import EmbeddingUtils
import time
from typing import List, Dict, Any

def setup_chroma_client(use_persistence=True):
    """ChromaDB 클라이언트 설정"""
    print("ChromaDB 클라이언트 설정")
    print("=" * 40)
    
    if use_persistence:
        # 영속성 모드: 데이터를 디스크에 저장
        print(f"영속성 모드: {CHROMA_PERSIST_DIRECTORY}")
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
    else:
        # 메모리 모드: 프로그램 종료 시 데이터 삭제
        print("메모리 모드: 임시 저장")
        client = chromadb.Client()
    
    print(f"ChromaDB 버전: {chromadb.__version__}")
    print(f"클라이언트 타입: {type(client).__name__}")
    
    return client

def create_collection(client, collection_name=None, reset=False):
    """컬렉션 생성 및 관리"""
    print(f"\n컬렉션 관리")
    print("=" * 40)
    
    if collection_name is None:
        collection_name = CHROMA_COLLECTION_NAME
    
    # 기존 컬렉션 목록 확인
    existing_collections = client.list_collections()
    print(f"기존 컬렉션: {[col.name for col in existing_collections]}")
    
    # 컬렉션 초기화 (필요시)
    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"기존 컬렉션 '{collection_name}' 삭제됨")
        except Exception as e:
            print(f"컬렉션 삭제 실패 (존재하지 않을 수 있음): {e}")
    
    # OpenAI 임베딩 함수 설정
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-ada-002",
        ssl_verify=False  # SSL 검증 비활성화
    )
    
    # 컬렉션 생성 또는 가져오기
    try:
        collection = client.create_collection(
            name=collection_name,
            embedding_function=openai_ef,
            metadata={"description": "AI 기초 실습용 벡터 데이터베이스"}
        )
        print(f"새 컬렉션 '{collection_name}' 생성됨 (OpenAI 임베딩 사용)")
    except Exception as e:
        # 이미 존재하는 경우 가져오기
        collection = client.get_collection(collection_name, embedding_function=openai_ef)
        print(f"기존 컬렉션 '{collection_name}' 로드됨 (OpenAI 임베딩 사용)")
    
    print(f"컬렉션 정보:")
    print(f"  이름: {collection.name}")
    print(f"  문서 수: {collection.count()}")
    
    return collection

def demonstrate_basic_operations():
    """기본 CRUD 작업 시연"""
    print(f"\n기본 CRUD 작업 시연")
    print("=" * 40)
    
    # 클라이언트 및 컬렉션 설정
    client = setup_chroma_client(use_persistence=True)
    collection = create_collection(client, reset=True)
    
    # 1. 문서 추가 (Create)
    print(f"\n1. 문서 추가 (Create)")
    print("-" * 20)
    
    sample_documents = [
        {
            "id": "doc1",
            "text": "인공지능은 인간의 지능을 모방하여 기계가 학습하고 추론할 수 있게 하는 기술입니다.",
            "metadata": {"category": "AI", "author": "김AI", "date": "2024-01-01"}
        },
        {
            "id": "doc2", 
            "text": "머신러닝은 데이터를 통해 컴퓨터가 자동으로 학습하는 인공지능의 한 분야입니다.",
            "metadata": {"category": "ML", "author": "박머신", "date": "2024-01-02"}
        },
        {
            "id": "doc3",
            "text": "딥러닝은 신경망을 여러 층으로 쌓아 복잡한 패턴을 학습하는 머신러닝 기법입니다.",
            "metadata": {"category": "DL", "author": "이딥", "date": "2024-01-03"}
        }
    ]
    
    # ChromaDB에 문서 추가
    documents = [doc["text"] for doc in sample_documents]
    metadatas = [doc["metadata"] for doc in sample_documents]
    ids = [doc["id"] for doc in sample_documents]
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"문서 {len(documents)}개 추가 완료")
    print(f"컬렉션 문서 수: {collection.count()}")
    
    # 2. 문서 검색 (Read)
    print(f"\n2. 문서 검색 (Read)")
    print("-" * 20)
    
    query = "딥러닝과 신경망의 관계"
    print(f"검색 쿼리: '{query}'")
    
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    
    print(f"검색 결과 ({len(results['ids'][0])}개):")
    for i, (doc_id, document, metadata, distance) in enumerate(zip(
        results['ids'][0],
        results['documents'][0], 
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"  {i+1}. ID: {doc_id}")
        print(f"     거리: {distance:.4f}")
        print(f"     카테고리: {metadata['category']}")
        print(f"     내용: {document[:50]}...")
        print()
    
    # 3. 문서 수정 (Update)
    print(f"\n3. 문서 수정 (Update)")
    print("-" * 20)
    
    # 기존 문서 업데이트
    updated_metadata = {"category": "AI-Advanced", "author": "김AI", "date": "2024-01-01", "updated": True}
    
    collection.update(
        ids=["doc1"],
        metadatas=[updated_metadata]
    )
    
    print("doc1의 메타데이터 업데이트 완료")
    
    # 업데이트 확인
    updated_result = collection.get(ids=["doc1"], include=["metadatas"])
    print(f"업데이트된 메타데이터: {updated_result['metadatas'][0]}")
    
    # 4. 문서 삭제 (Delete)
    print(f"\n4. 문서 삭제 (Delete)")
    print("-" * 20)
    
    print(f"삭제 전 문서 수: {collection.count()}")
    
    collection.delete(ids=["doc3"])
    print("doc3 삭제 완료")
    
    print(f"삭제 후 문서 수: {collection.count()}")
    
    return collection

def compare_embedding_methods(collection):
    """임베딩 자동 생성 vs 수동 제공 비교"""
    print(f"\n임베딩 방법 비교")
    print("=" * 40)
    
    test_text = "자연어 처리는 컴퓨터가 인간의 언어를 이해하고 생성하는 AI 기술입니다."
    
    # 1. ChromaDB 자동 임베딩 (기본값)
    print("1. ChromaDB 자동 임베딩")
    print("-" * 20)
    
    start_time = time.time()
    
    collection.add(
        documents=[test_text],
        ids=["auto_embed"],
        metadatas=[{"method": "auto", "category": "NLP"}]
    )
    
    auto_time = time.time() - start_time
    print(f"자동 임베딩 시간: {auto_time:.4f}초")
    
    # 2. 수동 임베딩 제공
    print(f"\n2. 수동 임베딩 제공")
    print("-" * 20)
    
    start_time = time.time()
    
    # OpenAI API로 임베딩 생성
    manual_embedding = EmbeddingUtils.get_embedding(test_text)
    
    collection.add(
        documents=[test_text],
        embeddings=[manual_embedding],
        ids=["manual_embed"],
        metadatas=[{"method": "manual", "category": "NLP"}]
    )
    
    manual_time = time.time() - start_time
    print(f"수동 임베딩 시간: {manual_time:.4f}초")
    print(f"임베딩 차원: {len(manual_embedding)}")
    
    # 3. 성능 비교
    print(f"\n3. 성능 비교")
    print("-" * 20)
    
    query = "언어 모델과 텍스트 분석"
    
    # 전체 검색
    results = collection.query(
        query_texts=[query],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )
    
    print(f"검색 쿼리: '{query}'")
    print("검색 결과:")
    
    for i, (doc_id, document, metadata, distance) in enumerate(zip(
        results['ids'][0],
        results['documents'][0],
        results['metadatas'][0], 
        results['distances'][0]
    )):
        if doc_id in ["auto_embed", "manual_embed"]:
            print(f"  {metadata['method']} 방식: 거리 {distance:.4f}")
    
    print(f"\n시간 비교:")
    print(f"  자동 임베딩: {auto_time:.4f}초")
    print(f"  수동 임베딩: {manual_time:.4f}초")
    print(f"  차이: {abs(auto_time - manual_time):.4f}초")

def explore_collection_features(collection):
    """컬렉션 고급 기능 탐색"""
    print(f"\n컬렉션 고급 기능")
    print("=" * 40)
    
    # 1. 메타데이터 필터링
    print("1. 메타데이터 필터링")
    print("-" * 20)
    
    # AI 카테고리 문서만 검색
    ai_results = collection.query(
        query_texts=["인공지능 기술"],
        n_results=10,
        where={"category": {"$in": ["AI", "AI-Advanced"]}}
    )
    
    print(f"AI 카테고리 문서 검색 결과: {len(ai_results['ids'][0])}개")
    for doc_id, metadata in zip(ai_results['ids'][0], ai_results['metadatas'][0]):
        print(f"  {doc_id}: {metadata['category']}")
    
    # 2. 다중 조건 필터링
    print(f"\n2. 다중 조건 필터링")
    print("-" * 20)
    
    # 특정 작성자의 NLP 카테고리 문서
    complex_results = collection.query(
        query_texts=["자연어 처리"],
        n_results=10,
        where={
            "$and": [
                {"category": "NLP"},
                {"method": {"$ne": "auto"}}  # auto가 아닌 것
            ]
        }
    )
    
    print(f"복합 조건 검색 결과: {len(complex_results['ids'][0])}개")
    
    # 3. 전체 문서 조회
    print(f"\n3. 전체 문서 조회")
    print("-" * 20)
    
    all_docs = collection.get(include=["documents", "metadatas"])
    
    print(f"전체 문서 수: {len(all_docs['ids'])}")
    print("문서 목록:")
    for doc_id, metadata in zip(all_docs['ids'], all_docs['metadatas']):
        print(f"  {doc_id}: 카테고리={metadata.get('category', 'N/A')}")

def demonstrate_persistence():
    """영속성 기능 시연"""
    print(f"\n영속성 기능 시연")
    print("=" * 40)
    
    print("1. 메모리 모드 테스트")
    print("-" * 20)
    
    # 메모리 클라이언트 생성
    memory_client = setup_chroma_client(use_persistence=False)
    memory_collection = memory_client.create_collection("temp_collection")
    
    memory_collection.add(
        documents=["임시 문서입니다."],
        ids=["temp_doc"],
        metadatas=[{"type": "temporary"}]
    )
    
    print(f"메모리 컬렉션 문서 수: {memory_collection.count()}")
    print("(프로그램 종료 시 삭제됨)")
    
    print(f"\n2. 영속성 모드 테스트")
    print("-" * 20)
    
    # 영속성 클라이언트 생성
    persistent_client = setup_chroma_client(use_persistence=True)
    
    try:
        persistent_collection = persistent_client.get_collection(CHROMA_COLLECTION_NAME)
        print(f"영속성 컬렉션 문서 수: {persistent_collection.count()}")
        print("(디스크에 저장되어 재시작 후에도 유지됨)")
    except Exception as e:
        print(f"영속성 컬렉션을 찾을 수 없음: {e}")
    
    print(f"\n저장 위치: {CHROMA_PERSIST_DIRECTORY}")
    
    # 저장된 파일 확인
    if os.path.exists(CHROMA_PERSIST_DIRECTORY):
        files = os.listdir(CHROMA_PERSIST_DIRECTORY)
        print(f"저장된 파일들: {files}")
    else:
        print("저장 디렉토리가 아직 생성되지 않음")

def main():
    """메인 실행 함수"""
    print("Lab 2 - Step 1: ChromaDB 기초")
    print("벡터 데이터베이스 기본 사용법 학습\n")
    
    # API 키 확인
    if not validate_api_keys():
        print("API 키 설정이 필요합니다.")
        return
    
    try:
        # 1. 기본 CRUD 작업
        collection = demonstrate_basic_operations()
        
        # 2. 임베딩 방법 비교
        compare_embedding_methods(collection)
        
        # 3. 고급 기능 탐색
        explore_collection_features(collection)
        
        # 4. 영속성 기능
        demonstrate_persistence()
        
        print("\n" + "=" * 50)
        print("ChromaDB 기초 학습 완료!")
        print("=" * 50)
        print("\n학습한 내용:")
        print("• ChromaDB 클라이언트 설정 및 컬렉션 관리")
        print("• 기본 CRUD 작업 (생성, 읽기, 수정, 삭제)")
        print("• 자동 vs 수동 임베딩 방법 비교")
        print("• 메타데이터 필터링 및 고급 검색")
        print("• 메모리 vs 영속성 모드 차이점")
        
        print("\n다음 단계:")
        print("document_indexing.py를 실행하여 대용량 문서 인덱싱을 학습해보세요!")
        
    except Exception as e:
        print(f"실습 중 오류 발생: {e}")
        print("ChromaDB 설치 상태를 확인해주세요: pip install chromadb")

if __name__ == "__main__":
    main() 