"""
Lab 1 - Step 1: 기본 임베딩 생성
OpenAI 임베딩 API를 사용하여 텍스트를 벡터로 변환하는 기초 실습
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.utils import EmbeddingUtils, TextUtils, PerformanceUtils
from shared.config import validate_api_keys
import numpy as np
import time

def demonstrate_single_embedding():
    """단일 텍스트 임베딩 생성 데모"""
    print("=" * 50)
    print("단일 텍스트 임베딩 생성")
    print("=" * 50)
    
    # 샘플 텍스트
    text = "인공지능은 인간의 지능을 모방하여 학습, 추론, 문제 해결 등을 수행하는 기술입니다."
    
    print(f"원본 텍스트: {text}")
    print(f"토큰 수: {TextUtils.count_tokens(text)}")
    
    # 임베딩 생성 (시간 측정)
    start_time = time.time()
    embedding = EmbeddingUtils.get_embedding(text)
    end_time = time.time()
    
    print(f"임베딩 생성 시간: {end_time - start_time:.2f}초")
    print(f"임베딩 차원: {len(embedding)}")
    print(f"임베딩 벡터 샘플 (첫 10개): {embedding[:10]}")
    print(f"벡터 범위: [{min(embedding):.4f}, {max(embedding):.4f}]")
    print(f"벡터 평균: {np.mean(embedding):.4f}")
    print(f"벡터 크기(L2 norm): {np.linalg.norm(embedding):.4f}")
    
    return embedding

def demonstrate_batch_embedding():
    """배치 텍스트 임베딩 생성 데모"""
    print("\n" + "=" * 50)
    print("배치 텍스트 임베딩 생성")
    print("=" * 50)
    
    # 다양한 주제의 샘플 텍스트들
    texts = [
        "머신러닝은 데이터로부터 패턴을 학습하는 AI 기술입니다.",
        "김치찌개는 한국의 대표적인 음식 중 하나입니다.",
        "축구는 전 세계에서 가장 인기 있는 스포츠입니다.",
        "파리는 프랑스의 수도이자 아름다운 여행 도시입니다.",
        "딥러닝은 다층 신경망을 사용하는 머신러닝 방법입니다."
    ]
    
    print(f"처리할 텍스트 수: {len(texts)}")
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text}")
    
    # 배치 임베딩 생성 (시간 측정)
    start_time = time.time()
    embeddings = EmbeddingUtils.get_embeddings_batch(texts)
    end_time = time.time()
    
    print(f"\n배치 임베딩 생성 시간: {end_time - start_time:.2f}초")
    print(f"평균 처리 시간/텍스트: {(end_time - start_time) / len(texts):.3f}초")
    print(f"각 임베딩 차원: {len(embeddings[0])}")
    
    # 개별 임베딩 시간과 비교
    print("\n개별 처리와 시간 비교...")
    individual_start = time.time()
    individual_embeddings = []
    for text in texts:
        embedding = EmbeddingUtils.get_embedding(text)
        individual_embeddings.append(embedding)
    individual_end = time.time()
    
    individual_time = individual_end - individual_start
    batch_time = end_time - start_time
    
    print(f"개별 처리 시간: {individual_time:.2f}초")
    print(f"배치 처리 시간: {batch_time:.2f}초")
    print(f"성능 향상: {individual_time / batch_time:.1f}배 빠름")
    
    return embeddings, texts

def analyze_embedding_properties(embeddings, texts):
    """임베딩의 특성 분석"""
    print("\n" + "=" * 50)
    print("임베딩 특성 분석")
    print("=" * 50)
    
    # 벡터 통계
    embeddings_array = np.array(embeddings)
    
    print(f"임베딩 행렬 형태: {embeddings_array.shape}")
    print(f"전체 평균: {np.mean(embeddings_array):.4f}")
    print(f"전체 표준편차: {np.std(embeddings_array):.4f}")
    print(f"평균 벡터 크기: {np.mean([np.linalg.norm(emb) for emb in embeddings]):.4f}")
    
    # 영벡터와의 거리 (원점에서의 거리)
    print("\n각 임베딩의 원점 거리:")
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        distance = np.linalg.norm(embedding)
        print(f"  {i+1}. {distance:.4f} - {text[:50]}...")
    
    # 차원별 분포 분석
    print(f"\n차원별 값 분포 (첫 10개 차원):")
    for dim in range(min(10, len(embeddings[0]))):
        values = [emb[dim] for emb in embeddings]
        print(f"  차원 {dim+1}: [{min(values):.4f}, {max(values):.4f}] (평균: {np.mean(values):.4f})")

def compare_similar_texts():
    """유사한 텍스트들의 임베딩 비교"""
    print("\n" + "=" * 50)
    print("유사한 텍스트 임베딩 비교")
    print("=" * 50)
    
    # 유사한 의미의 텍스트 쌍
    similar_pairs = [
        ("머신러닝은 AI의 핵심 기술입니다.", "인공지능의 중요한 분야가 머신러닝입니다."),
        ("강아지는 충성스러운 동물입니다.", "개는 사람의 가장 좋은 친구입니다."),
        ("비가 내리고 있습니다.", "날씨가 흐리고 비가 옵니다.")
    ]
    
    from shared.utils import SimilarityUtils
    
    for i, (text1, text2) in enumerate(similar_pairs, 1):
        print(f"\n쌍 {i}:")
        print(f"  A: {text1}")
        print(f"  B: {text2}")
        
        # 임베딩 생성
        emb1 = EmbeddingUtils.get_embedding(text1)
        emb2 = EmbeddingUtils.get_embedding(text2)
        
        # 유사도 계산
        similarity = SimilarityUtils.cosine_similarity_score(emb1, emb2)
        print(f"  코사인 유사도: {similarity:.4f}")
        
        # 유클리드 거리
        euclidean_dist = np.linalg.norm(np.array(emb1) - np.array(emb2))
        print(f"  유클리드 거리: {euclidean_dist:.4f}")

def main():
    """메인 실행 함수"""
    print("Lab 1 - Step 1: 기본 임베딩 생성 실습")
    print("OpenAI 임베딩 API를 사용한 텍스트 벡터 변환\n")
    
    # API 키 확인
    if not validate_api_keys():
        print("API 키 설정이 필요합니다. .env 파일을 확인해주세요.")
        return
    
    try:
        # 1. 단일 임베딩 생성
        single_embedding = demonstrate_single_embedding()
        
        # 2. 배치 임베딩 생성
        batch_embeddings, texts = demonstrate_batch_embedding()
        
        # 3. 임베딩 특성 분석
        analyze_embedding_properties(batch_embeddings, texts)
        
        # 4. 유사한 텍스트 비교
        compare_similar_texts()
        
        print("\n" + "=" * 50)
        print("기본 임베딩 생성 실습 완료!")
        print("=" * 50)
        print("\n학습한 내용:")
        print("• 텍스트를 고차원 벡터로 변환하는 방법")
        print("• 배치 처리를 통한 성능 최적화")
        print("• 임베딩 벡터의 기본 특성과 구조")
        print("• 의미적으로 유사한 텍스트의 벡터 관계")
        
        print("\n다음 단계:")
        print("similarity_calculator.py를 실행하여 유사도 계산을 학습해보세요!")
        
    except Exception as e:
        print(f"실습 중 오류 발생: {e}")
        print("문제 해결 방법:")
        print("1. API 키가 올바르게 설정되었는지 확인")
        print("2. 인터넷 연결 상태 확인")
        print("3. OpenAI 계정의 크레딧 잔액 확인")

if __name__ == "__main__":
    main() 