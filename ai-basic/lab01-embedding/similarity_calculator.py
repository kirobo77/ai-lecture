"""
Lab 1 - Step 2: 유사도 계산 실습
코사인 유사도와 다양한 거리 측정 방법을 학습합니다
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.utils import EmbeddingUtils, SimilarityUtils
from shared.config import validate_api_keys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from typing import List, Dict, Any

def implement_cosine_similarity():
    """코사인 유사도 직접 구현 및 검증"""
    print("=" * 50)
    print("코사인 유사도 직접 구현")
    print("=" * 50)
    
    def manual_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도 수동 구현"""
        # 벡터를 numpy 배열로 변환
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        # 내적 계산
        dot_product = np.dot(v1, v2)
        
        # 벡터 크기 계산 (L2 norm)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        # 코사인 유사도 계산
        similarity = dot_product / (norm_v1 * norm_v2)
        
        print(f"계산 과정:")
        print(f"  내적(dot product): {dot_product:.4f}")
        print(f"  벡터1 크기: {norm_v1:.4f}")
        print(f"  벡터2 크기: {norm_v2:.4f}")
        print(f"  코사인 유사도: {similarity:.4f}")
        
        return similarity
    
    # 테스트 벡터 생성
    test_texts = [
        "머신러닝은 인공지능의 한 분야입니다.",
        "딥러닝은 머신러닝의 하위 기술입니다."
    ]
    
    print("테스트 텍스트:")
    for i, text in enumerate(test_texts):
        print(f"  {i+1}. {text}")
    
    # 임베딩 생성
    embeddings = EmbeddingUtils.get_embeddings_batch(test_texts)
    
    print("\n수동 구현 결과:")
    manual_sim = manual_cosine_similarity(embeddings[0], embeddings[1])
    
    print("\n라이브러리 결과:")
    lib_sim = SimilarityUtils.cosine_similarity_score(embeddings[0], embeddings[1])
    sklearn_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    print(f"  유틸리티 함수: {lib_sim:.6f}")
    print(f"  scikit-learn: {sklearn_sim:.6f}")
    
    print(f"\n구현 검증: 차이 = {abs(manual_sim - lib_sim):.8f}")

def compare_distance_metrics():
    """다양한 거리 측정 방법 비교"""
    print("\n" + "=" * 50)
    print("다양한 거리 측정 방법 비교")
    print("=" * 50)
    
    # 다양한 주제의 텍스트
    texts = [
        "인공지능과 머신러닝 기술 발전",
        "AI와 딥러닝 알고리즘 연구",
        "맛있는 한국 음식 김치찌개",
        "전통 한식 요리법과 재료",
        "월드컵 축구 경기 결과",
        "프로야구 시즌 우승팀 예측"
    ]
    
    print("분석할 텍스트들:")
    for i, text in enumerate(texts):
        print(f"  {i+1}. {text}")
    
    # 임베딩 생성
    embeddings = EmbeddingUtils.get_embeddings_batch(texts)
    embeddings_array = np.array(embeddings)
    
    # 다양한 유사도/거리 계산
    print("\n거리 행렬 계산...")
    
    # 1. 코사인 유사도
    cosine_sim_matrix = cosine_similarity(embeddings_array)
    
    # 2. 유클리드 거리
    euclidean_dist_matrix = euclidean_distances(embeddings_array)
    
    # 3. 맨하탄 거리 (L1)
    manhattan_dist_matrix = np.array([
        [np.sum(np.abs(emb1 - emb2)) for emb2 in embeddings_array]
        for emb1 in embeddings_array
    ])
    
    # 결과 출력
    print("\n코사인 유사도 행렬 (높을수록 유사):")
    print_similarity_matrix(cosine_sim_matrix, texts)
    
    print("\n유클리드 거리 행렬 (낮을수록 유사):")
    print_distance_matrix(euclidean_dist_matrix[:3, :3], texts[:3])  # 크기 제한
    
    return {
        'texts': texts,
        'embeddings': embeddings,
        'cosine_similarity': cosine_sim_matrix,
        'euclidean_distance': euclidean_dist_matrix,
        'manhattan_distance': manhattan_dist_matrix
    }

def print_similarity_matrix(matrix, texts, decimals=3):
    """유사도 행렬 예쁘게 출력"""
    print("    ", end="")
    for i in range(min(len(texts), 4)):  # 처음 4개만
        print(f"{i+1:>8}", end="")
    print()
    
    for i in range(min(len(texts), 4)):
        print(f"{i+1:2}. ", end="")
        for j in range(min(len(texts), 4)):
            print(f"{matrix[i][j]:8.{decimals}f}", end="")
        print(f"  - {texts[i][:30]}...")

def print_distance_matrix(matrix, texts, decimals=2):
    """거리 행렬 예쁘게 출력"""
    print("    ", end="")
    for i in range(len(texts)):
        print(f"{i+1:>8}", end="")
    print()
    
    for i in range(len(texts)):
        print(f"{i+1:2}. ", end="")
        for j in range(len(texts)):
            print(f"{matrix[i][j]:8.{decimals}f}", end="")
        print(f"  - {texts[i][:30]}...")

def analyze_similarity_patterns(results):
    """유사도 패턴 분석"""
    print("\n" + "=" * 50)
    print("유사도 패턴 분석")
    print("=" * 50)
    
    texts = results['texts']
    cosine_sim = results['cosine_similarity']
    
    # 가장 유사한 쌍 찾기
    print("가장 유사한 텍스트 쌍:")
    max_sim = 0
    max_pair = (0, 0)
    
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            if cosine_sim[i][j] > max_sim:
                max_sim = cosine_sim[i][j]
                max_pair = (i, j)
    
    print(f"  유사도: {max_sim:.4f}")
    print(f"  텍스트 1: {texts[max_pair[0]]}")
    print(f"  텍스트 2: {texts[max_pair[1]]}")
    
    # 가장 다른 쌍 찾기
    print("\n가장 다른 텍스트 쌍:")
    min_sim = 1.0
    min_pair = (0, 0)
    
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            if cosine_sim[i][j] < min_sim:
                min_sim = cosine_sim[i][j]
                min_pair = (i, j)
    
    print(f"  유사도: {min_sim:.4f}")
    print(f"  텍스트 1: {texts[min_pair[0]]}")
    print(f"  텍스트 2: {texts[min_pair[1]]}")
    
    # 주제별 클러스터링 분석
    print("\n주제별 유사도 분석:")
    
    # AI 관련 텍스트 (0, 1)
    ai_similarity = cosine_sim[0][1]
    print(f"  AI 주제 내 유사도: {ai_similarity:.4f}")
    
    # 음식 관련 텍스트 (2, 3)
    food_similarity = cosine_sim[2][3]
    print(f"  음식 주제 내 유사도: {food_similarity:.4f}")
    
    # 스포츠 관련 텍스트 (4, 5)
    sports_similarity = cosine_sim[4][5]
    print(f"  스포츠 주제 내 유사도: {sports_similarity:.4f}")
    
    # 주제 간 유사도
    ai_food_similarity = cosine_sim[0][2]
    ai_sports_similarity = cosine_sim[0][4]
    food_sports_similarity = cosine_sim[2][4]
    
    print(f"\n주제 간 유사도:")
    print(f"  AI ↔ 음식: {ai_food_similarity:.4f}")
    print(f"  AI ↔ 스포츠: {ai_sports_similarity:.4f}")
    print(f"  음식 ↔ 스포츠: {food_sports_similarity:.4f}")

def interactive_similarity_test():
    """대화형 유사도 테스트"""
    print("\n" + "=" * 50)
    print("대화형 유사도 테스트")
    print("=" * 50)
    
    print("두 문장을 입력하면 유사도를 계산해드립니다!")
    print("(종료하려면 'quit' 입력)")
    
    while True:
        print("\n" + "-" * 30)
        text1 = input("첫 번째 문장: ").strip()
        
        if text1.lower() == 'quit':
            break
            
        text2 = input("두 번째 문장: ").strip()
        
        if text2.lower() == 'quit':
            break
        
        try:
            # 임베딩 생성
            print("임베딩 생성 중...")
            emb1 = EmbeddingUtils.get_embedding(text1)
            emb2 = EmbeddingUtils.get_embedding(text2)
            
            # 유사도 계산
            cosine_sim = SimilarityUtils.cosine_similarity_score(emb1, emb2)
            euclidean_dist = np.linalg.norm(np.array(emb1) - np.array(emb2))
            
            # 결과 출력
            print(f"\n결과:")
            print(f"  코사인 유사도: {cosine_sim:.4f}")
            print(f"  유클리드 거리: {euclidean_dist:.4f}")
            
            # 해석
            if cosine_sim > 0.8:
                print("  매우 유사한 문장입니다!")
            elif cosine_sim > 0.6:
                print("  어느 정도 유사한 문장입니다.")
            elif cosine_sim > 0.4:
                print("  약간 관련이 있는 문장입니다.")
            else:
                print("  다른 주제의 문장입니다.")
                
        except Exception as e:
            print(f"오류 발생: {e}")

def main():
    """메인 실행 함수"""
    print("Lab 1 - Step 2: 유사도 계산 실습")
    print("다양한 유사도 측정 방법 학습\n")
    
    # API 키 확인
    if not validate_api_keys():
        print("API 키 설정이 필요합니다.")
        return
    
    try:
        # 1. 코사인 유사도 직접 구현
        implement_cosine_similarity()
        
        # 2. 다양한 거리 측정 방법 비교
        results = compare_distance_metrics()
        
        # 3. 유사도 패턴 분석
        analyze_similarity_patterns(results)
        
        # 4. 대화형 테스트
        print("\n대화형 테스트를 실행하시겠습니까? (y/n): ", end="")
        if input().lower().startswith('y'):
            interactive_similarity_test()
        
        print("\n" + "=" * 50)
        print("유사도 계산 실습 완료!")
        print("=" * 50)
        print("\n학습한 내용:")
        print("• 코사인 유사도의 수학적 원리와 구현")
        print("• 다양한 거리 측정 방법의 특성")
        print("• 벡터 공간에서의 의미적 관계 분석")
        print("• 주제별 텍스트 클러스터링 패턴")
        
        print("\n다음 단계:")
        print("document_search.py를 실행하여 문서 검색 엔진을 구현해보세요!")
        
    except Exception as e:
        print(f"실습 중 오류 발생: {e}")

if __name__ == "__main__":
    main() 