"""
Lab 1 - Step 4: 임베딩 시각화 및 분석
t-SNE를 활용한 임베딩 벡터 시각화 및 의미적 클러스터링 분석
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.utils import EmbeddingUtils, print_progress
from shared.config import validate_api_keys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
from typing import List, Dict, Any

# 한글 폰트 설정 (안전한 방법)
import platform

def setup_korean_font():
    """한글 폰트를 안전하게 설정"""
    try:
        import matplotlib.font_manager as fm
        
        # macOS에서 사용 가능한 한글 폰트 목록 (우선순위 순)
        if platform.system() == 'Darwin':  # macOS
            preferred_fonts = [
                'Apple SD Gothic Neo',
                'AppleGothic',
                'Nanum Gothic',
                'NanumGothic',
                'Helvetica'
            ]
        elif platform.system() == 'Windows':
            preferred_fonts = ['Malgun Gothic', 'Microsoft YaHei', 'Arial']
        else:  # Linux
            preferred_fonts = ['Nanum Gothic', 'DejaVu Sans']
        
        # 시스템에 설치된 폰트 목록 가져오기
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 우선순위에 따라 사용 가능한 폰트 찾기
        selected_font = None
        for font in preferred_fonts:
            if font in available_fonts:
                selected_font = font
                break
        
        if selected_font:
            plt.rcParams['font.family'] = selected_font
            print(f"한글 폰트 설정: {selected_font}")
        else:
            # 대체 방법: sans-serif 계열 사용
            plt.rcParams['font.family'] = 'sans-serif'
            print("한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")
            
    except Exception as e:
        print(f"폰트 설정 오류: {e}")
        plt.rcParams['font.family'] = 'sans-serif'

# 폰트 설정 실행
setup_korean_font()
plt.rcParams['axes.unicode_minus'] = False

def create_diverse_texts():
    """다양한 주제의 텍스트 데이터 생성"""
    print("=" * 50)
    print("시각화용 텍스트 데이터 생성")
    print("=" * 50)
    
    texts_data = {
        # AI/기술 (빨간색)
        "AI": [
            "인공지능은 인간의 지능을 모방하는 컴퓨터 기술입니다.",
            "머신러닝은 데이터로부터 패턴을 학습하는 AI 기법입니다.",
            "딥러닝은 다층 신경망을 사용하는 머신러닝 방법입니다.",
            "자연어처리는 컴퓨터가 인간의 언어를 이해하는 기술입니다.",
            "컴퓨터 비전은 이미지를 분석하고 인식하는 AI 분야입니다."
        ],
        
        # 음식 (초록색)
        "음식": [
            "김치찌개는 한국의 대표적인 전통 음식입니다.",
            "스파게티는 이탈리아의 유명한 파스타 요리입니다.",
            "초밥은 일본의 전통적인 생선 요리입니다.",
            "타코는 멕시코의 인기 있는 길거리 음식입니다.",
            "카레는 인도에서 유래된 향신료 요리입니다."
        ],
        
        # 스포츠 (파란색)
        "스포츠": [
            "축구는 전 세계에서 가장 인기 있는 스포츠입니다.",
            "농구는 실내에서 하는 팀 스포츠입니다.",
            "테니스는 라켓을 사용하는 개인 스포츠입니다.",
            "수영은 물에서 하는 전신 운동입니다.",
            "골프는 정확성이 중요한 개인 스포츠입니다."
        ],
        
        # 자연 (보라색)
        "자연": [
            "산은 지구의 아름다운 자연 경관입니다.",
            "바다는 지구 표면의 대부분을 차지합니다.",
            "숲은 다양한 동식물의 서식지입니다.",
            "강은 담수가 흐르는 자연 수로입니다.",
            "사막은 건조한 기후의 자연 지형입니다."
        ],
        
        # 교통 (주황색)
        "교통": [
            "자동차는 도로에서 사용하는 교통수단입니다.",
            "기차는 철도를 이용한 대중교통입니다.",
            "비행기는 하늘을 나는 교통수단입니다.",
            "배는 물 위에서 움직이는 교통수단입니다.",
            "자전거는 친환경적인 개인 교통수단입니다."
        ]
    }
    
    # 플랫 리스트와 라벨 생성
    all_texts = []
    all_labels = []
    
    for category, texts in texts_data.items():
        all_texts.extend(texts)
        all_labels.extend([category] * len(texts))
    
    print(f"생성된 데이터:")
    for category, texts in texts_data.items():
        print(f"  {category}: {len(texts)}개 텍스트")
    
    print(f"  총합: {len(all_texts)}개 텍스트")
    
    return all_texts, all_labels, texts_data

def generate_embeddings(texts):
    """텍스트들의 임베딩 생성"""
    print("\n임베딩 생성 중...")
    
    embeddings = []
    batch_size = 10
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = EmbeddingUtils.get_embeddings_batch(batch_texts)
        embeddings.extend(batch_embeddings)
        
        print_progress(
            min(i + batch_size, len(texts)), 
            len(texts), 
            "임베딩 생성"
        )
    
    print(f"\n임베딩 생성 완료: {len(embeddings)}개")
    return np.array(embeddings)

def apply_tsne(embeddings, perplexity=5, random_state=42):
    """t-SNE를 사용한 차원 축소"""
    print(f"\nt-SNE 차원 축소 실행 (perplexity={perplexity})")
    
    # t-SNE 적용
    tsne = TSNE(
        n_components=2, 
        perplexity=perplexity,
        random_state=random_state,
        max_iter=1000,
        learning_rate=200
    )
    
    embeddings_2d = tsne.fit_transform(embeddings)
    
    print(f"t-SNE 완료: {embeddings.shape} → {embeddings_2d.shape}")
    
    return embeddings_2d

def apply_pca(embeddings, n_components=2):
    """PCA를 사용한 차원 축소"""
    print(f"\nPCA 차원 축소 실행 (n_components={n_components})")
    
    pca = PCA(n_components=n_components, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    
    explained_variance = pca.explained_variance_ratio_
    print(f"PCA 완료: 설명 분산 비율 = {explained_variance}")
    print(f"   PC1: {explained_variance[0]:.3f}, PC2: {explained_variance[1]:.3f}")
    
    return embeddings_2d, explained_variance

def visualize_embeddings(embeddings_2d, labels, method="t-SNE", title_suffix="", texts=None):
    """임베딩 시각화"""
    print(f"\n{method} 시각화 생성 중...")
    
    # 색상 맵 설정
    unique_labels = sorted(list(set(labels)))
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink', 'gray']
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    
    # 그래프 생성
    plt.figure(figsize=(15, 10))
    
    for label in unique_labels:
        # 해당 라벨의 데이터만 추출
        mask = np.array(labels) == label
        x = embeddings_2d[mask, 0]
        y = embeddings_2d[mask, 1]
        
        plt.scatter(x, y, c=color_map[label], label=label, alpha=0.7, s=100)
    
    # 텍스트 라벨 추가 (각 점에 원본 텍스트의 일부 표시)
    if texts is not None:
        for i, (x, y) in enumerate(embeddings_2d):
            # 텍스트 첫 15자만 표시 (너무 길면 가독성 떨어짐)
            short_text = texts[i][:15] + "..." if len(texts[i]) > 15 else texts[i]
            plt.annotate(short_text, 
                        (x, y), 
                        xytext=(5, 5), 
                        textcoords='offset points',
                        fontsize=8, 
                        alpha=0.8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.title(f'{method} Visualization of Text Embeddings{title_suffix}', fontsize=16, fontweight='bold')
    plt.xlabel(f'{method} Component 1', fontsize=12)
    plt.ylabel(f'{method} Component 2', fontsize=12)
    plt.legend(title='Category', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 그래프 저장
    filename = f"embedding_{method.lower().replace('-', '')}_visualization{title_suffix.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"시각화 저장: {filename}")
    
    plt.show()

def analyze_clusters(embeddings_2d, labels, texts):
    """클러스터 분석"""
    print("\n클러스터 분석")
    print("=" * 30)
    
    # K-means 클러스터링
    n_clusters = len(set(labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_2d)
    
    # 클러스터 중심점
    centers = kmeans.cluster_centers_
    
    print(f"K-means 클러스터링 결과 (k={n_clusters}):")
    
    # 각 클러스터 분석
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        cluster_texts = [texts[j] for j in range(len(texts)) if cluster_mask[j]]
        cluster_categories = [labels[j] for j in range(len(labels)) if cluster_mask[j]]
        
        print(f"\n클러스터 {i} ({len(cluster_texts)}개):")
        
        # 카테고리 분포
        from collections import Counter
        category_count = Counter(cluster_categories)
        for category, count in category_count.most_common():
            print(f"  {category}: {count}개")
        
        # 샘플 텍스트
        print(f"  샘플 텍스트:")
        for j, text in enumerate(cluster_texts[:2]):  # 처음 2개만
            print(f"    - {text}")
    
    # 클러스터 시각화
    plt.figure(figsize=(12, 8))
    
    # 실제 카테고리별 색상
    unique_labels = sorted(list(set(labels)))
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        x = embeddings_2d[mask, 0]
        y = embeddings_2d[mask, 1]
        plt.scatter(x, y, c=color_map[label], label=f'True: {label}', alpha=0.7, s=100)
    
    # 클러스터 중심점 표시
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=200, linewidths=3, label='Cluster Centers')
    
    plt.title('Clustering Analysis with K-means', fontsize=16, fontweight='bold')
    plt.xlabel('Component 1', fontsize=12)
    plt.ylabel('Component 2', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("clustering_analysis.png", dpi=300, bbox_inches='tight')
    print("\n클러스터 분석 저장: clustering_analysis.png")
    plt.show()
    
    return cluster_labels

def compare_methods(embeddings, labels, texts):
    """t-SNE와 PCA 비교"""
    print("\nt-SNE vs PCA 비교")
    print("=" * 30)
    
    # PCA 적용
    embeddings_pca, explained_var = apply_pca(embeddings)
    
    # t-SNE 적용
    embeddings_tsne = apply_tsne(embeddings, perplexity=5)
    
    # 비교 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 색상 설정
    unique_labels = sorted(list(set(labels)))
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # PCA 그래프
    plt.sca(ax1)
    for label in unique_labels:
        mask = np.array(labels) == label
        x = embeddings_pca[mask, 0]
        y = embeddings_pca[mask, 1]
        ax1.scatter(x, y, c=color_map[label], label=label, alpha=0.7, s=100)
    
    # PCA 텍스트 라벨 추가
    for i, (x, y) in enumerate(embeddings_pca):
        short_text = texts[i][:12] + "..." if len(texts[i]) > 12 else texts[i]
        ax1.annotate(short_text, 
                    (x, y), 
                    xytext=(3, 3), 
                    textcoords='offset points',
                    fontsize=7, 
                    alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    ax1.set_title(f'PCA (설명분산: {sum(explained_var):.3f})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('PC1', fontsize=12)
    ax1.set_ylabel('PC2', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # t-SNE 그래프
    plt.sca(ax2)
    for label in unique_labels:
        mask = np.array(labels) == label
        x = embeddings_tsne[mask, 0]
        y = embeddings_tsne[mask, 1]
        ax2.scatter(x, y, c=color_map[label], label=label, alpha=0.7, s=100)
    
    # t-SNE 텍스트 라벨 추가
    for i, (x, y) in enumerate(embeddings_tsne):
        short_text = texts[i][:12] + "..." if len(texts[i]) > 12 else texts[i]
        ax2.annotate(short_text, 
                    (x, y), 
                    xytext=(3, 3), 
                    textcoords='offset points',
                    fontsize=7, 
                    alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    ax2.set_title('t-SNE (perplexity=5)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('t-SNE 1', fontsize=12)
    ax2.set_ylabel('t-SNE 2', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("method_comparison.png", dpi=300, bbox_inches='tight')
    print("비교 시각화 저장: method_comparison.png")
    plt.show()
    
    # 분석 결과
    print("\n분석 결과:")
    print("PCA:")
    print(f"  - 전체 분산의 {sum(explained_var):.1%} 설명")
    print("  - 선형 변환으로 전역 구조 보존")
    print("  - 계산 속도 빠름")
    
    print("\nt-SNE:")
    print("  - 비선형 변환으로 지역 구조 강조")
    print("  - 클러스터링이 더 명확히 분리됨")
    print("  - 계산 시간 오래 걸림")

def interactive_analysis():
    """대화형 분석 도구"""
    print("\n대화형 분석 도구")
    print("=" * 30)
    print("원하는 분석을 선택하세요:")
    print("1. 다른 perplexity 값으로 t-SNE 시도")
    print("2. 사용자 정의 텍스트 추가")
    print("3. 유사도 분석")
    
    choice = input("선택 (1-3): ").strip()
    
    if choice == "1":
        perplexity_analysis()
    elif choice == "2":
        custom_text_analysis()
    elif choice == "3":
        similarity_analysis()
    else:
        print("잘못된 선택입니다.")

def print_detailed_analysis(embeddings_2d, labels, texts):
    """각 데이터 포인트의 상세 분석"""
    print("\n" + "=" * 50)
    print("데이터 포인트별 상세 분석")
    print("=" * 50)
    
    # 카테고리별 텍스트 목록
    categories = {}
    for i, (label, text) in enumerate(zip(labels, texts)):
        if label not in categories:
            categories[label] = []
        categories[label].append({
            'index': i,
            'text': text,
            'position': embeddings_2d[i]
        })
    
    # 각 카테고리 출력
    for category, items in categories.items():
        print(f"\n{category} 카테고리:")
        for i, item in enumerate(items, 1):
            pos = item['position']
            print(f"  {i}. 위치: ({pos[0]:.2f}, {pos[1]:.2f})")
            print(f"     텍스트: {item['text']}")
    
    # 카테고리 간 거리 분석 (이상치 찾기)
    print(f"\n카테고리 경계 분석:")
    
    from sklearn.metrics.pairwise import euclidean_distances
    distances = euclidean_distances(embeddings_2d)
    
    # 각 포인트에 대해 다른 카테고리와의 최소 거리 계산
    for i, (label, text) in enumerate(zip(labels, texts)):
        min_dist_other_category = float('inf')
        closest_other = None
        
        for j, other_label in enumerate(labels):
            if i != j and label != other_label:
                dist = distances[i][j]
                if dist < min_dist_other_category:
                    min_dist_other_category = dist
                    closest_other = (j, other_label, texts[j])
        
        if closest_other and min_dist_other_category < 5.0:  # 임계값
            print(f"\n경계에 있는 포인트 발견:")
            print(f"   {label}: {text[:30]}...")
            print(f"   가장 가까운 다른 카테고리 ({closest_other[1]}): {closest_other[2][:30]}...")
            print(f"   거리: {min_dist_other_category:.2f}")

def perplexity_analysis():
    """다양한 perplexity 값으로 t-SNE 분석"""
    print("Perplexity 값 비교 분석")
    
    texts, labels, _ = create_diverse_texts()
    embeddings = generate_embeddings(texts)
    
    perplexity_values = [2, 5, 10, 30]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, perp in enumerate(perplexity_values):
        print(f"\nPerplexity {perp} 계산 중...")
        embeddings_2d = apply_tsne(embeddings, perplexity=perp)
        
        ax = axes[i]
        unique_labels = sorted(list(set(labels)))
        colors = ['red', 'green', 'blue', 'purple', 'orange']
        color_map = {label: colors[j] for j, label in enumerate(unique_labels)}
        
        for label in unique_labels:
            mask = np.array(labels) == label
            x = embeddings_2d[mask, 0]
            y = embeddings_2d[mask, 1]
            ax.scatter(x, y, c=color_map[label], label=label, alpha=0.7, s=80)
        
        ax.set_title(f'Perplexity = {perp}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("perplexity_comparison.png", dpi=300, bbox_inches='tight')
    print("Perplexity 비교 저장: perplexity_comparison.png")
    plt.show()

def main():
    """메인 실행 함수"""
    print("Lab 1 - Step 4: 임베딩 시각화 및 분석")
    print("t-SNE와 PCA를 활용한 벡터 공간 시각화\n")
    
    # API 키 확인
    if not validate_api_keys():
        print("API 키 설정이 필요합니다.")
        return
    
    try:
        # 1. 데이터 생성
        texts, labels, texts_data = create_diverse_texts()
        
        # 2. 임베딩 생성
        embeddings = generate_embeddings(texts)
        
        # 3. t-SNE 시각화
        embeddings_tsne = apply_tsne(embeddings)
        visualize_embeddings(embeddings_tsne, labels, "t-SNE", texts=texts)
        
        # 4. 클러스터 분석
        cluster_labels = analyze_clusters(embeddings_tsne, labels, texts)
        
        # 5. 데이터 포인트별 상세 분석
        print_detailed_analysis(embeddings_tsne, labels, texts)
        
        # 6. 방법론 비교
        compare_methods(embeddings, labels, texts)
        
        # 7. 대화형 분석 (선택사항)
        print("\n추가 분석을 진행하시겠습니까? (y/n): ", end="")
        if input().lower().startswith('y'):
            interactive_analysis()
        
        print("\n" + "=" * 50)
        print("임베딩 시각화 및 분석 완료!")
        print("=" * 50)
        print("\n학습한 내용:")
        print("• 고차원 임베딩의 2D 시각화 방법")
        print("• t-SNE와 PCA의 특성과 차이점")
        print("• 의미적 클러스터링 패턴 분석")
        print("• K-means를 활용한 자동 클러스터링")
        
        print("\nLab 1 전체 완료!")
        print("이제 Lab 2로 진행하여 벡터 데이터베이스를 구축해보세요!")
        
        # 생성된 파일들 안내
        print("\n생성된 파일들:")
        print("• embedding_tsne_visualization.png - t-SNE 시각화")
        print("• clustering_analysis.png - 클러스터 분석")
        print("• method_comparison.png - t-SNE vs PCA 비교")
        
    except Exception as e:
        print(f"실습 중 오류 발생: {e}")
        print("matplotlib 설치가 필요할 수 있습니다: pip install matplotlib seaborn")

if __name__ == "__main__":
    main() 