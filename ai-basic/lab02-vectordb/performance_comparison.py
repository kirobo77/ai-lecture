"""
Lab 2 - Step 4: 성능 최적화
메모리 vs 디스크 영속성 비교 및 검색 성능 벤치마킹
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ChromaDB 텔레메트리 비활성화 (반복적인 컬렉션 생성으로 인한 텔레메트리 재시도 방지)
os.environ['CHROMA_ANALYTICS'] = 'false'
os.environ['ANONYMIZED_TELEMETRY'] = 'false'  
os.environ['CHROMA_TELEMETRY_DISABLED'] = 'true'

import chromadb
from shared.config import validate_api_keys, CHROMA_PERSIST_DIRECTORY
from shared.utils import EmbeddingUtils, ChromaUtils
import time
import random
import psutil
import shutil
from typing import List, Dict, Any, Tuple
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np

class PerformanceBenchmark:
    """성능 벤치마킹을 위한 클래스"""
    
    def __init__(self):
        self.results = {
            "memory_tests": [],
            "disk_tests": [],
            "search_performance": [],
            "scalability_tests": []
        }
    
    def setup_memory_client(self, collection_name: str = "memory_test"):
        """메모리 기반 클라이언트 설정"""
        client = chromadb.Client()
        openai_ef = ChromaUtils.create_openai_embedding_function()
        try:
            collection = client.create_collection(
                collection_name, 
                embedding_function=openai_ef
        )
        except Exception:
            collection = client.get_collection(
                collection_name,
                embedding_function=openai_ef
            )
        return client, collection
    
    def setup_disk_client(self, collection_name: str = "disk_test", 
                         persist_directory: str = None):
        """디스크 기반 클라이언트 설정"""
        if persist_directory is None:
            persist_directory = CHROMA_PERSIST_DIRECTORY
        
        client = chromadb.PersistentClient(path=persist_directory)
        openai_ef = ChromaUtils.create_openai_embedding_function()

        try:
            collection = client.create_collection(
                collection_name,
                embedding_function=openai_ef
            )
        except Exception as e:
            # 이미 존재하면 삭제 후 재생성
            try:
                client.delete_collection(collection_name)
                collection = client.create_collection(
                    collection_name,
                    embedding_function=openai_ef
                )
            except Exception:
                # 그래도 안 되면 get (임베딩 함수 없이)
                collection = client.get_collection(collection_name)
        return client, collection
    
    def generate_test_documents(self, num_docs: int) -> List[Dict[str, Any]]:
        """테스트용 문서 생성"""
        categories = ["기술", "과학", "경제", "문화", "스포츠", "교육", "의료", "환경"]
        authors = [f"작성자{i}" for i in range(1, 21)]
        
        documents = []
        
        for i in range(num_docs):
            # 다양한 길이의 텍스트 생성
            base_text = f"이것은 {i+1}번째 테스트 문서입니다. "
            
            # 길이 변화 (50자 ~ 1000자)
            repeat_count = random.randint(1, 20)
            content = base_text * repeat_count
            content += f" 카테고리는 {random.choice(categories)}이고, 중요한 키워드들을 포함하고 있습니다."
            
            doc = {
                "id": f"test_doc_{i+1:06d}",
                "content": content,
                "metadata": {
                    "title": f"테스트 문서 {i+1}",
                    "category": random.choice(categories),
                    "author": random.choice(authors),
                    "date": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                    "importance": random.randint(1, 10),
                    "word_count": len(content.split())
                }
            }
            documents.append(doc)
        
        return documents
    
    def measure_indexing_performance(self, client, collection, documents: List[Dict], 
                                   batch_size: int = 100) -> Dict[str, float]:
        """인덱싱 성능 측정"""
        print(f"  문서 {len(documents)}개 인덱싱 (배치 크기: {batch_size})")
        
        # 메모리 사용량 측정 시작
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        
        # 배치별 인덱싱
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            docs = [doc["content"] for doc in batch]
            metadatas = [doc["metadata"] for doc in batch]
            ids = [doc["id"] for doc in batch]
            
            collection.add(
                documents=docs,
                metadatas=metadatas,
                ids=ids
            )
        
        end_time = time.time()
        
        # 메모리 사용량 측정 종료
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            "indexing_time": end_time - start_time,
            "documents_per_second": len(documents) / (end_time - start_time),
            "memory_used": memory_after - memory_before,
            "final_document_count": collection.count()
        }
    
    def measure_search_performance(self, collection, num_queries: int = 50) -> Dict[str, float]:
        """검색 성능 측정"""
        print(f"  검색 성능 측정 ({num_queries}회 쿼리)")
        
        # 테스트 쿼리들
        test_queries = [
            "기술 혁신", "과학 연구", "경제 성장", "문화 발전", "스포츠 경기",
            "교육 시스템", "의료 서비스", "환경 보호", "데이터 분석", "인공지능",
            "클라우드 컴퓨팅", "모바일 애플리케이션", "웹 개발", "보안 시스템", "네트워크 구조"
        ]
        
        search_times = []
        total_results = 0
        
        start_time = time.time()
        
        for i in range(num_queries):
            query = random.choice(test_queries)
            
            query_start = time.time()
            results = collection.query(
                query_texts=[query],
                n_results=10
            )
            query_end = time.time()
            
            search_times.append(query_end - query_start)
            total_results += len(results['ids'][0]) if results['ids'] else 0
        
        end_time = time.time()
        
        return {
            "total_search_time": end_time - start_time,
            "avg_search_time": np.mean(search_times),
            "min_search_time": np.min(search_times),
            "max_search_time": np.max(search_times),
            "std_search_time": np.std(search_times),
            "queries_per_second": num_queries / (end_time - start_time),
            "avg_results_per_query": total_results / num_queries
        }
    
    def measure_disk_usage(self, persist_directory: str) -> Dict[str, float]:
        """디스크 사용량 측정"""
        if not os.path.exists(persist_directory):
            return {"disk_usage_mb": 0, "file_count": 0}
        
        total_size = 0
        file_count = 0
        
        for dirpath, dirnames, filenames in os.walk(persist_directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                    file_count += 1
                except (OSError, IOError):
                    pass
        
        return {
            "disk_usage_mb": total_size / 1024 / 1024,
            "file_count": file_count
        }

def compare_memory_vs_disk():
    """메모리 vs 디스크 성능 비교"""
    print("\n메모리 vs 디스크 성능 비교")
    print("=" * 50)
    
    benchmark = PerformanceBenchmark()
    
    # 테스트 시나리오
    test_scenarios = [
        {"docs": 100, "name": "소규모"},
        {"docs": 500, "name": "중규모"},
        {"docs": 1000, "name": "대규모"}
    ]
    
    comparison_results = []
    
    for scenario in test_scenarios:
        print(f"\n{scenario['name']} 테스트 ({scenario['docs']}개 문서)")
        print("-" * 30)
        
        documents = benchmark.generate_test_documents(scenario['docs'])
        
        # 1. 메모리 기반 테스트
        print("메모리 기반 ChromaDB:")
        memory_client, memory_collection = benchmark.setup_memory_client(
            f"memory_test_{scenario['docs']}")
        
        memory_results = benchmark.measure_indexing_performance(
            memory_client, memory_collection, documents)
        memory_search = benchmark.measure_search_performance(memory_collection)
        
        print(f"  인덱싱: {memory_results['indexing_time']:.2f}초")
        print(f"  문서/초: {memory_results['documents_per_second']:.1f}")
        print(f"  메모리 사용: {memory_results['memory_used']:.1f}MB")
        print(f"  평균 검색: {memory_search['avg_search_time']*1000:.2f}ms")
        
        # 2. 디스크 기반 테스트
        print("\n디스크 기반 ChromaDB:")
        disk_dir = f"./temp_benchmark_{scenario['docs']}"
        disk_client, disk_collection = benchmark.setup_disk_client(
            f"disk_test_{scenario['docs']}", disk_dir)
        
        disk_results = benchmark.measure_indexing_performance(
            disk_client, disk_collection, documents)
        disk_search = benchmark.measure_search_performance(disk_collection)
        disk_usage = benchmark.measure_disk_usage(disk_dir)
        
        print(f"  인덱싱: {disk_results['indexing_time']:.2f}초")
        print(f"  문서/초: {disk_results['documents_per_second']:.1f}")
        print(f"  메모리 사용: {disk_results['memory_used']:.1f}MB")
        print(f"  디스크 사용: {disk_usage['disk_usage_mb']:.1f}MB")
        print(f"  평균 검색: {disk_search['avg_search_time']*1000:.2f}ms")
        
        # 결과 저장
        comparison_results.append({
            "scenario": scenario['name'],
            "docs": scenario['docs'],
            "memory": {
                "indexing_time": memory_results['indexing_time'],
                "search_time": memory_search['avg_search_time'],
                "memory_usage": memory_results['memory_used']
            },
            "disk": {
                "indexing_time": disk_results['indexing_time'],
                "search_time": disk_search['avg_search_time'],
                "memory_usage": disk_results['memory_used'],
                "disk_usage": disk_usage['disk_usage_mb']
            }
        })

        # 임시 디렉토리 정리 전에 클라이언트 참조 제거
        del disk_collection
        del disk_client
        time.sleep(0.5)  # 파일 핸들 해제 대기

        if os.path.exists(disk_dir):
            try:
                shutil.rmtree(disk_dir)
            except PermissionError:
                # Windows에서 즉시 삭제 안 되면 무시
                print(f"  (임시 파일 삭제 지연: {disk_dir})")
    
    return comparison_results

def analyze_scalability():
    """확장성 분석"""
    print("\n확장성 분석")
    print("=" * 50)
    
    benchmark = PerformanceBenchmark()
    
    # 다양한 문서 수로 테스트
    doc_counts = [100, 250, 500, 750, 1000, 1500, 2000]
    scalability_results = []
    
    client, collection = benchmark.setup_memory_client("scalability_test")
    
    for doc_count in doc_counts:
        print(f"\n문서 수: {doc_count}")
        
        # 새로운 컬렉션 생성
        client, collection = benchmark.setup_memory_client(f"scale_test_{doc_count}")
        
        documents = benchmark.generate_test_documents(doc_count)
        
        # 성능 측정
        indexing_results = benchmark.measure_indexing_performance(
            client, collection, documents, batch_size=50)
        search_results = benchmark.measure_search_performance(collection, num_queries=20)
        
        result = {
            "doc_count": doc_count,
            "indexing_time": indexing_results['indexing_time'],
            "docs_per_second": indexing_results['documents_per_second'],
            "memory_used": indexing_results['memory_used'],
            "avg_search_time": search_results['avg_search_time'],
            "queries_per_second": search_results['queries_per_second']
        }
        
        scalability_results.append(result)
        
        print(f"  인덱싱 시간: {result['indexing_time']:.2f}초")
        print(f"  검색 시간: {result['avg_search_time']*1000:.2f}ms")
        print(f"  메모리 사용: {result['memory_used']:.1f}MB")
    
    return scalability_results

def benchmark_different_batch_sizes():
    """다양한 배치 크기 성능 비교"""
    print("\n배치 크기별 성능 비교")
    print("=" * 50)
    
    benchmark = PerformanceBenchmark()
    documents = benchmark.generate_test_documents(1000)
    
    batch_sizes = [10, 25, 50, 100, 200] # 500 제거 (kt 보안 정책에서 대용량 업로드 감지당함)
    batch_results = []
    
    for batch_size in batch_sizes:
        print(f"\n배치 크기: {batch_size}")
        
        client, collection = benchmark.setup_memory_client(f"batch_test_{batch_size}")
        
        results = benchmark.measure_indexing_performance(
            client, collection, documents, batch_size=batch_size)
        
        batch_results.append({
            "batch_size": batch_size,
            "indexing_time": results['indexing_time'],
            "docs_per_second": results['documents_per_second'],
            "memory_used": results['memory_used']
        })
        
        print(f"  인덱싱 시간: {results['indexing_time']:.2f}초")
        print(f"  처리량: {results['documents_per_second']:.1f} docs/sec")
    
    return batch_results

def test_filtering_performance():
    """필터링 성능 테스트"""
    print("\n필터링 검색 성능 테스트")
    print("=" * 50)
    
    benchmark = PerformanceBenchmark()
    documents = benchmark.generate_test_documents(2000)
    
    client, collection = benchmark.setup_memory_client("filtering_test")
    benchmark.measure_indexing_performance(client, collection, documents)
    
    # 다양한 필터링 조건 테스트
    filter_tests = [
        {
            "name": "필터 없음",
            "filter": None
        },
        {
            "name": "단일 카테고리",
            "filter": {"category": "기술"}
        },
        {
            "name": "다중 카테고리",
            "filter": {"category": {"$in": ["기술", "과학", "경제"]}}
        },
        {
            "name": "복합 조건",
            "filter": {
                "$and": [
                    {"category": {"$in": ["기술", "과학"]}},
                    {"importance": {"$gte": 7}}
                ]
            }
        }
    ]
    
    filtering_results = []
    
    for test in filter_tests:
        print(f"\n{test['name']}:")
        
        search_times = []
        result_counts = []
        
        # 10회 반복 측정
        for _ in range(10):
            start_time = time.time()
            
            if test['filter']:
                results = collection.query(
                    query_texts=["기술 혁신"],
                    n_results=10,
                    where=test['filter']
                )
            else:
                results = collection.query(
                    query_texts=["기술 혁신"],
                    n_results=10
                )
            
            end_time = time.time()
            
            search_times.append(end_time - start_time)
            result_counts.append(len(results['ids'][0]) if results['ids'] else 0)
        
        avg_time = np.mean(search_times)
        avg_results = np.mean(result_counts)
        
        filtering_results.append({
            "filter_type": test['name'],
            "avg_search_time": avg_time,
            "avg_result_count": avg_results
        })
        
        print(f"  평균 검색 시간: {avg_time*1000:.2f}ms")
        print(f"  평균 결과 수: {avg_results:.1f}")

def visualize_performance_results(comparison_results, scalability_results):
    """성능 결과 시각화"""
    try:
        print("\n성능 결과 시각화")
        print("=" * 50)
        
        # 1. 메모리 vs 디스크 비교 차트
        plt.figure(figsize=(15, 10))
        
        # 1-1. 인덱싱 시간 비교
        plt.subplot(2, 3, 1)
        scenarios = [r['scenario'] for r in comparison_results]
        memory_index_times = [r['memory']['indexing_time'] for r in comparison_results]
        disk_index_times = [r['disk']['indexing_time'] for r in comparison_results]
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        plt.bar(x - width/2, memory_index_times, width, label='메모리', alpha=0.8)
        plt.bar(x + width/2, disk_index_times, width, label='디스크', alpha=0.8)
        plt.xlabel('테스트 시나리오')
        plt.ylabel('인덱싱 시간 (초)')
        plt.title('인덱싱 성능 비교')
        plt.xticks(x, scenarios)
        plt.legend()
        
        # 1-2. 검색 시간 비교
        plt.subplot(2, 3, 2)
        memory_search_times = [r['memory']['search_time']*1000 for r in comparison_results]
        disk_search_times = [r['disk']['search_time']*1000 for r in comparison_results]
        
        plt.bar(x - width/2, memory_search_times, width, label='메모리', alpha=0.8)
        plt.bar(x + width/2, disk_search_times, width, label='디스크', alpha=0.8)
        plt.xlabel('테스트 시나리오')
        plt.ylabel('검색 시간 (ms)')
        plt.title('검색 성능 비교')
        plt.xticks(x, scenarios)
        plt.legend()
        
        # 1-3. 메모리 사용량 비교
        plt.subplot(2, 3, 3)
        memory_usage = [r['memory']['memory_usage'] for r in comparison_results]
        disk_memory_usage = [r['disk']['memory_usage'] for r in comparison_results]
        
        plt.bar(x - width/2, memory_usage, width, label='메모리 모드', alpha=0.8)
        plt.bar(x + width/2, disk_memory_usage, width, label='디스크 모드', alpha=0.8)
        plt.xlabel('테스트 시나리오')
        plt.ylabel('메모리 사용량 (MB)')
        plt.title('메모리 사용량 비교')
        plt.xticks(x, scenarios)
        plt.legend()
        
        # 2. 확장성 분석 차트
        if scalability_results:
            doc_counts = [r['doc_count'] for r in scalability_results]
            
            # 2-1. 인덱싱 시간 vs 문서 수
            plt.subplot(2, 3, 4)
            indexing_times = [r['indexing_time'] for r in scalability_results]
            plt.plot(doc_counts, indexing_times, 'o-', linewidth=2, markersize=6)
            plt.xlabel('문서 수')
            plt.ylabel('인덱싱 시간 (초)')
            plt.title('확장성: 인덱싱 시간')
            plt.grid(True, alpha=0.3)
            
            # 2-2. 검색 시간 vs 문서 수
            plt.subplot(2, 3, 5)
            search_times = [r['avg_search_time']*1000 for r in scalability_results]
            plt.plot(doc_counts, search_times, 'o-', color='orange', linewidth=2, markersize=6)
            plt.xlabel('문서 수')
            plt.ylabel('검색 시간 (ms)')
            plt.title('확장성: 검색 시간')
            plt.grid(True, alpha=0.3)
            
            # 2-3. 메모리 사용량 vs 문서 수
            plt.subplot(2, 3, 6)
            memory_usages = [r['memory_used'] for r in scalability_results]
            plt.plot(doc_counts, memory_usages, 'o-', color='green', linewidth=2, markersize=6)
            plt.xlabel('문서 수')
            plt.ylabel('메모리 사용량 (MB)')
            plt.title('확장성: 메모리 사용량')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
        print("성능 분석 차트 저장: performance_analysis.png")
        
        # 차트 표시 (환경에 따라 주석 처리 가능)
        # plt.show()
        
    except ImportError:
        print("matplotlib이 설치되지 않아 시각화를 건너뜁니다.")
        print("시각화를 보려면 'pip install matplotlib'을 실행하세요.")
    except Exception as e:
        print(f"시각화 중 오류 발생: {e}")

def generate_performance_report(comparison_results, scalability_results):
    """성능 분석 보고서 생성"""
    print("\n성능 분석 보고서")
    print("=" * 50)
    
    # 1. 메모리 vs 디스크 요약
    print("\n1. 메모리 vs 디스크 성능 요약")
    print("-" * 30)
    
    for result in comparison_results:
        memory_faster_index = result['memory']['indexing_time'] < result['disk']['indexing_time']
        memory_faster_search = result['memory']['search_time'] < result['disk']['search_time']
        
        index_speedup = result['disk']['indexing_time'] / result['memory']['indexing_time']
        search_speedup = result['disk']['search_time'] / result['memory']['search_time']
        
        print(f"\n{result['scenario']} ({result['docs']}개 문서):")
        print(f"  인덱싱: {'메모리' if memory_faster_index else '디스크'}가 {index_speedup:.1f}x 빠름")
        print(f"  검색: {'메모리' if memory_faster_search else '디스크'}가 {search_speedup:.1f}x 빠름")
        print(f"  디스크 사용량: {result['disk']['disk_usage']:.1f}MB")
    
    # 2. 확장성 분석
    if scalability_results and len(scalability_results) > 1:
        print("\n2. 확장성 분석")
        print("-" * 30)
        
        first = scalability_results[0]
        last = scalability_results[-1]
        
        doc_ratio = last['doc_count'] / first['doc_count']
        time_ratio = last['indexing_time'] / first['indexing_time']
        memory_ratio = last['memory_used'] / first['memory_used'] if first['memory_used'] > 0 else 1
        
        print(f"문서 수 {doc_ratio:.1f}x 증가 시:")
        print(f"  인덱싱 시간: {time_ratio:.1f}x 증가")
        print(f"  메모리 사용: {memory_ratio:.1f}x 증가")
        
        # 선형성 분석
        if time_ratio / doc_ratio < 1.5:
            print("  → 거의 선형적 확장성")
        elif time_ratio / doc_ratio < 3:
            print("  → 준선형적 확장성")
        else:
            print("  → 비선형적 확장성 (최적화 필요)")
    
    # 3. 권장사항
    print("\n3. 권장사항")
    print("-" * 30)
    print("메모리 모드 권장 상황:")
    print("  • 빠른 프로토타이핑")
    print("  • 임시 데이터 처리")
    print("  • 고속 검색이 중요한 경우")
    
    print("\n디스크 모드 권장 상황:")
    print("  • 데이터 영속성이 필요한 경우")
    print("  • 대용량 데이터 처리")
    print("  • 메모리 제약이 있는 환경")
    print("  • 프로덕션 환경")

def main():
    """메인 실행 함수"""
    print("Lab 2 - Step 4: 성능 최적화")
    print("메모리 vs 디스크 영속성 비교 및 성능 벤치마킹\n")
    
    # API 키 확인
    if not validate_api_keys():
        print("API 키 설정이 필요합니다.")
        return
    
    try:
        # 1. 메모리 vs 디스크 비교
        comparison_results = compare_memory_vs_disk()
        
        # 2. 확장성 분석
        scalability_results = analyze_scalability()
        
        # 3. 배치 크기 최적화
        batch_results = benchmark_different_batch_sizes()
        
        # 4. 필터링 성능 테스트
        test_filtering_performance()
        
        # 5. 결과 시각화
        visualize_performance_results(comparison_results, scalability_results)
        
        # 6. 성능 보고서 생성
        generate_performance_report(comparison_results, scalability_results)
        
        print("\n" + "=" * 50)
        print("성능 최적화 학습 완료!")
        print("=" * 50)
        print("\n학습한 내용:")
        print("• 메모리 vs 디스크 기반 저장소 성능 비교")
        print("• 문서 수에 따른 확장성 분석")
        print("• 배치 크기 최적화 전략")
        print("• 필터링 검색 성능 특성")
        print("• 성능 벤치마킹 및 분석 방법론")
        
        print("\nLab 2 전체 완료!")
        print("이제 Lab 3으로 진행하여 RAG 시스템을 구축해보세요!")
        
    except Exception as e:
        print(f"실습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 