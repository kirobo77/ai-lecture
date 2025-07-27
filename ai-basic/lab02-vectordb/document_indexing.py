"""
Lab 2 - Step 2: 문서 인덱싱 시스템
대용량 문서 컬렉션의 효율적인 배치 처리 및 인덱싱
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from shared.config import validate_api_keys, CHROMA_PERSIST_DIRECTORY
from shared.utils import EmbeddingUtils, TextUtils
import time
import random
from typing import List, Dict, Any, Generator
from datetime import datetime, timedelta
import json

class DocumentIndexer:
    """대용량 문서 인덱싱을 위한 클래스"""
    
    def __init__(self, collection_name="document_indexer", persist_directory=None):
        self.collection_name = collection_name
        self.persist_directory = persist_directory or CHROMA_PERSIST_DIRECTORY
        self.client = None
        self.collection = None
        self.stats = {
            "total_documents": 0,
            "processed_documents": 0,
            "failed_documents": 0,
            "start_time": None,
            "end_time": None
        }
    
    def initialize(self, reset=False):
        """인덱서 초기화"""
        print("문서 인덱서 초기화")
        print("=" * 40)
        
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # 컬렉션 초기화
        if reset:
            try:
                self.client.delete_collection(self.collection_name)
                print(f"기존 컬렉션 '{self.collection_name}' 삭제됨")
            except Exception as e:
                print(f"컬렉션 삭제 실패: {e}")
        
        # 컬렉션 생성 또는 가져오기
        try:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "대용량 문서 인덱싱용 컬렉션"}
            )
            print(f"새 컬렉션 '{self.collection_name}' 생성됨")
        except Exception:
            self.collection = self.client.get_collection(self.collection_name)
            print(f"기존 컬렉션 '{self.collection_name}' 로드됨")
        
        print(f"컬렉션 문서 수: {self.collection.count()}")
    
    def chunk_document(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        """문서를 청크로 분할"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # 문장 경계에서 자르기 시도
            if end < len(text):
                # 마지막 문장 부호 찾기
                last_sentence_end = max(
                    text.rfind('.', start, end),
                    text.rfind('!', start, end),
                    text.rfind('?', start, end),
                    text.rfind('\n', start, end)
                )
                
                if last_sentence_end > start:
                    end = last_sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - chunk_overlap
        
        return chunks
    
    def process_document_batch(self, documents: List[Dict[str, Any]], batch_size: int = 10) -> Dict[str, int]:
        """문서 배치 처리"""
        print(f"\n문서 배치 처리 시작")
        print(f"총 문서 수: {len(documents)}")
        print(f"배치 크기: {batch_size}")
        print("-" * 30)
        
        results = {"success": 0, "failed": 0, "chunks_created": 0}
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            print(f"배치 {batch_num}/{total_batches} 처리 중...")
            
            batch_docs = []
            batch_metadatas = []
            batch_ids = []
            
            for doc in batch:
                try:
                    # 문서 청킹
                    chunks = self.chunk_document(doc['content'])
                    results["chunks_created"] += len(chunks)
                    
                    # 각 청크에 대해 메타데이터 생성
                    for chunk_idx, chunk in enumerate(chunks):
                        chunk_id = f"{doc['id']}_chunk_{chunk_idx}"
                        
                        chunk_metadata = doc['metadata'].copy()
                        chunk_metadata.update({
                            "chunk_index": chunk_idx,
                            "total_chunks": len(chunks),
                            "chunk_id": chunk_id,
                            "parent_document": doc['id'],
                            "chunk_size": len(chunk)
                        })
                        
                        batch_docs.append(chunk)
                        batch_metadatas.append(chunk_metadata)
                        batch_ids.append(chunk_id)
                    
                    results["success"] += 1
                    
                except Exception as e:
                    print(f"문서 {doc['id']} 처리 실패: {e}")
                    results["failed"] += 1
            
            # ChromaDB에 배치 추가
            if batch_docs:
                try:
                    self.collection.add(
                        documents=batch_docs,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                    print(f"  배치 {batch_num}: {len(batch_docs)}개 청크 추가됨")
                except Exception as e:
                    print(f"  배치 {batch_num} 추가 실패: {e}")
            
            # 진행률 표시
            progress = min((i + batch_size) / len(documents) * 100, 100)
            print(f"  진행률: {progress:.1f}%")
        
        return results
    
    def index_documents_with_progress(self, documents: List[Dict[str, Any]], 
                                    batch_size: int = 10, 
                                    show_progress: bool = True) -> Dict[str, Any]:
        """진행률 추적과 함께 문서 인덱싱"""
        self.stats["total_documents"] = len(documents)
        self.stats["start_time"] = time.time()
        
        print(f"\n대용량 문서 인덱싱 시작")
        print("=" * 40)
        print(f"총 문서 수: {len(documents)}")
        print(f"배치 크기: {batch_size}")
        print(f"예상 배치 수: {(len(documents) + batch_size - 1) // batch_size}")
        
        # 배치 처리
        results = self.process_document_batch(documents, batch_size)
        
        # 통계 업데이트
        self.stats["processed_documents"] = results["success"]
        self.stats["failed_documents"] = results["failed"]
        self.stats["end_time"] = time.time()
        
        # 결과 출력
        processing_time = self.stats["end_time"] - self.stats["start_time"]
        
        print("\n" + "=" * 40)
        print("인덱싱 완료!")
        print("=" * 40)
        print(f"처리된 문서: {results['success']}/{len(documents)}")
        print(f"실패한 문서: {results['failed']}")
        print(f"생성된 청크: {results['chunks_created']}")
        print(f"처리 시간: {processing_time:.2f}초")
        print(f"문서당 평균 시간: {processing_time/len(documents):.3f}초")
        print(f"컬렉션 총 문서 수: {self.collection.count()}")
        
        return {
            "stats": self.stats,
            "results": results,
            "processing_time": processing_time
        }

def generate_sample_documents(num_docs: int = 50) -> List[Dict[str, Any]]:
    """샘플 문서 생성"""
    print(f"샘플 문서 {num_docs}개 생성")
    print("=" * 40)
    
    categories = ["기술", "과학", "경제", "문화", "스포츠"]
    authors = ["김연구", "박박사", "이전문", "최분석", "정리뷰"]
    
    # 다양한 길이의 샘플 텍스트 템플릿
    text_templates = {
        "기술": [
            "인공지능 기술의 발전은 우리 사회에 혁명적인 변화를 가져오고 있습니다. 머신러닝과 딥러닝 알고리즘의 발달로 자율주행차, 의료 진단, 금융 분석 등 다양한 분야에서 활용되고 있습니다.",
            "클라우드 컴퓨팅은 기업의 IT 인프라를 근본적으로 변화시키고 있습니다. AWS, Azure, GCP와 같은 플랫폼을 통해 기업들은 더 유연하고 효율적인 서비스를 제공할 수 있게 되었습니다.",
            "블록체인 기술은 중앙화된 시스템의 한계를 극복하는 새로운 패러다임을 제시합니다. 암호화폐뿐만 아니라 공급망 관리, 투표 시스템, 디지털 신원 인증 등에서 활용 가능성이 높습니다."
        ],
        "과학": [
            "양자 컴퓨팅은 전통적인 컴퓨터의 한계를 뛰어넘는 혁신적인 기술입니다. 양자 중첩과 얽힘 현상을 이용하여 복잡한 문제를 기존 컴퓨터보다 훨씬 빠르게 해결할 수 있습니다.",
            "유전자 편집 기술인 CRISPR은 의학과 농업 분야에 새로운 가능성을 열어주고 있습니다. 질병 치료와 작물 개량에 활용되어 인류의 삶의 질을 향상시킬 것으로 기대됩니다.",
            "재생 에너지 기술의 발전으로 태양광, 풍력, 수력 발전의 효율성이 크게 향상되었습니다. 이는 기후 변화 대응과 지속 가능한 발전에 핵심적인 역할을 하고 있습니다."
        ],
        "경제": [
            "디지털 경제의 성장으로 전통적인 비즈니스 모델이 급격히 변화하고 있습니다. 플랫폼 경제, 공유 경제, 구독 경제 등 새로운 형태의 비즈니스가 등장하고 있습니다.",
            "암호화폐와 중앙은행 디지털화폐(CBDC)의 등장으로 금융 시스템이 근본적인 변화를 겪고 있습니다. 이는 결제 시스템과 통화 정책에 새로운 도전과 기회를 제공합니다.",
            "ESG(환경, 사회, 지배구조) 투자가 글로벌 금융 시장의 새로운 트렌드로 자리잡고 있습니다. 지속 가능한 경영과 사회적 책임이 기업 가치 평가의 중요한 요소가 되고 있습니다."
        ]
    }
    
    documents = []
    
    for i in range(num_docs):
        category = random.choice(categories)
        author = random.choice(authors)
        
        # 날짜 생성 (최근 1년 내)
        days_ago = random.randint(0, 365)
        doc_date = datetime.now() - timedelta(days=days_ago)
        
        # 문서 길이 랜덤 설정 (짧은 문서 30%, 중간 문서 50%, 긴 문서 20%)
        length_type = random.choices(
            ["short", "medium", "long"], 
            weights=[30, 50, 20]
        )[0]
        
        # 텍스트 생성
        if category in text_templates:
            base_text = random.choice(text_templates[category])
        else:
            base_text = f"{category} 분야의 전문적인 내용을 다루는 문서입니다."
        
        # 길이에 따라 텍스트 확장
        if length_type == "short":
            content = base_text
        elif length_type == "medium":
            content = base_text + " " + base_text + " 추가적인 상세 정보와 분석 내용이 포함되어 있습니다."
        else:  # long
            content = base_text + " " + base_text + " " + base_text
            content += " 더욱 심층적인 분석과 다양한 관점에서의 해석, 그리고 미래 전망에 대한 내용이 포함되어 있습니다."
            content += " 관련 연구 결과와 통계 데이터, 전문가 의견 등이 종합적으로 제시됩니다."
        
        doc = {
            "id": f"doc_{i+1:03d}",
            "content": content,
            "metadata": {
                "title": f"{category} - {i+1:03d}번 문서",
                "category": category,
                "author": author,
                "date": doc_date.strftime("%Y-%m-%d"),
                "length_type": length_type,
                "word_count": len(content.split()),
                "created_at": datetime.now().isoformat()
            }
        }
        
        documents.append(doc)
    
    # 통계 출력
    categories_count = {}
    length_count = {}
    for doc in documents:
        cat = doc["metadata"]["category"]
        length = doc["metadata"]["length_type"]
        categories_count[cat] = categories_count.get(cat, 0) + 1
        length_count[length] = length_count.get(length, 0) + 1
    
    print(f"카테고리별 분포: {categories_count}")
    print(f"길이별 분포: {length_count}")
    
    return documents

def demonstrate_chunking_strategies():
    """다양한 청킹 전략 비교"""
    print("\n청킹 전략 비교")
    print("=" * 40)
    
    # 긴 텍스트 샘플
    long_text = """
    인공지능의 발전은 21세기의 가장 중요한 기술 혁명 중 하나입니다. 
    머신러닝과 딥러닝 기술의 발달로 우리는 이전에는 불가능했던 일들을 할 수 있게 되었습니다.
    
    자연어 처리 분야에서는 GPT, BERT와 같은 대형 언어 모델이 등장하여 
    인간 수준에 가까운 텍스트 이해와 생성이 가능해졌습니다.
    이러한 모델들은 번역, 요약, 질문 답변 등 다양한 작업에서 뛰어난 성능을 보여주고 있습니다.
    
    컴퓨터 비전 분야에서도 CNN과 같은 딥러닝 기술의 발달로 
    이미지 인식, 객체 탐지, 의료 영상 분석 등에서 혁신적인 발전이 이루어지고 있습니다.
    
    하지만 인공지능 기술의 발전과 함께 윤리적, 사회적 문제들도 대두되고 있습니다.
    편향성, 개인정보 보호, 일자리 대체 등의 문제에 대한 해결책 마련이 필요합니다.
    """
    
    indexer = DocumentIndexer()
    
    # 다양한 청킹 전략 테스트
    strategies = [
        {"chunk_size": 200, "chunk_overlap": 20, "name": "작은 청크"},
        {"chunk_size": 500, "chunk_overlap": 50, "name": "중간 청크"},
        {"chunk_size": 1000, "chunk_overlap": 100, "name": "큰 청크"},
        {"chunk_size": 300, "chunk_overlap": 0, "name": "겹침 없음"},
        {"chunk_size": 300, "chunk_overlap": 100, "name": "많은 겹침"}
    ]
    
    for strategy in strategies:
        print(f"\n{strategy['name']} 전략:")
        print(f"  청크 크기: {strategy['chunk_size']}")
        print(f"  겹침: {strategy['chunk_overlap']}")
        
        chunks = indexer.chunk_document(
            long_text.strip(),
            chunk_size=strategy['chunk_size'],
            chunk_overlap=strategy['chunk_overlap']
        )
        
        print(f"  생성된 청크 수: {len(chunks)}")
        print(f"  평균 청크 길이: {sum(len(chunk) for chunk in chunks) / len(chunks):.1f}")
        
        # 첫 번째 청크 미리보기
        if chunks:
            print(f"  첫 번째 청크: {chunks[0][:100]}...")

def analyze_indexing_performance():
    """인덱싱 성능 분석"""
    print("\n인덱싱 성능 분석")
    print("=" * 40)
    
    # 다양한 배치 크기로 테스트
    batch_sizes = [5, 10, 20]
    document_counts = [20, 50]
    
    results = []
    
    for doc_count in document_counts:
        print(f"\n문서 수: {doc_count}")
        print("-" * 20)
        
        documents = generate_sample_documents(doc_count)
        
        for batch_size in batch_sizes:
            print(f"\n배치 크기: {batch_size}")
            
            indexer = DocumentIndexer(f"perf_test_{doc_count}_{batch_size}")
            indexer.initialize(reset=True)
            
            start_time = time.time()
            result = indexer.index_documents_with_progress(
                documents, 
                batch_size=batch_size,
                show_progress=False
            )
            end_time = time.time()
            
            processing_time = end_time - start_time
            docs_per_second = doc_count / processing_time
            
            print(f"  처리 시간: {processing_time:.2f}초")
            print(f"  초당 문서 수: {docs_per_second:.1f}")
            
            results.append({
                "doc_count": doc_count,
                "batch_size": batch_size,
                "processing_time": processing_time,
                "docs_per_second": docs_per_second
            })
    
    # 결과 요약
    print(f"\n성능 분석 결과 요약")
    print("=" * 40)
    for result in results:
        print(f"문서 {result['doc_count']}개, 배치 {result['batch_size']}: "
              f"{result['docs_per_second']:.1f} docs/sec")

def demonstrate_error_handling():
    """에러 처리 시연"""
    print("\n에러 처리 시연")
    print("=" * 40)
    
    # 의도적으로 문제가 있는 문서들 생성
    problematic_docs = [
        {
            "id": "good_doc_1",
            "content": "정상적인 문서입니다.",
            "metadata": {"category": "normal", "author": "test"}
        },
        {
            "id": "empty_content",
            "content": "",  # 빈 내용
            "metadata": {"category": "empty", "author": "test"}
        },
        {
            "id": "good_doc_2", 
            "content": "또 다른 정상적인 문서입니다.",
            "metadata": {"category": "normal", "author": "test"}
        },
        {
            "id": "very_long_content",
            "content": "x" * 10000,  # 매우 긴 내용
            "metadata": {"category": "long", "author": "test"}
        }
    ]
    
    indexer = DocumentIndexer("error_test")
    indexer.initialize(reset=True)
    
    print("문제가 있는 문서들을 포함한 배치 처리 테스트:")
    result = indexer.index_documents_with_progress(
        problematic_docs,
        batch_size=2
    )
    
    print(f"\n처리 결과:")
    print(f"성공: {result['results']['success']}")
    print(f"실패: {result['results']['failed']}")
    print(f"생성된 청크: {result['results']['chunks_created']}")

def main():
    """메인 실행 함수"""
    print("Lab 2 - Step 2: 문서 인덱싱 시스템")
    print("대용량 문서 컬렉션의 효율적인 배치 처리\n")
    
    # API 키 확인
    if not validate_api_keys():
        print("API 키 설정이 필요합니다.")
        return
    
    try:
        # 1. 청킹 전략 비교
        demonstrate_chunking_strategies()
        
        # 2. 기본 인덱싱 데모
        print("\n" + "=" * 50)
        print("기본 문서 인덱싱 데모")
        print("=" * 50)
        
        documents = generate_sample_documents(30)
        indexer = DocumentIndexer("demo_collection")
        indexer.initialize(reset=True)
        
        result = indexer.index_documents_with_progress(documents, batch_size=10)
        
        # 3. 성능 분석
        analyze_indexing_performance()
        
        # 4. 에러 처리
        demonstrate_error_handling()
        
        print("\n" + "=" * 50)
        print("문서 인덱싱 시스템 학습 완료!")
        print("=" * 50)
        print("\n학습한 내용:")
        print("• 대용량 문서의 배치 처리 기법")
        print("• 다양한 청킹 전략과 성능 비교")
        print("• 진행률 추적 및 상태 모니터링")
        print("• 에러 처리 및 복구 메커니즘")
        print("• 인덱싱 성능 최적화 방법")
        
        print("\n다음 단계:")
        print("metadata_filtering.py를 실행하여 메타데이터 필터링을 학습해보세요!")
        
    except Exception as e:
        print(f"실습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 