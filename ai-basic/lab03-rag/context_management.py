"""
Lab 3 - Step 3: 컨텍스트 관리
효율적인 컨텍스트 윈도우 관리 및 최적화 기법
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from shared.config import validate_api_keys, CHROMA_PERSIST_DIRECTORY, OPENAI_API_KEY, CHAT_MODEL, MAX_TOKENS
from shared.utils import EmbeddingUtils, ChatUtils, ChromaUtils
import tiktoken
import numpy as np
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

@dataclass
class DocumentChunk:
    """문서 청크를 나타내는 데이터 클래스"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_idx: int = 0
    end_idx: int = 0
    token_count: int = 0
    relevance_score: float = 0.0
    chunk_type: str = "content"  # content, summary, title

@dataclass 
class ContextWindow:
    """컨텍스트 윈도우를 나타내는 데이터 클래스"""
    chunks: List[DocumentChunk] = field(default_factory=list)
    total_tokens: int = 0
    max_tokens: int = 4000
    relevance_threshold: float = 0.3
    compression_ratio: float = 1.0

class TokenCounter:
    """토큰 계산을 위한 유틸리티 클래스"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # 모델을 찾을 수 없는 경우 기본 인코딩 사용
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 계산"""
        try:
            return len(self.encoding.encode(text))
        except Exception:
            # 인코딩 실패 시 근사값 사용 (1토큰 ≈ 4자)
            return len(text) // 4
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """지정된 토큰 수로 텍스트 자르기"""
        try:
            tokens = self.encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
            
            truncated_tokens = tokens[:max_tokens]
            return self.encoding.decode(truncated_tokens)
        except Exception:
            # 인코딩 실패 시 문자 기반 자르기
            char_limit = max_tokens * 4
            return text[:char_limit] if len(text) > char_limit else text

class DocumentChunker(ABC):
    """문서 청킹을 위한 추상 기본 클래스"""
    
    @abstractmethod
    def chunk_document(self, content: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        pass

class FixedSizeChunker(DocumentChunker):
    """고정 크기 청킹"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.token_counter = TokenCounter()
    
    def chunk_document(self, content: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """고정 크기로 문서 청킹"""
        chunks = []
        text_length = len(content)
        
        start_idx = 0
        chunk_id = 0
        
        while start_idx < text_length:
            end_idx = min(start_idx + self.chunk_size, text_length)
            
            # 단어 경계에서 자르기
            if end_idx < text_length:
                # 뒤로 가면서 공백 찾기
                while end_idx > start_idx and content[end_idx] not in [' ', '\n', '\t']:
                    end_idx -= 1
                
                if end_idx == start_idx:  # 공백을 찾지 못한 경우
                    end_idx = start_idx + self.chunk_size
            
            chunk_content = content[start_idx:end_idx].strip()
            
            if chunk_content:  # 빈 청크 제외
                chunk = DocumentChunk(
                    id=f"{metadata.get('id', 'doc')}_{chunk_id}",
                    content=chunk_content,
                    metadata=metadata.copy(),
                    start_idx=start_idx,
                    end_idx=end_idx,
                    token_count=self.token_counter.count_tokens(chunk_content),
                    chunk_type="content"
                )
                chunks.append(chunk)
                chunk_id += 1
            
            # 다음 청크 시작 위치 (오버랩 고려)
            start_idx = max(start_idx + 1, end_idx - self.overlap)
        
        return chunks

class SemanticChunker(DocumentChunker):
    """의미 기반 청킹 (문단, 문장 경계 고려)"""
    
    def __init__(self, max_chunk_size: int = 800, min_chunk_size: int = 200):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.token_counter = TokenCounter()
    
    def chunk_document(self, content: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """의미 기반으로 문서 청킹"""
        chunks = []
        
        # 문단 단위로 분할
        paragraphs = content.split('\n\n')
        
        current_chunk = ""
        current_size = 0
        chunk_id = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            para_size = len(paragraph)
            
            # 현재 청크에 추가할 수 있는지 확인
            if current_size + para_size <= self.max_chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_size += para_size
            else:
                # 현재 청크 저장 (최소 크기 이상인 경우)
                if current_chunk and current_size >= self.min_chunk_size:
                    chunk = DocumentChunk(
                        id=f"{metadata.get('id', 'doc')}_{chunk_id}",
                        content=current_chunk,
                        metadata=metadata.copy(),
                        token_count=self.token_counter.count_tokens(current_chunk),
                        chunk_type="content"
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                
                # 새 청크 시작
                current_chunk = paragraph
                current_size = para_size
                
                # 단일 문단이 너무 큰 경우 문장 단위로 분할
                if current_size > self.max_chunk_size:
                    sentence_chunks = self._split_by_sentences(paragraph, metadata, chunk_id)
                    chunks.extend(sentence_chunks)
                    chunk_id += len(sentence_chunks)
                    current_chunk = ""
                    current_size = 0
        
        # 마지막 청크 처리
        if current_chunk and current_size >= self.min_chunk_size:
            chunk = DocumentChunk(
                id=f"{metadata.get('id', 'doc')}_{chunk_id}",
                content=current_chunk,
                metadata=metadata.copy(),
                token_count=self.token_counter.count_tokens(current_chunk),
                chunk_type="content"
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_by_sentences(self, text: str, metadata: Dict[str, Any], base_chunk_id: int) -> List[DocumentChunk]:
        """문장 단위로 텍스트 분할"""
        sentences = re.split(r'[.!?]\s+', text)
        chunks = []
        
        current_chunk = ""
        current_size = 0
        local_chunk_id = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_size = len(sentence)
            
            if current_size + sentence_size <= self.max_chunk_size:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
                current_size += sentence_size
            else:
                if current_chunk:
                    chunk = DocumentChunk(
                        id=f"{metadata.get('id', 'doc')}_{base_chunk_id}_{local_chunk_id}",
                        content=current_chunk,
                        metadata=metadata.copy(),
                        token_count=self.token_counter.count_tokens(current_chunk),
                        chunk_type="content"
                    )
                    chunks.append(chunk)
                    local_chunk_id += 1
                
                current_chunk = sentence
                current_size = sentence_size
        
        if current_chunk:
            chunk = DocumentChunk(
                id=f"{metadata.get('id', 'doc')}_{base_chunk_id}_{local_chunk_id}",
                content=current_chunk,
                metadata=metadata.copy(),
                token_count=self.token_counter.count_tokens(current_chunk),
                chunk_type="content"
            )
            chunks.append(chunk)
        
        return chunks

class ContextCompressor:
    """컨텍스트 압축을 위한 클래스"""
    
    def __init__(self):
        self.token_counter = TokenCounter()
    
    def compress_context(self, chunks: List[DocumentChunk], 
                        max_tokens: int, method: str = "truncate") -> List[DocumentChunk]:
        """컨텍스트 압축"""
        
        if method == "truncate":
            return self._truncate_compression(chunks, max_tokens)
        elif method == "summarize":
            return self._summarize_compression(chunks, max_tokens)
        elif method == "selective":
            return self._selective_compression(chunks, max_tokens)
        else:
            return chunks
    
    def _truncate_compression(self, chunks: List[DocumentChunk], max_tokens: int) -> List[DocumentChunk]:
        """단순 자르기 압축"""
        compressed_chunks = []
        current_tokens = 0
        
        for chunk in sorted(chunks, key=lambda x: x.relevance_score, reverse=True):
            if current_tokens + chunk.token_count <= max_tokens:
                compressed_chunks.append(chunk)
                current_tokens += chunk.token_count
            else:
                # 남은 토큰으로 청크 일부 포함
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 50:  # 최소 토큰 수 확보
                    truncated_content = self.token_counter.truncate_to_tokens(
                        chunk.content, remaining_tokens
                    )
                    
                    truncated_chunk = DocumentChunk(
                        id=chunk.id + "_truncated",
                        content=truncated_content,
                        metadata=chunk.metadata,
                        token_count=remaining_tokens,
                        relevance_score=chunk.relevance_score,
                        chunk_type="truncated"
                    )
                    compressed_chunks.append(truncated_chunk)
                break
        
        return compressed_chunks
    
    def _summarize_compression(self, chunks: List[DocumentChunk], max_tokens: int) -> List[DocumentChunk]:
        """요약 기반 압축"""
        if not chunks:
            return []
        
        # 토큰 수가 이미 제한 내인 경우
        total_tokens = sum(chunk.token_count for chunk in chunks)
        if total_tokens <= max_tokens:
            return chunks
        
        # 높은 관련성 청크들을 선택적으로 요약
        high_relevance_chunks = [
            chunk for chunk in chunks 
            if chunk.relevance_score >= 0.5
        ]
        
        if not high_relevance_chunks:
            high_relevance_chunks = chunks[:3]  # 최소 3개
        
        summarized_chunks = []
        
        for chunk in high_relevance_chunks:
            try:
                # 청크 요약 생성
                summary = self._generate_summary(chunk.content)
                summary_tokens = self.token_counter.count_tokens(summary)
                
                summary_chunk = DocumentChunk(
                    id=chunk.id + "_summary",
                    content=summary,
                    metadata=chunk.metadata,
                    token_count=summary_tokens,
                    relevance_score=chunk.relevance_score,
                    chunk_type="summary"
                )
                summarized_chunks.append(summary_chunk)
                
            except Exception as e:
                print(f"요약 생성 실패: {e}")
                # 요약 실패 시 원본 청크 일부 사용
                truncated_content = self.token_counter.truncate_to_tokens(
                    chunk.content, min(chunk.token_count // 2, 200)
                )
                truncated_chunk = DocumentChunk(
                    id=chunk.id + "_truncated",
                    content=truncated_content,
                    metadata=chunk.metadata,
                    token_count=self.token_counter.count_tokens(truncated_content),
                    relevance_score=chunk.relevance_score,
                    chunk_type="truncated"
                )
                summarized_chunks.append(truncated_chunk)
        
        # 토큰 제한 내로 최종 조정
        return self._truncate_compression(summarized_chunks, max_tokens)
    
    def _selective_compression(self, chunks: List[DocumentChunk], max_tokens: int) -> List[DocumentChunk]:
        """선택적 압축 (중요도 기반)"""
        # 관련성 점수로 정렬
        sorted_chunks = sorted(chunks, key=lambda x: x.relevance_score, reverse=True)
        
        selected_chunks = []
        current_tokens = 0
        
        # 고관련성 청크 우선 선택
        for chunk in sorted_chunks:
            if chunk.relevance_score >= 0.7:  # 높은 관련성
                if current_tokens + chunk.token_count <= max_tokens:
                    selected_chunks.append(chunk)
                    current_tokens += chunk.token_count
            elif chunk.relevance_score >= 0.4:  # 중간 관련성
                # 요약해서 포함
                summary_length = min(chunk.token_count // 2, 150)
                if current_tokens + summary_length <= max_tokens:
                    try:
                        summary = self._generate_summary(chunk.content, summary_length)
                        summary_chunk = DocumentChunk(
                            id=chunk.id + "_selective",
                            content=summary,
                            metadata=chunk.metadata,
                            token_count=self.token_counter.count_tokens(summary),
                            relevance_score=chunk.relevance_score,
                            chunk_type="summary"
                        )
                        selected_chunks.append(summary_chunk)
                        current_tokens += summary_chunk.token_count
                    except:
                        # 요약 실패 시 건너뛰기
                        continue
        
        return selected_chunks
    
    def _generate_summary(self, content: str, max_length: int = 200) -> str:
        """텍스트 요약 생성"""
        try:
            prompt = f"""
다음 텍스트를 {max_length}자 이내로 핵심 내용만 요약해주세요:

{content}

요약:"""
            
            messages = [
                {"role": "system", "content": "당신은 텍스트 요약 전문가입니다."},
                {"role": "user", "content": prompt}
            ]
            
            summary = ChatUtils.get_chat_response(messages)
            
            # 길이 제한 확인
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            
            return summary.strip()
            
        except Exception as e:
            raise Exception(f"요약 생성 실패: {e}")

class ContextManager:
    """컨텍스트 윈도우 관리를 위한 메인 클래스"""
    
    def __init__(self, max_tokens: int = 4000, compression_threshold: float = 0.8):
        self.max_tokens = max_tokens
        self.compression_threshold = compression_threshold  # 압축 시작 임계값
        self.token_counter = TokenCounter()
        self.compressor = ContextCompressor()
        
        # 청킹 전략들
        self.chunkers = {
            "fixed": FixedSizeChunker(),
            "semantic": SemanticChunker()
        }
    
    def prepare_context(self, documents: List[Dict[str, Any]], 
                       query: str, chunking_strategy: str = "semantic") -> ContextWindow:
        """문서들로부터 컨텍스트 윈도우 준비"""
        
        print(f"컨텍스트 준비 시작 - 전략: {chunking_strategy}")
        
        # 1. 문서 청킹
        all_chunks = []
        chunker = self.chunkers.get(chunking_strategy, self.chunkers["semantic"])
        
        for doc in documents:
            chunks = chunker.chunk_document(doc['content'], doc.get('metadata', {}))
            all_chunks.extend(chunks)
        
        print(f"총 {len(all_chunks)}개 청크 생성")
        
        # 2. 관련성 점수 계산
        all_chunks = self._calculate_relevance_scores(all_chunks, query)
        
        # 3. 관련성 기반 필터링
        relevant_chunks = [
            chunk for chunk in all_chunks 
            if chunk.relevance_score >= 0.2  # 최소 관련성 임계값
        ]
        
        print(f"관련성 필터링 후: {len(relevant_chunks)}개 청크")
        
        # 4. 토큰 수 계산
        total_tokens = sum(chunk.token_count for chunk in relevant_chunks)
        
        # 5. 압축 필요성 판단
        compression_needed = total_tokens > (self.max_tokens * self.compression_threshold)
        
        final_chunks = relevant_chunks
        compression_ratio = 1.0
        
        if compression_needed:
            print(f"컨텍스트 압축 필요: {total_tokens} > {self.max_tokens * self.compression_threshold}")
            
            # 압축 방법 결정 (관련성 점수에 따라)
            avg_relevance = np.mean([chunk.relevance_score for chunk in relevant_chunks])
            
            if avg_relevance > 0.6:
                compression_method = "selective"
            elif avg_relevance > 0.4:
                compression_method = "summarize"
            else:
                compression_method = "truncate"
            
            print(f"압축 방법: {compression_method}")
            
            final_chunks = self.compressor.compress_context(
                relevant_chunks, self.max_tokens, compression_method
            )
            
            final_tokens = sum(chunk.token_count for chunk in final_chunks)
            compression_ratio = final_tokens / total_tokens if total_tokens > 0 else 1.0
            
            print(f"압축 완료: {total_tokens} -> {final_tokens} 토큰 (비율: {compression_ratio:.2f})")
        
        # 6. 컨텍스트 윈도우 구성
        context_window = ContextWindow(
            chunks=final_chunks,
            total_tokens=sum(chunk.token_count for chunk in final_chunks),
            max_tokens=self.max_tokens,
            compression_ratio=compression_ratio
        )
        
        return context_window
    
    def _calculate_relevance_scores(self, chunks: List[DocumentChunk], query: str) -> List[DocumentChunk]:
        """쿼리와 청크 간 관련성 점수 계산"""
        
        try:
            # 쿼리 임베딩 생성
            query_embedding = EmbeddingUtils.get_embedding(query)
            
            # 각 청크의 임베딩 생성 및 유사도 계산
            for chunk in chunks:
                chunk_embedding = EmbeddingUtils.get_embedding(chunk.content)
                
                # 코사인 유사도 계산
                similarity = np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                )
                
                # 0-1 범위로 정규화
                chunk.relevance_score = max(0, (similarity + 1) / 2)
            
        except Exception as e:
            print(f"관련성 점수 계산 실패: {e}")
            # 실패 시 기본 점수 할당
            for chunk in chunks:
                chunk.relevance_score = 0.5
        
        return chunks
    
    def format_context(self, context_window: ContextWindow, include_metadata: bool = True) -> str:
        """컨텍스트 윈도우를 문자열로 포맷팅"""
        
        if not context_window.chunks:
            return ""
        
        formatted_parts = []
        
        # 관련성 순으로 정렬
        sorted_chunks = sorted(
            context_window.chunks, 
            key=lambda x: x.relevance_score, 
            reverse=True
        )
        
        for i, chunk in enumerate(sorted_chunks):
            part = f"[문서 {i+1}]"
            
            if include_metadata and chunk.metadata:
                meta_info = []
                for key, value in chunk.metadata.items():
                    if key in ['category', 'author', 'date', 'type']:
                        meta_info.append(f"{key}: {value}")
                
                if meta_info:
                    part += f" ({', '.join(meta_info)})"
            
            part += f"\n{chunk.content}\n"
            
            # 청크 타입 표시 (요약, 자르기 등)
            if chunk.chunk_type != "content":
                part += f"[{chunk.chunk_type.upper()}]\n"
            
            formatted_parts.append(part)
        
        return "\n".join(formatted_parts)
    
    def get_context_stats(self, context_window: ContextWindow) -> Dict[str, Any]:
        """컨텍스트 윈도우 통계 정보"""
        
        chunk_types = {}
        relevance_scores = []
        
        for chunk in context_window.chunks:
            chunk_type = chunk.chunk_type
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            relevance_scores.append(chunk.relevance_score)
        
        return {
            "total_chunks": len(context_window.chunks),
            "total_tokens": context_window.total_tokens,
            "max_tokens": context_window.max_tokens,
            "utilization": context_window.total_tokens / context_window.max_tokens,
            "compression_ratio": context_window.compression_ratio,
            "chunk_types": chunk_types,
            "avg_relevance": np.mean(relevance_scores) if relevance_scores else 0,
            "min_relevance": np.min(relevance_scores) if relevance_scores else 0,
            "max_relevance": np.max(relevance_scores) if relevance_scores else 0
        }

def create_test_documents() -> List[Dict[str, Any]]:
    """컨텍스트 관리 테스트용 문서 생성"""
    documents = [
        {
            "content": """
            RAG(Retrieval-Augmented Generation) 시스템은 현대 AI 응용 프로그램에서 점점 더 중요해지고 있습니다. 
            
            RAG의 핵심 아이디어는 외부 지식 베이스에서 관련 정보를 검색한 후, 이 정보를 언어 모델에 제공하여 
            더 정확하고 최신의 응답을 생성하는 것입니다.
            
            전통적인 언어 모델의 한계점들:
            1. 훈련 데이터의 컷오프 날짜 이후 정보 부족
            2. 할루시네이션(잘못된 정보 생성) 문제
            3. 도메인 특화 지식의 부족
            4. 실시간 정보 업데이트 불가
            
            RAG 시스템의 구성 요소:
            - 문서 저장소 (Document Store)
            - 벡터 데이터베이스 (Vector Database)  
            - 검색 엔진 (Retriever)
            - 생성 모델 (Generator)
            - 컨텍스트 관리자 (Context Manager)
            
            RAG 시스템의 작동 과정:
            1. 사용자 쿼리 입력
            2. 쿼리 임베딩 생성
            3. 유사도 기반 문서 검색
            4. 관련 문서들의 컨텍스트 구성
            5. 언어 모델에 컨텍스트와 쿼리 전달
            6. 최종 응답 생성
            
            이러한 과정을 통해 RAG는 정확성과 최신성을 크게 향상시킬 수 있습니다.
            """,
            "metadata": {
                "id": "rag_overview",
                "category": "RAG",
                "author": "AI연구팀",
                "date": "2024-01-15",
                "type": "overview"
            }
        },
        {
            "content": """
            컨텍스트 관리는 RAG 시스템에서 가장 중요한 요소 중 하나입니다. 
            
            왜 컨텍스트 관리가 중요한가?
            
            언어 모델에는 토큰 제한이 있습니다. GPT-3.5-turbo는 4,096토큰, GPT-4는 8,192토큰(또는 32,768토큰)의 
            컨텍스트 윈도우를 가집니다. 이 제한 내에서 최대한 효과적인 정보를 제공해야 합니다.
            
            컨텍스트 관리의 주요 과제:
            1. 토큰 제한 내 최적 정보 선별
            2. 관련성 높은 정보 우선순위 부여
            3. 중복 정보 제거
            4. 정보 압축 및 요약
            5. 컨텍스트 일관성 유지
            
            효과적인 컨텍스트 관리 전략:
            
            1. 계층적 청킹 (Hierarchical Chunking)
            - 문서를 의미 단위로 분할
            - 문단, 섹션, 장 단위 구분
            - 중요도에 따른 우선순위 설정
            
            2. 동적 컨텍스트 조정 (Dynamic Context Adjustment)
            - 쿼리 복잡도에 따른 컨텍스트 크기 조절
            - 실시간 관련성 평가
            - 적응형 압축 비율 적용
            
            3. 메타데이터 활용 (Metadata Utilization)
            - 문서 출처, 신뢰도, 최신성 고려
            - 카테고리별 가중치 적용
            - 사용자 권한 및 접근성 확인
            
            이러한 전략들을 통해 제한된 컨텍스트 내에서도 최대한의 정보 가치를 제공할 수 있습니다.
            """,
            "metadata": {
                "id": "context_management",
                "category": "컨텍스트관리",
                "author": "시스템아키텍트",
                "date": "2024-01-12",
                "type": "technical"
            }
        },
        {
            "content": """
            청킹(Chunking) 전략은 대용량 문서를 처리 가능한 크기로 나누는 핵심 기술입니다.
            
            청킹이 필요한 이유:
            - 대부분의 언어 모델은 토큰 제한이 있음
            - 긴 문서는 직접 처리 불가
            - 관련 부분만 선별적으로 사용 가능
            - 검색 정확도 향상
            
            주요 청킹 방법들:
            
            1. 고정 크기 청킹 (Fixed-size Chunking)
            장점: 구현 간단, 일관된 크기
            단점: 의미 단위 무시, 문맥 단절 가능
            
            2. 의미 기반 청킹 (Semantic Chunking)
            장점: 의미 단위 보존, 자연스러운 분할
            단점: 크기 불균등, 구현 복잡
            
            3. 겹침 청킹 (Overlapping Chunking)
            장점: 문맥 연속성 보장
            단점: 저장 공간 증가, 중복 처리 필요
            
            4. 계층적 청킹 (Hierarchical Chunking)
            장점: 다중 해상도 정보 제공
            단점: 복잡한 관리, 높은 계산 비용
            
            청킹 최적화 팁:
            - 문서 타입에 따른 전략 선택
            - 토큰 카운터로 정확한 크기 측정
            - 중요 정보 손실 방지
            - 검색 성능 고려한 크기 설정
            
            실제 구현 시 고려사항:
            - 언어별 특성 (한국어는 어절 단위 고려)
            - 특수 문자 및 마크업 처리
            - 메타데이터 보존
            - 효율적인 저장 및 인덱싱
            """,
            "metadata": {
                "id": "chunking_strategies",
                "category": "문서처리",
                "author": "데이터엔지니어링팀",
                "date": "2024-01-08",
                "type": "guide"
            }
        }
    ]
    
    return documents

def demonstrate_chunking_strategies():
    """다양한 청킹 전략 시연"""
    print("Lab 3 - Step 3: 컨텍스트 관리")
    print("문서 청킹 전략 비교\n")
    
    # 테스트 문서
    test_doc = create_test_documents()[0]
    
    print("원본 문서 정보:")
    print(f"  길이: {len(test_doc['content'])}자")
    print(f"  카테고리: {test_doc['metadata']['category']}")
    print()
    
    # 다양한 청킹 전략 테스트
    strategies = {
        "고정 크기 (500자)": FixedSizeChunker(chunk_size=500, overlap=50),
        "고정 크기 (300자)": FixedSizeChunker(chunk_size=300, overlap=30),
        "의미 기반": SemanticChunker(max_chunk_size=600, min_chunk_size=200)
    }
    
    for strategy_name, chunker in strategies.items():
        print(f"{strategy_name} 청킹 결과:")
        print("-" * 40)
        
        chunks = chunker.chunk_document(test_doc['content'], test_doc['metadata'])
        
        total_tokens = sum(chunk.token_count for chunk in chunks)
        
        print(f"  생성된 청크 수: {len(chunks)}")
        print(f"  총 토큰 수: {total_tokens}")
        print(f"  평균 청크 크기: {total_tokens/len(chunks):.1f} 토큰")
        
        print("\n  청크별 상세:")
        for i, chunk in enumerate(chunks[:3]):  # 처음 3개만 표시
            print(f"    청크 {i+1}: {chunk.token_count} 토큰")
            print(f"             {chunk.content[:80]}...")
        
        if len(chunks) > 3:
            print(f"    ... (총 {len(chunks)}개 청크)")
        
        print()

def demonstrate_context_compression():
    """컨텍스트 압축 기법 시연"""
    print("컨텍스트 압축 기법 시연")
    print("=" * 50)
    
    # 테스트 문서들 준비
    test_docs = create_test_documents()
    
    # 문서들을 청킹
    chunker = SemanticChunker()
    all_chunks = []
    
    for doc in test_docs:
        chunks = chunker.chunk_document(doc['content'], doc['metadata'])
        all_chunks.extend(chunks)
    
    # 관련성 점수 시뮬레이션 (실제로는 쿼리 기반 계산)
    for i, chunk in enumerate(all_chunks):
        chunk.relevance_score = max(0.1, 1.0 - (i * 0.1))  # 순서대로 관련성 감소
    
    print(f"원본 정보:")
    print(f"  총 청크 수: {len(all_chunks)}")
    original_tokens = sum(chunk.token_count for chunk in all_chunks)
    print(f"  총 토큰 수: {original_tokens}")
    print()
    
    # 압축 기법별 테스트
    compressor = ContextCompressor()
    target_tokens = 1500  # 목표 토큰 수
    
    compression_methods = ["truncate", "summarize", "selective"]
    
    for method in compression_methods:
        print(f"{method.upper()} 압축 결과:")
        print("-" * 30)
        
        try:
            compressed_chunks = compressor.compress_context(all_chunks.copy(), target_tokens, method)
            
            compressed_tokens = sum(chunk.token_count for chunk in compressed_chunks)
            compression_ratio = compressed_tokens / original_tokens
            
            print(f"  압축 후 청크 수: {len(compressed_chunks)}")
            print(f"  압축 후 토큰 수: {compressed_tokens}")
            print(f"  압축 비율: {compression_ratio:.2f}")
            print(f"  목표 달성: {'✓' if compressed_tokens <= target_tokens else '✗'}")
            
            # 청크 타입 분포
            chunk_types = {}
            for chunk in compressed_chunks:
                chunk_type = chunk.chunk_type
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            print(f"  청크 타입: {chunk_types}")
            
        except Exception as e:
            print(f"  압축 실패: {e}")
        
        print()

def demonstrate_context_management():
    """통합 컨텍스트 관리 시연"""
    print("통합 컨텍스트 관리 시연")
    print("=" * 50)
    
    if not validate_api_keys():
        print("API 키가 설정되지 않아 제한된 기능만 시연됩니다.")
    
    # 컨텍스트 매니저 초기화
    context_manager = ContextManager(max_tokens=2000)
    
    # 테스트 문서들
    test_docs = create_test_documents()
    test_query = "RAG 시스템에서 컨텍스트 관리가 왜 중요한가요?"
    
    print(f"테스트 쿼리: '{test_query}'")
    print(f"문서 수: {len(test_docs)}")
    print()
    
    # 청킹 전략별 테스트
    chunking_strategies = ["fixed", "semantic"]
    
    for strategy in chunking_strategies:
        print(f"{strategy.upper()} 청킹 전략:")
        print("-" * 40)
        
        try:
            # 컨텍스트 윈도우 준비
            context_window = context_manager.prepare_context(
                test_docs, test_query, strategy
            )
            
            # 통계 정보
            stats = context_manager.get_context_stats(context_window)
            
            print(f"컨텍스트 통계:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
            
            # 포맷된 컨텍스트 (일부만 표시)
            formatted_context = context_manager.format_context(context_window)
            print(f"\n포맷된 컨텍스트 (처음 300자):")
            print(formatted_context[:300] + "...")
            
        except Exception as e:
            print(f"컨텍스트 관리 실패: {e}")
        
        print("\n" + "=" * 50)

def analyze_token_usage():
    """토큰 사용량 분석"""
    print("토큰 사용량 분석")
    print("=" * 50)
    
    token_counter = TokenCounter()
    
    # 테스트 텍스트들
    test_texts = [
        "안녕하세요",
        "RAG 시스템은 검색과 생성을 결합한 AI 기술입니다.",
        "Retrieval-Augmented Generation (RAG) is an advanced AI technique.",
        "인공지능, 머신러닝, 딥러닝, 자연어처리, 컴퓨터비전",
        create_test_documents()[0]['content'][:500]
    ]
    
    print("텍스트별 토큰 분석:")
    print("-" * 30)
    
    for i, text in enumerate(test_texts, 1):
        token_count = token_counter.count_tokens(text)
        char_count = len(text)
        ratio = token_count / char_count if char_count > 0 else 0
        
        print(f"텍스트 {i}:")
        print(f"  문자 수: {char_count}")
        print(f"  토큰 수: {token_count}")
        print(f"  토큰/문자 비율: {ratio:.3f}")
        print(f"  내용: {text[:50]}{'...' if len(text) > 50 else ''}")
        print()

def main():
    """메인 실행 함수"""
    try:
        # 청킹 전략 비교
        demonstrate_chunking_strategies()
        
        # 컨텍스트 압축 기법
        demonstrate_context_compression()
        
        # 통합 컨텍스트 관리
        demonstrate_context_management()
        
        # 토큰 사용량 분석
        analyze_token_usage()
        
        print("컨텍스트 관리 실습 완료!")
        print("다음 단계에서는 RAG 웹 인터페이스를 구현합니다.")
        
    except Exception as e:
        print(f"실습 중 오류 발생: {e}")
        print("시스템 설정 및 의존성을 확인해주세요.")

if __name__ == "__main__":
    main() 