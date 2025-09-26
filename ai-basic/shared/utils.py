"""
AI 기초 실습 공통 유틸리티 함수
임베딩, 텍스트 처리, 로깅 등의 기능을 제공합니다.
"""

from openai import OpenAI
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
import time
import httpx
try:
    # lab 파일에서 실행될 때
    from shared.config import *
except ImportError:
    # shared 디렉토리에서 직접 실행될 때
    from config import *

# 로깅 설정
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# SSL 검증 비활성화 HTTP 클라이언트 생성
no_ssl_httpx = httpx.Client(verify=False)

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY, http_client=no_ssl_httpx)

class EmbeddingUtils:
    """임베딩 관련 유틸리티 클래스"""
    
    @staticmethod
    def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
        """텍스트를 임베딩 벡터로 변환"""
        try:
            text = text.replace("\n", " ")
            response = client.embeddings.create(
                input=[text],
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            raise
    
    @staticmethod
    def get_embeddings_batch(texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
        """여러 텍스트를 배치로 임베딩 처리"""
        try:
            # 텍스트 전처리
            texts = [text.replace("\n", " ") for text in texts]
            
            response = client.embeddings.create(
                input=texts,
                model=model
            )
            
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"배치 임베딩 생성 실패: {e}")
            raise

class SimilarityUtils:
    """유사도 계산 유틸리티 클래스"""
    
    @staticmethod
    def cosine_similarity_score(vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도 계산"""
        vec1_np = np.array(vec1).reshape(1, -1)
        vec2_np = np.array(vec2).reshape(1, -1)
        return cosine_similarity(vec1_np, vec2_np)[0][0]
    
    @staticmethod
    def find_most_similar(query_embedding: List[float], 
                         embeddings: List[List[float]], 
                         texts: List[str], 
                         top_k: int = 5) -> List[Dict[str, Any]]:
        """가장 유사한 텍스트 찾기"""
        similarities = []
        
        for i, embedding in enumerate(embeddings):
            similarity = SimilarityUtils.cosine_similarity_score(
                query_embedding, embedding
            )
            similarities.append({
                'index': i,
                'text': texts[i],
                'similarity': similarity
            })
        
        # 유사도 기준 정렬
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]

class TextUtils:
    """텍스트 처리 유틸리티 클래스"""
    
    @staticmethod
    def count_tokens(text: str, model: str = "cl100k_base") -> int:
        """토큰 수 계산"""
        try:
            encoding = tiktoken.get_encoding(model)
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"토큰 계산 실패, 대략적으로 계산: {e}")
            return len(text.split()) * 1.3  # 대략적 계산
    
    @staticmethod
    def chunk_text(text: str, 
                   chunk_size: int = DEFAULT_CHUNK_SIZE, 
                   overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
        """텍스트를 청크로 분할"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    @staticmethod
    def clean_text(text: str) -> str:
        """텍스트 정리"""
        # 기본적인 정리
        text = text.strip()
        text = ' '.join(text.split())  # 연속 공백 제거
        return text

class ChatUtils:
    """채팅 관련 유틸리티 클래스"""
    
    @staticmethod
    def get_chat_response(prompt: Union[str, List[Dict[str, str]]], 
                         model: str = CHAT_MODEL,
                         temperature: float = TEMPERATURE) -> str:
        """ChatGPT 응답 생성"""
        try:
            # 문자열인 경우 메시지 형식으로 변환
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = prompt
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=MAX_TOKENS
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"채팅 응답 생성 실패: {e}")
            raise

class PerformanceUtils:
    """성능 측정 유틸리티 클래스"""
    
    @staticmethod
    def timer(func):
        """함수 실행 시간 측정 데코레이터"""
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            logger.info(f"{func.__name__} 실행 시간: {end - start:.2f}초")
            return result
        return wrapper
    
    @staticmethod
    def measure_performance(func, *args, **kwargs):
        """함수 성능 측정"""
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        
        return {
            'result': result,
            'execution_time': end - start,
            'function_name': func.__name__
        }

def print_progress(current: int, total: int, prefix: str = "Progress"):
    """진행률 출력"""
    percent = (current / total) * 100
    bar_length = 30
    filled_length = int(bar_length * current // total)
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    print(f'\r{prefix}: |{bar}| {percent:.1f}% ({current}/{total})', end='')
    if current == total:
        print()  # 완료 시 새 줄

def format_results(results: List[Dict[str, Any]]) -> str:
    """검색 결과 포맷팅"""
    formatted = "검색 결과:\n\n"
    
    for i, result in enumerate(results, 1):
        similarity = result.get('similarity', 0)
        text = result.get('text', '')[:200]  # 처음 200자만
        
        formatted += f"{i}. 유사도: {similarity:.3f}\n"
        formatted += f"   내용: {text}...\n\n"
    
    return formatted

if __name__ == "__main__":
    # 간단한 테스트
    print("유틸리티 테스트")
    
    # 토큰 계산 테스트
    sample_text = "안녕하세요! 이것은 테스트 텍스트입니다."
    token_count = TextUtils.count_tokens(sample_text)
    print(f"토큰 수: {token_count}")
    
    # 텍스트 청킹 테스트
    chunks = TextUtils.chunk_text(sample_text * 10, chunk_size=50)
    print(f"청크 수: {len(chunks)}")
    
    print("테스트 완료!") 