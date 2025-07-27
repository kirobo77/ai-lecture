"""
AI 기초 실습 공통 설정 파일
API 키와 기본 설정을 관리합니다.
"""

import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# API 키 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 임베딩 모델 설정
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIMENSION = 1536

# LLM 모델 설정
CHAT_MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.1

# Chroma 설정 (로컬 벡터 데이터베이스)
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
CHROMA_COLLECTION_NAME = "ai-basic-practice"

# 공통 설정
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
MAX_TOKENS = 4000

# 로깅 설정
LOG_LEVEL = "INFO"

def validate_api_keys():
    """API 키 유효성 검사"""
    missing_keys = []
    
    if not OPENAI_API_KEY:
        missing_keys.append("OPENAI_API_KEY")
    
    if missing_keys:
        print(f"누락된 API 키: {', '.join(missing_keys)}")
        print(".env 파일에 다음 키들을 추가해주세요:")
        for key in missing_keys:
            print(f"   {key}=your_key_here")
        return False
    
    print("모든 API 키가 설정되었습니다!")
    return True

if __name__ == "__main__":
    validate_api_keys() 