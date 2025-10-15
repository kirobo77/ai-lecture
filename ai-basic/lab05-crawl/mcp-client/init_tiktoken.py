"""tiktoken 캐시 초기화 스크립트 (사내망 환경용)"""
import ssl
import os

# 1. SSL 검증 비활성화
ssl._create_default_https_context = ssl._create_unverified_context

# 2. urllib3 비활성화
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 3. requests 라이브러리 SSL 검증 비활성화
import requests
from functools import wraps

# 원본 함수 백업
original_request = requests.Session.request

# SSL 검증을 비활성화하는 래퍼 함수
@wraps(original_request)
def no_ssl_verification(self, method, url, **kwargs):
    kwargs['verify'] = False
    return original_request(self, method, url, **kwargs)

# requests.Session.request 패치
requests.Session.request = no_ssl_verification

# 4. tiktoken 초기화
try:
    import tiktoken
    print("tiktoken 인코딩 파일 다운로드 중...")
    
    enc1 = tiktoken.get_encoding("cl100k_base")
    print("✓ cl100k_base 다운로드 완료")
    
    enc2 = tiktoken.encoding_for_model("gpt-4o")
    print("✓ gpt-4o 인코딩 다운로드 완료")
    
    print("\n✅ 모든 인코딩 파일이 성공적으로 캐시되었습니다!")
    print(f"캐시 위치: {os.path.expanduser('~/.cache/tiktoken')}")
    
except Exception as e:
    print(f"❌ 오류 발생: {e}")
    print("\n대안: 외부 네트워크에서 캐시 파일을 생성하여 복사하세요.")