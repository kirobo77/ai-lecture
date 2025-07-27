# Lab 3: Basic RAG System

## 연계 이론
- **3.1**: RAG 아키텍처와 구성 요소
- **3.2**: 문서 검색과 생성 모델 통합

## 학습 목표
1. RAG(Retrieval-Augmented Generation) 시스템의 기본 아키텍처 이해
2. 검색(Retrieval) 컴포넌트 구현 및 최적화
3. 생성(Generation) 컴포넌트와 프롬프트 엔지니어링
4. RAG 파이프라인 통합 및 성능 측정
5. 실제 사용 가능한 웹 인터페이스 구현

## 실습 단계

### Step 1: 기본 RAG 파이프라인 (45분)
**파일: `basic_rag.py`**
- RAG 시스템의 핵심 컴포넌트 구현
- 문서 검색 → 컨텍스트 구성 → 답변 생성 파이프라인
- 기본 프롬프트 템플릿 설계
- 검색 결과와 생성 답변의 품질 평가

### Step 2: 고급 검색 기법 (60분)  
**파일: `advanced_retrieval.py`**
- 다양한 검색 전략 (유사도 기반, 하이브리드 검색)
- 검색 결과 재순위(Re-ranking) 기법
- 쿼리 확장 및 의미적 검색 최적화
- 검색 성능 비교 및 분석

### Step 3: 컨텍스트 관리 (45분)
**파일: `context_management.py`**
- 효율적인 컨텍스트 윈도우 관리
- 긴 문서 처리 및 청킹 전략
- 관련성 높은 정보 선별 및 요약
- 토큰 제한 내에서 최적 컨텍스트 구성

### Step 4: RAG 웹 인터페이스 (60분)
**파일: `rag_web_app.py`**
- Streamlit을 활용한 사용자 친화적 인터페이스
- 실시간 문서 업로드 및 검색
- 검색 과정 시각화 및 투명성 제공
- 답변 품질 피드백 시스템

**웹 앱 실행:**
```bash
# 권장 실행 방법 (환경 문제 없음)
python -m streamlit run rag_web_app.py

# 대안 실행 방법 (환경에 따라 문제 발생 가능)
streamlit run rag_web_app.py

# 잘못된 실행 방법 (경고 발생)
python rag_web_app.py  # ❌ 사용하지 마세요
```

**환경 문제 해결:**
- `python -m streamlit`은 현재 Python 환경의 streamlit을 사용
- `streamlit` 명령어는 시스템 PATH의 streamlit을 사용 (환경 불일치 가능)
- conda 가상환경 사용 시 `python -m streamlit` 권장

## 사전 준비사항

### 환경 설정

#### 1. 필수 패키지 설치
```bash
# 프로젝트 루트에서 모든 패키지 한 번에 설치 (권장)
pip install -r requirements.txt

# 또는 Lab 3 필수 패키지만 설치
pip install chromadb streamlit langchain langchain-openai langchain-community tiktoken rank_bm25 plotly

# 개별 패키지 설명
pip install chromadb          # 벡터 데이터베이스
pip install streamlit         # 웹 인터페이스
pip install langchain langchain-openai langchain-community  # LangChain 프레임워크
pip install tiktoken          # 토큰 계산
pip install rank_bm25         # BM25 검색
pip install plotly           # 데이터 시각화
```

#### 2. 설치 확인
```bash
# 핵심 패키지 확인
python -c "import chromadb; print(f'ChromaDB: {chromadb.__version__}')"
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
python -c "import langchain; print(f'LangChain: {langchain.__version__}')"
python -c "import plotly; print(f'Plotly: {plotly.__version__}')"

# Streamlit 실행 확인 (권장 방법)
python -m streamlit --version

# 환경 일치 확인
python -c "
import sys, streamlit, chromadb
print(f'Python: {sys.executable}')
print(f'Streamlit: {streamlit.__version__}')
print(f'ChromaDB: {chromadb.__version__}')
print('모든 패키지가 같은 환경에 설치됨!')
"
```

### 구현 방식 비교

Lab 3에서는 **두 가지 구현 방식**을 제공합니다:

#### 순수 구현 (Pure Implementation)
- **파일**: `basic_rag.py`, `advanced_retrieval.py`, `context_management.py`
- **특징**: RAG의 내부 동작 원리를 직접 구현
- **장점**: 
  - 모든 컴포넌트의 작동 방식 이해 가능
  - 세밀한 커스터마이징 가능
  - 디버깅과 최적화에 유리
- **단점**: 
  - 코드 복잡도 높음
  - 개발 시간 많이 소요

#### LangChain 구현 (LangChain Implementation)  
- **파일**: `basic_rag_langchain.py`, `advanced_retrieval_langchain.py`
- **특징**: LangChain 프레임워크의 고급 컴포넌트 활용
- **장점**:
  - 간결하고 효율적인 코드
  - 검증된 컴포넌트 사용
  - 빠른 프로토타이핑 가능
  - 실무 환경과 유사한 접근법
- **단점**:
  - 프레임워크 의존성
  - 내부 동작 추상화로 학습 효과 제한

#### 권장 학습 순서
1. **순수 구현**으로 RAG 기본 원리 이해
2. **LangChain 구현**으로 실무 접근법 체험
3. 두 방식 비교를 통한 장단점 파악

### 디렉토리 구조
```
lab03-rag/
├── README.md
├── basic_rag.py           # Step 1: 기본 RAG 파이프라인
├── advanced_retrieval.py  # Step 2: 고급 검색 기법
├── context_management.py  # Step 3: 컨텍스트 관리
├── rag_web_app.py        # Step 4: 웹 인터페이스
├── basic_rag_langchain.py         # Step 1 (LangChain 버전)
├── advanced_retrieval_langchain.py # Step 2 (LangChain 버전)
├── sample_documents/     # 테스트용 문서 컬렉션
│   ├── ai_overview.txt
│   ├── ml_basics.txt
│   └── dl_concepts.txt
└── templates/            # 프롬프트 템플릿
    ├── basic_qa.txt
    ├── detailed_explanation.txt
    └── source_citation.txt
```

## 핵심 개념

### RAG 시스템 아키텍처
```
사용자 질문 → 검색 쿼리 변환 → 벡터 검색 → 관련 문서 추출 → 컨텍스트 구성 → LLM 답변 생성 → 최종 응답
```

### 주요 컴포넌트
1. **Query Processor**: 사용자 질문을 검색 쿼리로 변환
2. **Retriever**: 벡터 데이터베이스에서 관련 문서 검색
3. **Context Manager**: 검색된 문서를 효율적으로 구성
4. **Generator**: LLM을 활용한 최종 답변 생성
5. **Evaluator**: 답변 품질 평가 및 피드백

### 성능 지표
- **검색 정확도**: 관련 문서 검색 성공률
- **답변 품질**: 정확성, 완전성, 일관성
- **응답 시간**: 전체 파이프라인 처리 속도
- **토큰 효율성**: 컨텍스트 토큰 사용량 최적화

## 예상 학습 성과

### 완료 후 습득 역량
1. **RAG 시스템 설계**: 전체 아키텍처 이해 및 구현
2. **검색 최적화**: 다양한 검색 기법 활용
3. **프롬프트 엔지니어링**: 효과적인 프롬프트 작성
4. **성능 튜닝**: 시스템 성능 측정 및 개선
5. **사용자 인터페이스**: 실용적인 웹 애플리케이션 개발

### 실무 적용 가능 영역
- 고객 지원 챗봇
- 기업 내부 지식 검색 시스템
- 기술 문서 QA 시스템
- 교육용 학습 도우미
- 연구 논문 분석 도구

## 문제 해결 가이드

### 일반적인 오류
1. **검색 결과 부족**: 임베딩 품질 또는 청킹 전략 점검
2. **답변 품질 저하**: 프롬프트 템플릿 및 컨텍스트 구성 개선
3. **응답 시간 지연**: 검색 범위 조정 및 캐싱 적용
4. **토큰 제한 초과**: 컨텍스트 요약 또는 선별 로직 강화

### 성능 최적화 팁
- 적절한 청크 크기 설정 (300-800 토큰)
- 검색 결과 개수 조절 (3-7개 권장)
- 프롬프트 길이 최적화
- 캐싱을 통한 반복 검색 최소화

## 다음 단계 미리보기
Lab 4에서는 RAG 성능 최적화, 고급 검색 알고리즘, A/B 테스트를 통한 시스템 개선 방법을 학습합니다. 