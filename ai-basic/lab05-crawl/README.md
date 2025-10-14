# Lab 5: 지능형 웹 크롤링 및 MCP 통합 시스템

## 연계 이론
- **4.1**: MCP(Model Context Protocol) 아키텍처와 표준화된 AI 도구 연동
- **4.2**: 웹 크롤링과 HTML 콘텐츠 구조화 기법
- **4.3**: 마이크로서비스 아키텍처와 HTTP 기반 통신

## 학습 목표
1. **MCP 프로토콜**: 표준화된 AI 도구 통신 프로토콜 이해 및 구현
2. **지능형 크롤링**: HTML 콘텐츠를 구조화된 데이터로 변환하는 파이프라인 구축
3. **마이크로서비스**: 클라이언트-서버 분리 아키텍처 설계 및 구현
4. **LLM 통합**: OpenAI API와 MCP 도구를 연동한 지능형 처리 시스템

## 시스템 아키텍처

```
사용자 요청 (REST API)
        ↓
┌─────────────────────────────────────┐
│        MCP Client (FastAPI)         │  Port: 8000
│  ┌─────────────┬─────────────────┐  │
│  │ ARI Service │   LLM Service   │  │
│  │ (HTML 처리)  │   (OpenAI)      │  │
│  └─────────────┴─────────────────┘  │
│           MCP Service               │
└─────────────────────────────────────┘
        ↓ HTTP 통신
┌─────────────────────────────────────┐
│       MCP Server (FastMCP)          │  Port: 4200
│  ┌─────────────┬─────────────────┐  │
│  │ HTML Parser │ Markdown Conv.  │  │
│  │(BeautifulSoup)│(markdownify)  │  │
│  └─────────────┴─────────────────┘  │
│        JSON 구조화 변환                │
└─────────────────────────────────────┘
```

## 디렉토리 구조

```
lab05-crawl/
├── README.md                    # 이 파일
├── requirements.txt             # 통합 패키지 의존성
├── mcp-client/                  # MCP 클라이언트 (FastAPI 서버)
│   ├── main.py                 # 메인 애플리케이션 진입점
│   └── app/                    # 애플리케이션 코드
│       ├── config.py           # 설정 관리
│       ├── models.py           # 데이터 모델 정의
│       ├── routers/            # API 라우터
│       │   └── api.py          # REST API 엔드포인트
│       ├── application/        # 애플리케이션 서비스
│       │   └── ari/
│       │       └── ari_service.py  # ARI HTML 처리 서비스
│       ├── infrastructure/     # 인프라스트럭처 레이어
│       │   ├── llm/
│       │   │   └── llm_service.py  # OpenAI LLM 서비스
│       │   └── mcp/
│       │       └── mcp_service.py  # MCP 클라이언트 서비스
│       ├── core/               # 핵심 유틸리티
│       │   └── logging.py      # 로깅 설정
│       ├── exceptions/         # 예외 처리
│       │   └── base.py         # 기본 예외 클래스
│       └── utils/              # 유틸리티
│           └── schema_converter.py  # 스키마 변환 도구
└── mcp-server/                  # MCP 서버 (FastMCP)
    └── server.py               # MCP 도구 서버
```

## 실습 단계

### Step 1: 환경 설정 및 의존성 설치 (15분)

#### 1. uv 설치 (필요한 경우)
```bash
# uv가 설치되어 있지 않은 경우
curl -LsSf https://astral.sh/uv/install.sh | sh

# 또는 pip로 설치
pip install uv
```

#### 2. 가상환경 생성 및 활성화
```bash
# lab05 디렉토리로 이동
cd lab05-crawl

# uv로 가상환경 생성
uv venv

# 가상환경 활성화
# Linux/Mac:
source .venv/bin/activate

# Windows:
# .venv\Scripts\activate
```

#### 3. 패키지 설치
```bash
# 가상환경이 활성화된 상태에서 패키지 설치
uv pip install -r requirements.txt

# 또는 프로젝트 루트의 requirements.txt 사용
uv pip install -r ../requirements.txt
```

#### 4. 환경 변수 설정
```bash
# lab05 디렉토리에 .env 파일 생성
cd lab05-crawl
cat > .env << EOF
# OpenAI API 키 (필수)
OPENAI_API_KEY=your_actual_openai_api_key_here

# 서버 설정
HOST=0.0.0.0
PORT=8000
DEBUG=false
LOG_LEVEL=info

# MCP 서버 설정
MCP_SERVER_URL=http://127.0.0.1:4200/my-custom-path/
MCP_CONNECTION_TIMEOUT=30
MCP_RETRY_ATTEMPTS=3

# 모델 설정
OPENAI_MODEL=gpt-4o

# 선택사항: 벡터 DB 및 검색 엔진 설정
# QDRANT_HOST=
# OPENSEARCH_HOST=
# DATABASE_URL=
EOF
```

**중요**: `.env` 파일은 lab05-crawl 디렉토리에 위치하며, 이미 `.gitignore`에 포함되어 있어 Git에 커밋되지 않습니다.

#### 5. 설치 확인
```bash
# 핵심 패키지 확인
python -c "import fastmcp; print(f'FastMCP: {fastmcp.__version__}')"
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
python -c "import openai; print(f'OpenAI: {openai.__version__}')"
python -c "import bs4; print('BeautifulSoup4: OK')"
```

### Step 2: MCP 서버 실행 (10분)

```bash
# MCP 서버 시작
cd lab05-crawl/mcp-server
python server.py

# 서버 상태 확인 (별도 터미널)
curl http://localhost:4200/my-custom-path/health
```

### Step 3: MCP 클라이언트 실행 (10분)

```bash
# MCP 클라이언트 시작 (새 터미널)
cd lab05-crawl/mcp-client
python main.py

# 클라이언트 상태 확인 (별도 터미널)
curl http://localhost:8000/health
```

### Step 4: API 테스트 및 기능 확인 (15분)

#### Swagger UI를 통한 테스트 (권장)
```bash
# MCP 클라이언트가 실행된 상태에서 브라우저로 접속
http://localhost:8000/docs
```

**Swagger UI에서 테스트하기:**
1. 브라우저에서 `http://localhost:8000/docs` 접속
2. `/api/llm/query-with-files` 엔드포인트 클릭 (일반 응답용)
   - 또는 `/api/llm/query-with-files-download` 클릭 (대용량 파일 다운로드용)
3. "Try it out" 버튼 클릭
4. `question` 필드에 질문 입력 (예: "이 HTML 파일의 내용을 요약해줘")
5. `files` 필드에서 HTML 파일 업로드
6. "Execute" 버튼 클릭하여 테스트

**엔드포인트 선택 가이드:**
- **일반 응답**: `/api/llm/query-with-files` - 작은 HTML 파일용 (JSON 응답)
- **파일 다운로드**: `/api/llm/query-with-files-download` - 대용량 HTML 파일용 (JSON 파일 다운로드)

**예상 응답 형태:**
```
## [25 년 9 월 3 주차 - 완료 ] AI 사업개발 **AI** 사업개발

구분 전주 실적 **(2025. 9. 8 ~ 2025. 9. 12 )**

## **[K-** **intelligence** AI Agent 구 축 프로젝트 ]

주요

업무

현황

[image]

## 하반기 에이전트 진행 시 개 선 요청 사항 및 개선 방향

내용 유관부서

개발총괄 **PM** 역할 필요

- 예를들면, Agent 개발팀
과 앱 개발팀이 분리되
어 있는데, 총괄 개발
**PM** 역할이 없어 커뮤니
케이션 및 의사결정이
어려웠음

**[** 개발 프로젝트 관리 **]**

**WBS** 기반으로 개발 프로젝트
관리가 되어야 함

... (전체 HTML 콘텐츠가 깔끔하게 정리되어 표시됨)
```

**특징:**
- HTML의 모든 텍스트 콘텐츠가 표시됨
- 불필요한 HTML 태그는 제거됨
- 섹션별로 깔끔하게 정리됨
- 테이블 데이터도 마크다운 형식으로 변환됨

#### curl을 통한 테스트 (선택사항)
```bash
# 일반 응답 (작은 파일용)
curl -X POST "http://localhost:8000/api/llm/query-with-files" \
  -H "Content-Type: multipart/form-data" \
  -F "question=이 HTML 파일의 내용을 요약해줘" \
  -F "files=@sample.html"

# 파일 다운로드 (대용량 파일용)
curl -X POST "http://localhost:8000/api/llm/query-with-files-download" \
  -H "Content-Type: multipart/form-data" \
  -F "question=이 HTML 파일의 모든 내용을 추출해줘" \
  -F "files=@large_sample.html" \
  --output "extracted_content.json"
```

## 핵심 기능

### 1. MCP 프로토콜 구현
- **표준화된 통신**: JSON-RPC 2.0 기반 도구 통신
- **동적 도구 발견**: 런타임에 사용 가능한 도구 자동 감지
- **타입 안전성**: Pydantic 기반 스키마 검증

### 2. 지능형 HTML 처리
- **구조화 파싱**: BeautifulSoup을 활용한 HTML 구조 분석
- **마크다운 변환**: HTML을 의미 보존 마크다운으로 변환
- **JSON 구조화**: 마크다운을 구조화된 JSON 데이터로 변환

### 3. LLM 통합 시스템
- **의도 분류**: 사용자 질문을 분석하여 적절한 도구 선택
- **도구 체이닝**: 여러 MCP 도구를 연계한 복합 작업 수행
- **컨텍스트 관리**: 대화 컨텍스트와 파일 컨텍스트 통합 관리

### 4. 마이크로서비스 아키텍처
- **서비스 분리**: 클라이언트와 서버의 명확한 역할 분담
- **확장성**: 독립적인 서비스 스케일링 가능
- **장애 격리**: 서비스 간 장애 전파 방지

## 사용 시나리오

### 시나리오 1: Confluence HTML 문서 처리
**목적**: 기업 내부 Confluence 페이지를 RAG 시스템용 데이터로 변환

```bash
# 1. HTML 파일 업로드 및 구조화
curl -X POST "http://localhost:8000/api/ari/crawl" \
  -F "files=@confluence_page.html"

# 응답: 구조화된 JSON 데이터
{
  "taskId": "ari_20241014_143022",
  "status": "COMPLETED",
  "result": [
    {
      "title": "프로젝트 개발 가이드",
      "breadcrumbs": ["홈", "개발", "가이드"],
      "content": {
        "contents": [
          {"id": 1, "type": "text", "title": "개요", "data": "..."},
          {"id": 2, "type": "table", "headers": ["항목", "설명"], "rows": [...]}
        ]
      }
    }
  ]
}
```

### 시나리오 2: 지능형 문서 분석
**목적**: LLM을 활용한 문서 내용 분석 및 요약

```bash
# 작은 HTML 파일 처리 (일반 응답)
curl -X POST "http://localhost:8000/api/llm/query-with-files" \
  -F "question=이 문서의 주요 내용을 3줄로 요약해줘" \
  -F "files=@technical_doc.html"

# 응답: LLM이 생성한 요약
{
  "success": true,
  "answer": "1. 이 문서는 마이크로서비스 아키텍처 설계 가이드입니다.\n2. 서비스 분리 원칙과 API 설계 방법론을 다룹니다.\n3. Docker와 Kubernetes를 활용한 배포 전략을 제시합니다.",
  "tools_used": ["ari_html_to_markdown", "ari_markdown_to_json"]
}

# 대용량 HTML 파일 처리 (파일 다운로드)
curl -X POST "http://localhost:8000/api/llm/query-with-files-download" \
  -F "question=이 HTML 파일의 모든 내용을 추출해줘" \
  -F "files=@large_document.html" \
  --output "extracted_content_20241014.json"

# 다운로드된 JSON 파일 내용 예시:
{
  "success": true,
  "question": "이 HTML 파일의 모든 내용을 추출해줘",
  "file_info": [{"filename": "large_document.html", "content_length": 298434}],
  "answer": "## 문서 제목\n\n전체 HTML 콘텐츠가 깔끔하게 정리되어 표시됨...",
  "content_length": 7381,
  "processed_at": "2025-10-14T10:05:51.121315",
  "download_type": "json_file"
}
```

### 시나리오 3: 시스템 모니터링
**목적**: MCP 도구를 활용한 시스템 상태 확인

```bash
# 시스템 상태 질문
curl -X POST "http://localhost:8000/api/llm/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "현재 시스템 상태를 확인해줘"}'

# 응답: 시스템 상태 보고
{
  "success": true,
  "answer": "시스템이 정상적으로 작동 중입니다.\n- MCP 서버: 연결됨\n- 사용 가능한 도구: 4개\n- BeautifulSoup: 정상",
  "tools_used": ["health_check"]
}
```

## 실습 결과물

1. **MCP 기반 크롤링 시스템**: 표준화된 프로토콜로 구현된 웹 크롤링 파이프라인
2. **지능형 문서 처리기**: HTML을 구조화된 데이터로 변환하는 자동화 시스템
3. **LLM 통합 플랫폼**: 다양한 AI 도구를 연동한 지능형 처리 시스템
4. **HTTP 기반 마이크로서비스**: 클라이언트-서버 분리 아키텍처

## 체크포인트

### 기본 이해도 확인
- [ ] MCP 프로토콜의 목적과 장점을 설명할 수 있는가?
- [ ] 마이크로서비스 아키텍처의 구성 요소를 이해했는가?
- [ ] HTML 구조화 파이프라인의 각 단계를 파악했는가?

### 실습 완료도 확인
- [ ] MCP 서버와 클라이언트를 성공적으로 실행할 수 있는가?
- [ ] HTML 파일을 마크다운으로 변환할 수 있는가?
- [ ] 구조화된 JSON 데이터를 생성할 수 있는가?
- [ ] LLM과 MCP 도구를 연동하여 질문에 답할 수 있는가?
- [ ] HTTP API를 통해 서비스 간 통신을 확인할 수 있는가?

## 도전 과제

### 기본 도전
1. **새로운 MCP 도구 추가**: PDF 처리나 이미지 분석 도구 개발
2. **배치 처리 시스템**: 대량 HTML 파일 동시 처리 기능 구현
3. **웹 UI 개발**: Streamlit이나 React를 활용한 사용자 인터페이스 구축

### 고급 도전
1. **분산 처리**: 여러 MCP 서버를 활용한 로드 밸런싱 구현
2. **실시간 모니터링**: 시스템 메트릭 수집 및 대시보드 구축
3. **성능 최적화**: 캐싱 및 비동기 처리 개선

## 주요 MCP 도구

### 서버 측 도구 (mcp-server)
- **`health_check`**: 서버 상태 및 의존성 확인
- **`ari_parse_html`**: HTML 파싱 및 메타데이터 추출
- **`ari_html_to_markdown`**: HTML을 마크다운으로 변환
- **`ari_markdown_to_json`**: 마크다운을 구조화된 JSON으로 변환

### 클라이언트 측 서비스
- **ARI Service**: HTML 파일 처리 및 변환 관리
- **LLM Service**: OpenAI API 연동 및 의도 분류
- **MCP Service**: MCP 서버와의 통신 및 도구 관리

## 기술 스택

### 백엔드 프레임워크
- **FastAPI**: 고성능 비동기 웹 프레임워크
- **FastMCP**: MCP 프로토콜 구현 라이브러리
- **Pydantic**: 데이터 검증 및 직렬화

### AI 및 처리 라이브러리
- **OpenAI**: GPT 모델 API 연동
- **BeautifulSoup4**: HTML 파싱 및 처리
- **markdownify**: HTML to Markdown 변환
- **pymupdf4llm**: 고급 문서 변환

### 인프라스트럭처
- **uvicorn**: ASGI 서버
- **asyncio**: 비동기 프로그래밍
- **HTTP**: 서비스 간 통신 프로토콜

## 문제 해결

### 일반적인 오류

#### 1. MCP 연결 실패
```bash
# MCP 서버 상태 확인
curl http://localhost:4200/my-custom-path/health

# 포트 충돌 해결
lsof -ti:4200 | xargs kill -9
lsof -ti:8000 | xargs kill -9
```

#### 2. 대용량 HTML 파일 처리 오류
```bash
# 응답이 잘리는 경우 - 다운로드 엔드포인트 사용
curl -X POST "http://localhost:8000/api/llm/query-with-files-download" \
  -F "question=이 HTML 파일의 모든 내용을 추출해줘" \
  -F "files=@large_file.html" \
  --output "result.json"

# 파일 크기 확인
ls -la result.json

# JSON 파일 내용 확인
cat result.json | jq '.content_length'
```

**해결 방법:**
- 작은 파일(~100KB): `/api/llm/query-with-files` 사용
- 대용량 파일(>100KB): `/api/llm/query-with-files-download` 사용
- 응답이 잘리면 항상 다운로드 엔드포인트 사용

#### 3. 패키지 의존성 오류
```bash
# uv 가상환경 재생성
cd lab05-crawl
rm -rf .venv  # 기존 가상환경 삭제
uv venv       # 새 가상환경 생성

# 가상환경 활성화
source .venv/bin/activate  # Linux/Mac
# 또는
# .venv\Scripts\activate   # Windows

# uv로 패키지 재설치
uv pip install -r requirements.txt

# 또는 캐시 클리어 후 재설치
uv cache clean
uv pip install -r requirements.txt
```

#### 3. 서비스 통신 오류
```bash
# 네트워크 연결 확인
curl -v http://localhost:4200/my-custom-path/health
curl -v http://localhost:8000/api/health

# 방화벽 설정 확인
netstat -tlnp | grep :4200
netstat -tlnp | grep :8000
```

### 성능 최적화

#### 1. 메모리 사용량 최적화
```python
# HTML 처리 시 청킹 적용
async def process_large_html(html_content: str, chunk_size: int = 1000000):
    if len(html_content) > chunk_size:
        # 청크 단위로 처리
        pass
```

#### 2. 비동기 처리 최적화
```python
# 병렬 파일 처리
import asyncio
async def process_multiple_files(files):
    tasks = [process_single_file(file) for file in files]
    results = await asyncio.gather(*tasks)
    return results
```

## 실무 적용 사례

### 1. 기업 문서 관리 시스템
- **Confluence/SharePoint** 페이지 자동 수집 및 구조화
- **RAG 시스템** 용 지식베이스 구축
- **검색 엔진** 최적화를 위한 메타데이터 추출

### 2. 콘텐츠 마이그레이션
- **레거시 시스템** 문서를 현대적 형식으로 변환
- **다국어 콘텐츠** 일괄 처리 및 번역 준비
- **품질 관리** 자동화를 통한 일관성 확보

### 3. 지능형 고객 지원
- **FAQ 자동 생성**: 기존 문서에서 Q&A 추출
- **챗봇 학습 데이터**: 구조화된 대화 데이터 생성
- **실시간 문서 분석**: 고객 문의와 관련 문서 자동 매칭

---

**참고**: 이 실습은 실제 기업 환경에서 사용되는 MCP 프로토콜과 마이크로서비스 아키텍처를 기반으로 설계되었습니다. 완성된 시스템은 프로덕션 환경에서도 활용 가능한 수준의 안정성과 확장성을 제공합니다.
