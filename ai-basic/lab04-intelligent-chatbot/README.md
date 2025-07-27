# Lab 4: 지능형 API 라우팅 챗봇 시스템

## 프로젝트 개요

**RAG + Multi-Agent + MCP를 통합한 실무급 지능형 챗봇 시스템**

사용자의 자연어 입력을 분석하여 적절한 API를 선택하고 호출하는 지능형 챗봇을 구축합니다.
외부 API 의존성 없이 자체 Mock Backend로 실제 서비스 환경을 시뮬레이션합니다.

## 학습 목표

- **RAG 시스템**: 과거 대화 패턴과 지식베이스를 활용한 컨텍스트 이해
- **Multi-Agent**: 전문화된 에이전트들의 협력과 의사결정
- **MCP 프로토콜**: 표준화된 API 통신과 도구 연동
- **실무 적용**: 기업 환경에서 활용 가능한 챗봇 아키텍처

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                사용자 웹 인터페이스 (Streamlit)                │
├─────────────────────────────────────────────────────────────┤
│      Intent Classification Agent (RAG 기반 의도 분석)         │
├─────────────────────────────────────────────────────────────┤
│     Multi-Agent 협력 시스템                                  │
│  ├─ Weather Agent    ├─ Calendar Agent                      │
│  ├─ File Agent       ├─ Notification Agent                  │
├─────────────────────────────────────────────────────────────┤
│     MCP 통신 레이어                                          │
├─────────────────────────────────────────────────────────────┤
│     Mock API Backend                                       │
│  ├─ Weather API      ├─ Calendar API                        │
│  ├─ File Manager API ├─ Notification API                    │
└─────────────────────────────────────────────────────────────┘
```

## 디렉토리 구조

```
lab04-intelligent-chatbot/
├── README.md                    # 이 파일
├── main_chatbot.py             # 메인 챗봇 시스템
├── mock_apis/                   # Mock API 서버들
│   ├── weather_api.py          # 날씨 정보 API
│   ├── calendar_api.py         # 일정 관리 API
│   ├── file_manager_api.py     # 파일 관리 API
│   ├── notification_api.py     # 알림/메시징 API
│   └── database_api.py         # 데이터 조회 API
├── agents/                      # 전문 에이전트들
│   ├── intent_classifier.py    # 의도 분석 에이전트
│   ├── weather_agent.py        # 날씨 전문 에이전트
│   ├── calendar_agent.py       # 일정 관리 에이전트
│   ├── file_agent.py           # 파일 관리 에이전트
│   └── notification_agent.py   # 알림 전문 에이전트
├── mcp_layer/                   # MCP 통신 레이어
│   ├── mcp_client.py           # MCP 클라이언트
│   └── api_connectors.py       # API 연결 관리
├── rag_system/                  # RAG 엔진
│   ├── knowledge_base.py       # 지식베이스 관리
│   └── context_manager.py      # 컨텍스트 관리
├── web_interface/               # 웹 인터페이스
│   └── chatbot_ui.py           # Streamlit 챗봇 UI
└── data/                        # 데이터 파일들
    ├── weather_data.json       # 날씨 더미 데이터
    ├── calendar_events.json    # 일정 더미 데이터
    ├── documents/              # 문서 샘플들
    └── conversation_history.db # 대화 기록 DB
```

## 실습 단계

### Step 1: Mock API 서버 구축 (30분)

#### 방법 1: 자동 실행 스크립트 사용 (권장)
```bash
# 모든 API 서버를 한 번에 실행
python start_apis.py
```

#### 방법 2: 개별 실행
```bash
# 각 API 서버를 개별적으로 실행
python mock_apis/weather_api.py &
python mock_apis/calendar_api.py &
python mock_apis/file_manager_api.py &
python mock_apis/notification_api.py &
```

### Step 2: Multi-Agent 시스템 (45분)
```bash
# Agent 테스트
python agents/intent_classifier.py
python agents/weather_agent.py
```

### Step 3: MCP 통합 레이어 (30분)
```bash
# MCP 통신 테스트
python mcp_layer/mcp_client.py
```

### Step 4: 통합 챗봇 실행 (15분)

#### 웹 인터페이스 실행
```bash
# 권장 실행 방법 (환경 문제 없음)
python -m streamlit run web_interface/chatbot_ui.py

# 대안 실행 방법
streamlit run web_interface/chatbot_ui.py
```

#### 환경 문제 해결
- `python -m streamlit`은 현재 Python 환경의 streamlit을 사용
- `streamlit` 명령어는 시스템 PATH의 streamlit을 사용 (환경 불일치 가능)
- conda 가상환경 사용 시 `python -m streamlit` 권장

## 사용 시나리오

### 시나리오 1: 날씨 정보 조회
**사용자**: "오늘 서울 날씨 어때?"
- **Intent Agent**: 날씨 정보 요청으로 분류
- **Weather Agent**: Weather API 호출
- **응답**: "서울은 현재 24도, 맑음입니다."

### 시나리오 2: 일정 관리
**사용자**: "내일 회의 일정 확인하고 팀에게 알려줘"
- **Intent Agent**: 일정 조회 + 알림 요청으로 분류
- **Calendar Agent**: 일정 조회
- **Notification Agent**: 팀 알림 발송
- **응답**: "내일 3개 회의가 있습니다. 팀에게 알림을 보냈습니다."

### 시나리오 3: 문서 검색
**사용자**: "프로젝트 계획서 찾아서 요약해줘"
- **Intent Agent**: 파일 검색 + 요약 요청으로 분류
- **File Agent**: 문서 검색
- **RAG System**: 문서 내용 분석 및 요약
- **응답**: "프로젝트 계획서를 찾았습니다. 주요 내용: ..."

## 환경 설정

### 1. 패키지 설치
```bash
# 프로젝트 루트에서 모든 패키지 한 번에 설치
cd .. # lab04 폴더에서 루트로 이동
pip install -r requirements.txt

# 또는 루트에서 직접 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정
```bash
# .env 파일 생성 (프로젝트 루트)
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. 데이터 초기화
```bash
# Mock 데이터 생성
python -c "
import json
import os

# 날씨 데이터
weather_data = {
    'seoul': {'temp': 24, 'condition': '맑음', 'humidity': 65},
    'busan': {'temp': 26, 'condition': '흐림', 'humidity': 70},
    'incheon': {'temp': 23, 'condition': '비', 'humidity': 80}
}

# 일정 데이터
calendar_data = {
    'today': [
        {'time': '09:00', 'title': '팀 스탠드업', 'type': 'meeting'},
        {'time': '14:00', 'title': '프로젝트 리뷰', 'type': 'review'},
        {'time': '16:00', 'title': '클라이언트 미팅', 'type': 'external'}
    ],
    'tomorrow': [
        {'time': '10:00', 'title': '디자인 싱크', 'type': 'meeting'},
        {'time': '15:00', 'title': '스프린트 계획', 'type': 'planning'}
    ]
}

os.makedirs('data', exist_ok=True)
with open('data/weather_data.json', 'w', encoding='utf-8') as f:
    json.dump(weather_data, f, ensure_ascii=False, indent=2)
    
with open('data/calendar_events.json', 'w', encoding='utf-8') as f:
    json.dump(calendar_data, f, ensure_ascii=False, indent=2)

print('Mock 데이터 생성 완료!')
"
```

## 핵심 기능

### 지능형 의도 분석
- RAG 기반 과거 대화 패턴 학습
- 복합 요청에 대한 멀티 액션 계획
- 컨텍스트 기반 의도 추론

### 전문 에이전트 협력
- 도메인별 전문화된 에이전트
- 에이전트 간 데이터 공유 및 협력
- 동적 에이전트 선택 및 조합

### MCP 표준 통신
- JSON-RPC 2.0 기반 표준 프로토콜
- 플러그인 방식의 API 연동
- 확장 가능한 도구 생태계

### 실시간 모니터링
- 에이전트 상호작용 시각화
- API 호출 성능 모니터링
- 대화 품질 평가 메트릭

## 실무 적용 사례

### 기업 내부 어시스턴트
- 사내 시스템 (HR, IT, 프로젝트 관리) 통합
- 직원 문의 자동 처리
- 업무 프로세스 자동화

### 고객 서비스 챗봇
- 다채널 고객 지원
- 백엔드 시스템 자동 연동
- 에스컬레이션 자동 처리

### 개발팀 어시스턴트
- GitHub, Jira, Slack 통합
- 코드 리뷰 자동화
- 배포 프로세스 관리

## 문제 해결

### API 서버 실행 오류
```bash
# 포트 충돌 시
lsof -ti:8000 | xargs kill -9
```

### 패키지 충돌
```bash
# 가상환경 재생성
conda deactivate
conda create -n ai-basic-lab4 python=3.10
conda activate ai-basic-lab4
pip install -r requirements.txt
```

### 성능 최적화
```bash
# 메모리 사용량 모니터링
python -c "import psutil; print(f'메모리: {psutil.virtual_memory().percent}%')"
```

---