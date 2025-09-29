"""
Lab 4 - Mock Calendar API Server
일정 관리를 위한 Mock API 서버
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import json
import os
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional
import uuid

# FastAPI 앱 생성
app = FastAPI(
    title="Mock Calendar API",
    description="일정 관리를 위한 Mock API 서버",
    version="1.0.0"
)

# 데이터 모델 정의
class Event(BaseModel):
    id: Optional[str] = None
    title: str
    start_time: str
    end_time: Optional[str] = None
    type: str = "meeting"  # meeting, task, reminder, personal
    location: Optional[str] = None
    attendees: List[str] = []
    description: Optional[str] = None
    created_at: Optional[str] = None

class EventCreate(BaseModel):
    title: str
    start_time: str
    end_time: Optional[str] = None
    type: str = "meeting"
    location: Optional[str] = None
    attendees: List[str] = []
    description: Optional[str] = None

class DaySchedule(BaseModel):
    date: str
    events: List[Event]
    total_events: int
    free_time_slots: List[str]

# Mock 데이터 로드
def load_calendar_data():
    """일정 데이터 로드"""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'calendar_events.json')
    
    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 날짜별로 이벤트 재구성
            calendar_data = {}
            for date_key, events in data.items():
                calendar_data[date_key] = []
                for event in events:
                    event_obj = Event(
                        id=event.get('id', str(uuid.uuid4())[:8]),
                        title=event['title'],
                        start_time=event['start_time'],
                        end_time=event.get('end_time', ''),
                        type=event['type'],
                        location=event.get('location', ''),
                        attendees=event.get('attendees', []),
                        description=event.get('description', ''),
                        created_at=datetime.now().isoformat()
                    )
                    calendar_data[date_key].append(event_obj)
            return calendar_data
    else:
        # 기본 데이터
        today = datetime.now().strftime('%Y-%m-%d')
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        return {
            'today': [
                Event(
                    id="evt001",
                    title="팀 스탠드업 미팅",
                    start_time="09:00",
                    end_time="09:30",
                    type="meeting",
                    location="회의실 A",
                    attendees=["김개발", "박디자인", "이기획"],
                    created_at=datetime.now().isoformat()
                ),
                Event(
                    id="evt002",
                    title="프로젝트 리뷰",
                    start_time="14:00",
                    end_time="15:00",
                    type="review",
                    location="회의실 B",
                    attendees=["최팀장", "김개발"],
                    created_at=datetime.now().isoformat()
                )
            ],
            'tomorrow': [
                Event(
                    id="evt003",
                    title="클라이언트 미팅",
                    start_time="10:00",
                    end_time="11:30",
                    type="external",
                    location="외부 미팅룸",
                    attendees=["김대표", "박부장"],
                    created_at=datetime.now().isoformat()
                )
            ]
        }

# 글로벌 데이터
CALENDAR_DATA = load_calendar_data()

@app.get("/")
async def root():
    """API 루트 엔드포인트"""
    return {
        "service": "Mock Calendar API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/calendar/today",
            "/calendar/tomorrow",
            "/calendar/date/{date}",
            "/calendar/events",
            "/calendar/events/{event_id}",
            "/calendar/free-time/{date}"
        ]
    }

@app.get("/calendar/today", response_model=DaySchedule)
async def get_today_schedule():
    """오늘 일정 조회"""
    today = datetime.now().strftime('%Y-%m-%d')
    events = CALENDAR_DATA.get('today', [])
    
    # 빈 시간대 계산
    free_slots = calculate_free_time(events)
    
    return DaySchedule(
        date=today,
        events=events,
        total_events=len(events),
        free_time_slots=free_slots
    )

@app.get("/calendar/tomorrow", response_model=DaySchedule)
async def get_tomorrow_schedule():
    """내일 일정 조회"""
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    events = CALENDAR_DATA.get('tomorrow', [])
    
    free_slots = calculate_free_time(events)
    
    return DaySchedule(
        date=tomorrow,
        events=events,
        total_events=len(events),
        free_time_slots=free_slots
    )

@app.get("/calendar/date/{target_date}", response_model=DaySchedule)
async def get_schedule_by_date(target_date: str):
    """특정 날짜 일정 조회 (YYYY-MM-DD 형식)"""
    try:
        # 날짜 형식 검증
        datetime.strptime(target_date, '%Y-%m-%d')
    except ValueError:
        raise HTTPException(status_code=400, detail="날짜 형식이 올바르지 않습니다. (YYYY-MM-DD)")
    
    events = CALENDAR_DATA.get(target_date, [])
    free_slots = calculate_free_time(events)
    
    return DaySchedule(
        date=target_date,
        events=events,
        total_events=len(events),
        free_time_slots=free_slots
    )

@app.get("/calendar/events", response_model=List[Event])
async def get_all_events():
    """모든 일정 조회"""
    all_events = []
    for date_key, events in CALENDAR_DATA.items():
        all_events.extend(events)
    
    return all_events

@app.get("/calendar/events/{event_id}", response_model=Event)
async def get_event(event_id: str):
    """특정 이벤트 조회"""
    for date_key, events in CALENDAR_DATA.items():
        for event in events:
            if event.id == event_id:
                return event
    
    raise HTTPException(status_code=404, detail="이벤트를 찾을 수 없습니다")

@app.post("/calendar/events", response_model=Event)
async def create_event(event_data: EventCreate):
    """새 이벤트 생성"""
    new_event = Event(
        id=str(uuid.uuid4())[:8],
        title=event_data.title,
        start_time=event_data.start_time,
        end_time=event_data.end_time,
        type=event_data.type,
        location=event_data.location,
        attendees=event_data.attendees,
        description=event_data.description,
        created_at=datetime.now().isoformat()
    )
    
    # 오늘 날짜에 추가 (실제로는 사용자가 지정한 날짜에 추가해야 함)
    if 'today' not in CALENDAR_DATA:
        CALENDAR_DATA['today'] = []
    
    CALENDAR_DATA['today'].append(new_event)
    
    return new_event

@app.put("/calendar/events/{event_id}", response_model=Event)
async def update_event(event_id: str, event_data: EventCreate):
    """이벤트 수정"""
    for date_key, events in CALENDAR_DATA.items():
        for i, event in enumerate(events):
            if event.id == event_id:
                # 기존 이벤트 업데이트
                updated_event = Event(
                    id=event_id,
                    title=event_data.title,
                    start_time=event_data.start_time,
                    end_time=event_data.end_time,
                    type=event_data.type,
                    location=event_data.location,
                    attendees=event_data.attendees,
                    description=event_data.description,
                    created_at=event.created_at
                )
                
                CALENDAR_DATA[date_key][i] = updated_event
                return updated_event
    
    raise HTTPException(status_code=404, detail="이벤트를 찾을 수 없습니다")

@app.delete("/calendar/events/{event_id}")
async def delete_event(event_id: str):
    """이벤트 삭제"""
    for date_key, events in CALENDAR_DATA.items():
        for i, event in enumerate(events):
            if event.id == event_id:
                del CALENDAR_DATA[date_key][i]
                return {"message": f"이벤트 '{event.title}'가 삭제되었습니다"}
    
    raise HTTPException(status_code=404, detail="이벤트를 찾을 수 없습니다")

@app.get("/calendar/free-time/{target_date}")
async def get_free_time(target_date: str):
    """특정 날짜의 빈 시간대 조회"""
    events = CALENDAR_DATA.get(target_date, [])
    free_slots = calculate_free_time(events)
    
    return {
        "date": target_date,
        "free_time_slots": free_slots,
        "total_free_hours": len(free_slots),
        "recommendation": "가장 긴 빈 시간: " + (free_slots[0] if free_slots else "없음")
    }

@app.get("/calendar/summary")
async def get_calendar_summary():
    """일정 요약 정보"""
    total_events = sum(len(events) for events in CALENDAR_DATA.values())
    
    upcoming_events = []
    for date_key in ['today', 'tomorrow']:
        if date_key in CALENDAR_DATA:
            upcoming_events.extend(CALENDAR_DATA[date_key])
    
    return {
        "total_events": total_events,
        "upcoming_events_count": len(upcoming_events),
        "today_events": len(CALENDAR_DATA.get('today', [])),
        "tomorrow_events": len(CALENDAR_DATA.get('tomorrow', [])),
        "summary": f"총 {total_events}개 일정, 오늘 {len(CALENDAR_DATA.get('today', []))}개, 내일 {len(CALENDAR_DATA.get('tomorrow', []))}개"
    }

def calculate_free_time(events: List[Event]) -> List[str]:
    """빈 시간대 계산"""
    # 간단한 빈 시간 계산 (9시-18시 기준)
    work_hours = [f"{h:02d}:00-{h+1:02d}:00" for h in range(9, 18)]
    
    # 이벤트가 있는 시간대 제외
    busy_hours = []
    for event in events:
        start_hour = int(event.start_time.split(':')[0])
        if 9 <= start_hour < 18:
            busy_hours.append(start_hour)
    
    free_slots = []
    for hour in range(9, 18):
        if hour not in busy_hours:
            free_slots.append(f"{hour:02d}:00-{hour+1:02d}:00")
    
    return free_slots

# 서버 실행 함수
def run_server():
    """Calendar API 서버 실행"""
    print(" Calendar API 서버 시작 중...")
    print(" URL: http://localhost:8002")
    print(" API 문서: http://localhost:8002/docs")
    
    uvicorn.run(
        "calendar_api:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    run_server() 