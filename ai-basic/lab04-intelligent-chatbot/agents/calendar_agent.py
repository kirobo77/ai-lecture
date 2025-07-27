"""
Lab 4 - Calendar Agent
Calendar API 전문 호출 및 일정 관리 에이전트
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import httpx
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Union
import json
import re

class CalendarAgent:
    """Calendar API 전문 호출 에이전트"""
    
    def __init__(self, api_base_url: str = "http://localhost:8002"):
        """Calendar Agent 초기화"""
        self.name = "Calendar Agent"
        self.version = "1.0.0"
        self.api_base_url = api_base_url.rstrip('/')
        
        # HTTP 클라이언트 설정
        self.client = httpx.Client(timeout=30.0)
        
        # 날짜 표현 패턴 매핑
        self.date_patterns = {
            '오늘': 'today',
            '내일': 'tomorrow',
            '어제': 'yesterday',
            '이번주': 'this_week',
            '다음주': 'next_week',
            'today': 'today',
            'tomorrow': 'tomorrow',
            'yesterday': 'yesterday'
        }
        
        print(f"{self.name} 초기화 완료 (API: {self.api_base_url})")
    
    def parse_date(self, date_str: str) -> str:
        """날짜 표현을 API에서 사용할 형식으로 변환"""
        date_str = date_str.lower().strip()
        
        # 패턴 매핑 확인
        if date_str in self.date_patterns:
            return self.date_patterns[date_str]
        
        # 날짜 패턴 매칭 (YYYY-MM-DD, MM-DD 등)
        if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
            return date_str
        elif re.match(r'\d{2}-\d{2}', date_str):
            current_year = datetime.now().year
            return f"{current_year}-{date_str}"
        
        # 기본값: 오늘
        return 'today'
    
    def parse_time(self, time_str: str) -> str:
        """시간 표현을 HH:MM 형식으로 변환"""
        if not time_str:
            return "09:00"  # 기본값
        
        # 기존 HH:MM 형식
        if re.match(r'\d{2}:\d{2}', time_str):
            return time_str
        
        # 한 자리 숫자 시간 (예: "3시" -> "03:00")
        hour_match = re.search(r'(\d{1,2})\s*시', time_str)
        if hour_match:
            hour = int(hour_match.group(1))
            return f"{hour:02d}:00"
        
        # 영어 시간 표현 (예: "3pm" -> "15:00")
        pm_match = re.search(r'(\d{1,2})\s*pm', time_str.lower())
        if pm_match:
            hour = int(pm_match.group(1))
            if hour != 12:
                hour += 12
            return f"{hour:02d}:00"
        
        am_match = re.search(r'(\d{1,2})\s*am', time_str.lower())
        if am_match:
            hour = int(am_match.group(1))
            if hour == 12:
                hour = 0
            return f"{hour:02d}:00"
        
        # 숫자만 있는 경우 (예: "15" -> "15:00")
        if time_str.isdigit():
            hour = int(time_str)
            if 0 <= hour <= 23:
                return f"{hour:02d}:00"
        
        return "09:00"  # 기본값
    
    def get_today_schedule(self) -> Dict:
        """오늘 일정 조회"""
        try:
            response = self.client.get(f"{self.api_base_url}/calendar/today")
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json(),
                    "date_type": "today"
                }
            else:
                return {
                    "success": False,
                    "error": f"오늘 일정 조회 실패: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"오늘 일정 조회 실패: {str(e)}"
            }
    
    def get_tomorrow_schedule(self) -> Dict:
        """내일 일정 조회"""
        try:
            response = self.client.get(f"{self.api_base_url}/calendar/tomorrow")
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json(),
                    "date_type": "tomorrow"
                }
            else:
                return {
                    "success": False,
                    "error": f"내일 일정 조회 실패: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"내일 일정 조회 실패: {str(e)}"
            }
    
    def get_schedule_by_date(self, target_date: str) -> Dict:
        """특정 날짜 일정 조회"""
        try:
            response = self.client.get(f"{self.api_base_url}/calendar/date/{target_date}")
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json(),
                    "date_type": "specific",
                    "target_date": target_date
                }
            else:
                return {
                    "success": False,
                    "error": f"일정 조회 실패: {response.status_code}",
                    "target_date": target_date
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"일정 조회 실패: {str(e)}",
                "target_date": target_date
            }
    
    def create_event(self, event_data: Dict) -> Dict:
        """새 일정 생성"""
        try:
            response = self.client.post(
                f"{self.api_base_url}/calendar/events",
                json=event_data
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json(),
                    "action": "create"
                }
            else:
                return {
                    "success": False,
                    "error": f"일정 생성 실패: {response.status_code}",
                    "event_data": event_data
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"일정 생성 실패: {str(e)}",
                "event_data": event_data
            }
    
    def get_free_time(self, target_date: str) -> Dict:
        """특정 날짜의 빈 시간 조회"""
        try:
            response = self.client.get(f"{self.api_base_url}/calendar/free-time/{target_date}")
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json(),
                    "action": "free_time",
                    "target_date": target_date
                }
            else:
                return {
                    "success": False,
                    "error": f"빈 시간 조회 실패: {response.status_code}",
                    "target_date": target_date
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"빈 시간 조회 실패: {str(e)}",
                "target_date": target_date
            }
    
    def get_calendar_summary(self) -> Dict:
        """일정 요약 정보 조회"""
        try:
            response = self.client.get(f"{self.api_base_url}/calendar/summary")
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json(),
                    "action": "summary"
                }
            else:
                return {
                    "success": False,
                    "error": f"일정 요약 조회 실패: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"일정 요약 조회 실패: {str(e)}"
            }
    
    def format_calendar_response(self, calendar_result: Dict, request_type: str = "query") -> str:
        """일정 정보를 사용자 친화적 메시지로 변환"""
        if not calendar_result.get("success"):
            error_msg = calendar_result.get("error", "알 수 없는 오류")
            return f"죄송합니다. 일정 정보를 가져올 수 없습니다. ({error_msg})"
        
        data = calendar_result["data"]
        
        if request_type == "query":
            # 일정 조회 포맷
            date_info = data.get("date", "날짜정보없음")
            events = data.get("events", [])
            total_events = data.get("total_events", 0)
            free_slots = data.get("free_time_slots", [])
            
            if total_events == 0:
                response = f"📅 {date_info}\n일정이 없습니다. 자유로운 하루입니다!"
            else:
                response = f"📅 {date_info} 일정 ({total_events}개)\n\n"
                
                for i, event in enumerate(events, 1):
                    title = event.get("title", "제목없음")
                    start_time = event.get("start_time", "시간미정")
                    end_time = event.get("end_time", "")
                    location = event.get("location", "")
                    event_type = event.get("type", "meeting")
                    attendees = event.get("attendees", [])
                    
                    response += f"🕐 {start_time}"
                    if end_time:
                        response += f" - {end_time}"
                    response += f"\n📋 {title}"
                    
                    if location:
                        response += f"\n📍 {location}"
                    
                    if attendees:
                        response += f"\n👥 참석자: {', '.join(attendees[:3])}"
                        if len(attendees) > 3:
                            response += f" 외 {len(attendees)-3}명"
                    
                    # 이벤트 타입에 따른 아이콘
                    type_icons = {
                        "meeting": "💼",
                        "review": "📊", 
                        "external": "🤝",
                        "personal": "🏠",
                        "task": "✅"
                    }
                    if event_type in type_icons:
                        response += f"\n{type_icons[event_type]} {event_type.title()}"
                    
                    if i < len(events):
                        response += "\n\n"
            
            # 빈 시간 정보 추가
            if free_slots:
                response += f"\n\n⏰ 여유 시간: {len(free_slots)}시간"
                if len(free_slots) <= 3:
                    response += f" ({', '.join(free_slots)})"
            
            return response
            
        elif request_type == "create":
            # 일정 생성 포맷
            event = data
            title = event.get("title", "새 일정")
            start_time = event.get("start_time", "시간미정")
            
            response = f"✅ 일정이 생성되었습니다!\n\n"
            response += f"📋 제목: {title}\n"
            response += f"🕐 시간: {start_time}"
            
            if event.get("end_time"):
                response += f" - {event['end_time']}"
            
            if event.get("location"):
                response += f"\n📍 장소: {event['location']}"
            
            return response
            
        elif request_type == "free_time":
            # 빈 시간 조회 포맷
            target_date = calendar_result.get("target_date", "날짜정보없음")
            free_slots = data.get("free_time_slots", [])
            total_free_hours = data.get("total_free_hours", 0)
            recommendation = data.get("recommendation", "")
            
            response = f"⏰ {target_date} 여유 시간\n\n"
            
            if total_free_hours == 0:
                response += "죄송합니다. 여유 시간이 없습니다. 꽉 찬 하루네요!"
            else:
                response += f"총 {total_free_hours}시간의 여유가 있습니다.\n\n"
                
                if len(free_slots) <= 5:
                    for slot in free_slots:
                        response += f"⏱️ {slot}\n"
                else:
                    response += f"⏱️ {', '.join(free_slots[:3])} 외 {len(free_slots)-3}개 시간대"
                
                if recommendation:
                    response += f"\n\n💡 추천: {recommendation}"
            
            return response
            
        elif request_type == "summary":
            # 일정 요약 포맷
            total_events = data.get("total_events", 0)
            today_events = data.get("today_events", 0)
            tomorrow_events = data.get("tomorrow_events", 0)
            summary = data.get("summary", "")
            
            response = f"📊 일정 요약\n\n"
            response += f"📅 전체 일정: {total_events}개\n"
            response += f"🌟 오늘: {today_events}개\n"
            response += f"🌅 내일: {tomorrow_events}개\n"
            
            if summary:
                response += f"\n💼 {summary}"
            
            return response
        
        return "일정 정보를 처리할 수 없습니다."
    
    def process_calendar_request(self, parameters: Dict) -> Dict:
        """일정 요청 처리 (Intent Classifier에서 호출)"""
        try:
            # 매개변수 추출
            action = parameters.get("action", "query")  # query, create, free_time, summary
            date_param = parameters.get("date", parameters.get("when", "오늘"))
            title = parameters.get("title", parameters.get("event_title", ""))
            time_param = parameters.get("time", parameters.get("start_time", ""))
            
            print(f"일정 요청 처리: {action} - {date_param}")
            
            # 날짜 파싱
            parsed_date = self.parse_date(date_param)
            
            # 액션에 따른 처리
            if action == "create":
                # 일정 생성
                if not title:
                    return {
                        "success": False,
                        "agent": self.name,
                        "response": "일정 제목을 알려주세요.",
                        "error": "제목 누락"
                    }
                
                event_data = {
                    "title": title,
                    "start_time": self.parse_time(time_param),
                    "type": parameters.get("type", "meeting"),
                    "location": parameters.get("location", ""),
                    "attendees": parameters.get("attendees", []),
                    "description": parameters.get("description", "")
                }
                
                result = self.create_event(event_data)
                formatted_response = self.format_calendar_response(result, "create")
                
            elif action == "free_time":
                # 빈 시간 조회
                if parsed_date in ['today', 'tomorrow']:
                    # today/tomorrow를 날짜 형식으로 변환
                    if parsed_date == 'today':
                        target_date = datetime.now().strftime('%Y-%m-%d')
                    else:
                        target_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
                else:
                    target_date = parsed_date
                
                result = self.get_free_time(target_date)
                formatted_response = self.format_calendar_response(result, "free_time")
                
            elif action == "summary":
                # 일정 요약
                result = self.get_calendar_summary()
                formatted_response = self.format_calendar_response(result, "summary")
                
            else:  # query (기본값)
                # 일정 조회
                if parsed_date == 'today':
                    result = self.get_today_schedule()
                elif parsed_date == 'tomorrow':
                    result = self.get_tomorrow_schedule()
                else:
                    # 특정 날짜 조회
                    if parsed_date not in ['today', 'tomorrow']:
                        result = self.get_schedule_by_date(parsed_date)
                    else:
                        result = self.get_today_schedule()
                
                formatted_response = self.format_calendar_response(result, "query")
            
            return {
                "success": result.get("success", False),
                "agent": self.name,
                "response": formatted_response,
                "raw_data": result,
                "processed_at": datetime.now().isoformat(),
                "action": action,
                "parsed_date": parsed_date
            }
            
        except Exception as e:
            return {
                "success": False,
                "agent": self.name,
                "response": f"일정 처리 중 오류가 발생했습니다: {str(e)}",
                "error": str(e),
                "processed_at": datetime.now().isoformat()
            }
    
    def get_schedule_summary_for_notification(self, date_param: str = "today") -> str:
        """알림용 간단한 일정 요약"""
        try:
            parsed_date = self.parse_date(date_param)
            
            if parsed_date == 'today':
                result = self.get_today_schedule()
            elif parsed_date == 'tomorrow':
                result = self.get_tomorrow_schedule()
            else:
                result = self.get_schedule_by_date(parsed_date)
            
            if result.get("success"):
                data = result["data"]
                events = data.get("events", [])
                total_events = len(events)
                
                if total_events == 0:
                    return f"{date_param} 일정이 없습니다."
                else:
                    summary = f"{date_param} {total_events}개 일정: "
                    event_titles = [e.get("title", "제목없음") for e in events[:3]]
                    summary += ", ".join(event_titles)
                    if total_events > 3:
                        summary += f" 외 {total_events-3}개"
                    return summary
            else:
                return f"{date_param} 일정 정보를 가져올 수 없습니다."
                
        except Exception as e:
            return f"일정 요약 중 오류: {str(e)}"
    
    def get_capabilities(self) -> Dict:
        """에이전트 능력 정보"""
        return {
            "name": self.name,
            "version": self.version,
            "description": "Calendar API 전문 호출 및 일정 관리",
            "supported_operations": [
                "일정 조회 (오늘, 내일, 특정 날짜)",
                "일정 생성",
                "빈 시간 조회",
                "일정 요약",
                "날짜/시간 파싱"
            ],
            "supported_date_formats": [
                "오늘, 내일",
                "YYYY-MM-DD",
                "MM-DD",
                "상대적 표현"
            ],
            "supported_time_formats": [
                "HH:MM",
                "N시 (한국어)",
                "Nam/pm (영어)",
                "24시간 형식"
            ],
            "api_endpoint": self.api_base_url
        }
    
    def __del__(self):
        """소멸자 - HTTP 클라이언트 정리"""
        try:
            self.client.close()
        except:
            pass

def test_calendar_agent():
    """Calendar Agent 테스트"""
    print("=" * 60)
    print("Calendar Agent 테스트")
    print("=" * 60)
    
    # Calendar Agent 초기화
    agent = CalendarAgent()
    
    # 테스트 케이스들
    test_cases = [
        {"action": "query", "date": "오늘"},
        {"action": "query", "date": "내일"},
        {"action": "summary"},
        {"action": "free_time", "date": "today"},
        {"action": "create", "title": "테스트 미팅", "time": "15시", "location": "회의실 A"},
        {"action": "query", "date": "2024-12-31"}  # 특정 날짜
    ]
    
    print(f"\n{len(test_cases)}개 테스트 케이스로 Calendar Agent 테스트:")
    print("-" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}] 테스트: {test_case}")
        
        # Calendar Agent 호출
        result = agent.process_calendar_request(test_case)
        
        print(f"성공 여부: {result['success']}")
        print(f"응답:\n{result['response']}")
        
        if not result['success']:
            print(f"오류: {result.get('error', '알 수 없는 오류')}")
    
    # 알림용 요약 테스트
    print(f"\n" + "=" * 60)
    print("알림용 일정 요약 테스트:")
    summary = agent.get_schedule_summary_for_notification("오늘")
    print(f"  오늘 요약: {summary}")
    
    # 에이전트 능력 정보
    print(f"\n📋 Calendar Agent 정보:")
    capabilities = agent.get_capabilities()
    print(f"  이름: {capabilities['name']}")
    print(f"  설명: {capabilities['description']}")
    print(f"  지원 기능: {len(capabilities['supported_operations'])}개")
    print(f"  API 엔드포인트: {capabilities['api_endpoint']}")
    
    print(f"\n" + "=" * 60)
    print("Calendar Agent 테스트 완료!")

if __name__ == "__main__":
    test_calendar_agent() 