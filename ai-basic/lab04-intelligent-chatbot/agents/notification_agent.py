"""
Lab 4 - Notification Agent
Notification API 전문 호출 및 알림 발송 에이전트
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import httpx
from datetime import datetime
from typing import Dict, List, Optional, Union
import json

class NotificationAgent:
    """Notification API 전문 호출 에이전트"""
    
    def __init__(self, api_base_url: str = "http://localhost:8004"):
        """Notification Agent 초기화"""
        self.name = "Notification Agent"
        self.version = "1.0.0"
        self.api_base_url = api_base_url.rstrip('/')
        
        # HTTP 클라이언트 설정
        self.client = httpx.Client(timeout=30.0)
        
        # 채널 타입 매핑
        self.channel_mapping = {
            '슬랙': 'slack',
            '이메일': 'email',
            '문자': 'sms',
            '팀즈': 'teams',
            'slack': 'slack',
            'email': 'email',
            'sms': 'sms',
            'teams': 'teams'
        }
        
        # 우선순위 매핑
        self.priority_mapping = {
            '높음': 'high',
            '보통': 'normal',
            '낮음': 'low',
            '긴급': 'urgent',
            'high': 'high',
            'normal': 'normal',
            'low': 'low',
            'urgent': 'urgent'
        }
        
        # 알림 타입 매핑
        self.notification_type_mapping = {
            '정보': 'info',
            '경고': 'warning',
            '오류': 'error',
            '성공': 'success',
            '작업': 'task',
            'info': 'info',
            'warning': 'warning',
            'error': 'error',
            'success': 'success',
            'task': 'task'
        }
        
        print(f"{self.name} 초기화 완료 (API: {self.api_base_url})")
    
    def normalize_channel(self, channel: str) -> str:
        """채널 타입 정규화"""
        if not channel:
            return 'slack'  # 기본값
        
        channel_lower = channel.lower().strip()
        
        # 직접 매핑
        if channel_lower in self.channel_mapping:
            return self.channel_mapping[channel_lower]
        
        # 부분 매칭
        for korean, english in self.channel_mapping.items():
            if korean in channel or channel in korean:
                return english
        
        return 'slack'  # 기본값
    
    def normalize_priority(self, priority: str) -> str:
        """우선순위 정규화"""
        if not priority:
            return 'normal'  # 기본값
        
        priority_lower = priority.lower().strip()
        return self.priority_mapping.get(priority_lower, 'normal')
    
    def normalize_notification_type(self, notification_type: str) -> str:
        """알림 타입 정규화"""
        if not notification_type:
            return 'info'  # 기본값
        
        type_lower = notification_type.lower().strip()
        return self.notification_type_mapping.get(type_lower, 'info')
    
    def send_general_notification(self, notification_data: Dict) -> Dict:
        """일반 알림 발송"""
        try:
            response = self.client.post(
                f"{self.api_base_url}/notifications/send",
                json=notification_data
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json(),
                    "action": "send_general"
                }
            else:
                return {
                    "success": False,
                    "error": f"일반 알림 발송 실패: {response.status_code}",
                    "notification_data": notification_data
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"일반 알림 발송 실패: {str(e)}",
                "notification_data": notification_data
            }
    
    def send_slack_message(self, slack_data: Dict) -> Dict:
        """Slack 메시지 발송"""
        try:
            # MCP와 일관성 있는 파라미터 구조로 정규화
            normalized_data = {
                "channel": slack_data.get("channel", "general"),
                "text": slack_data.get("message", slack_data.get("text", "")),
                "username": slack_data.get("username", "ChatBot"),
                "icon": slack_data.get("icon", ":robot_face:")
            }
            
            response = self.client.post(
                f"{self.api_base_url}/notifications/slack",
                json=normalized_data
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json(),
                    "action": "send_slack"
                }
            else:
                error_detail = f"Slack 메시지 발송 실패: {response.status_code}"
                try:
                    error_detail += f" - {response.text}"
                except:
                    pass
                print(f"🔍 [DEBUG] Slack API 에러: {error_detail}")
                print(f"🔍 [DEBUG] 전송 데이터: {normalized_data}")
                return {
                    "success": False,
                    "error": error_detail,
                    "slack_data": normalized_data
                }
                
        except Exception as e:
            error_msg = f"Slack 메시지 발송 실패: {str(e)}"
            print(f"🔍 [DEBUG] Slack 연결 에러: {error_msg}")
            print(f"🔍 [DEBUG] API URL: {self.api_base_url}/notifications/slack")
            print(f"🔍 [DEBUG] 전송 데이터: {normalized_data}")
            return {
                "success": False,
                "error": error_msg,
                "slack_data": normalized_data
            }
    
    def send_email(self, email_data: Dict) -> Dict:
        """이메일 발송"""
        try:
            # MCP와 일관성 있는 파라미터 구조로 정규화
            normalized_data = {
                "to": email_data.get("to", ""),
                "subject": email_data.get("subject", "알림"),
                "body": email_data.get("message", email_data.get("body", ""))
            }
            
            response = self.client.post(
                f"{self.api_base_url}/notifications/email",
                json=normalized_data
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json(),
                    "action": "send_email"
                }
            else:
                return {
                    "success": False,
                    "error": f"이메일 발송 실패: {response.status_code}",
                    "email_data": normalized_data
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"이메일 발송 실패: {str(e)}",
                "email_data": normalized_data
            }
    
    def send_sms(self, sms_data: Dict) -> Dict:
        """SMS 발송"""
        try:
            # Mock API는 개별 매개변수를 기대함
            params = {
                "phone": sms_data.get("phone_number", sms_data.get("recipient_phone", "")),
                "message": sms_data.get("message", ""),
                "priority": sms_data.get("priority", "normal")
            }
            response = self.client.post(
                f"{self.api_base_url}/notifications/sms",
                params=params
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json(),
                    "action": "send_sms"
                }
            else:
                return {
                    "success": False,
                    "error": f"SMS 발송 실패: {response.status_code}",
                    "sms_data": sms_data
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"SMS 발송 실패: {str(e)}",
                "sms_data": sms_data
            }
    
    def send_broadcast(self, broadcast_data: Dict) -> Dict:
        """브로드캐스트 메시지 발송"""
        try:
            response = self.client.post(
                f"{self.api_base_url}/notifications/broadcast",
                json=broadcast_data
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json(),
                    "action": "send_broadcast"
                }
            else:
                return {
                    "success": False,
                    "error": f"브로드캐스트 발송 실패: {response.status_code}",
                    "broadcast_data": broadcast_data
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"브로드캐스트 발송 실패: {str(e)}",
                "broadcast_data": broadcast_data
            }
    
    def get_notification_history(self, limit: int = 10) -> Dict:
        """알림 발송 이력 조회"""
        try:
            params = {"limit": limit}
            response = self.client.get(
                f"{self.api_base_url}/notifications/history",
                params=params
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json(),
                    "action": "get_history"
                }
            else:
                return {
                    "success": False,
                    "error": f"알림 이력 조회 실패: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"알림 이력 조회 실패: {str(e)}"
            }
    
    def get_notification_stats(self) -> Dict:
        """알림 통계 정보 조회"""
        try:
            response = self.client.get(f"{self.api_base_url}/notifications/stats")
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json(),
                    "action": "get_stats"
                }
            else:
                return {
                    "success": False,
                    "error": f"알림 통계 조회 실패: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"알림 통계 조회 실패: {str(e)}"
            }
    
    def format_notification_response(self, notification_result: Dict, request_type: str = "send") -> str:
        """알림 정보를 사용자 친화적 메시지로 변환"""
        if not notification_result.get("success"):
            error_msg = notification_result.get("error", "알 수 없는 오류")
            return f"죄송합니다. 알림 발송에 실패했습니다. ({error_msg})"
        
        data = notification_result["data"]
        
        if request_type == "send":
            # 알림 발송 성공 포맷
            action = notification_result.get("action", "send_general")
            
            if action == "send_slack":
                channel = data.get("channel", "채널")
                username = data.get("username", "사용자")
                text = data.get("text", "메시지")
                message_id = data.get("message_id", "")
                
                response = f"✅ Slack 메시지 발송 완료!\n\n"
                response += f"📢 채널: {channel}\n"
                response += f"👤 발송자: {username}\n"
                response += f"💬 내용: {text[:100]}"
                if len(text) > 100:
                    response += "..."
                response += f"\n🆔 메시지 ID: {message_id}"
                
            elif action == "send_email":
                recipient = data.get("to", "수신자")
                subject = data.get("subject", "제목")
                message_id = data.get("message_id", "")
                
                response = f"✅ 이메일 발송 완료!\n\n"
                response += f"📧 수신자: {recipient}\n"
                response += f"📝 제목: {subject}\n"
                response += f"🆔 메시지 ID: {message_id}"
                
            elif action == "send_sms":
                phone_number = data.get("phone_number", "전화번호")
                message = data.get("message", "메시지")
                message_id = data.get("message_id", "")
                
                response = f"✅ SMS 발송 완료!\n\n"
                response += f"📱 수신자: {phone_number}\n"
                response += f"💬 내용: {message[:50]}"
                if len(message) > 50:
                    response += "..."
                response += f"\n🆔 메시지 ID: {message_id}"
                
            elif action == "send_broadcast":
                channels = data.get("channels", [])
                total_sent = data.get("total_sent", 0)
                successful = data.get("successful", 0)
                failed = data.get("failed", 0)
                
                response = f"✅ 브로드캐스트 발송 완료!\n\n"
                response += f"📢 대상 채널: {', '.join(channels)}\n"
                response += f"📊 총 발송: {total_sent}개\n"
                response += f"✅ 성공: {successful}개\n"
                if failed > 0:
                    response += f"❌ 실패: {failed}개"
                
            else:  # send_general
                recipient = data.get("recipient", "수신자")
                channel = data.get("channel", "채널")
                notification_id = data.get("notification_id", "")
                
                response = f"✅ 알림 발송 완료!\n\n"
                response += f"👤 수신자: {recipient}\n"
                response += f"📢 채널: {channel}\n"
                response += f"🆔 알림 ID: {notification_id}"
            
            return response
            
        elif request_type == "history":
            # 알림 이력 포맷
            notifications = data.get("notifications", [])
            total_count = data.get("total_count", 0)
            
            if total_count == 0:
                return "📋 알림 발송 이력이 없습니다."
            
            response = f"📋 최근 알림 발송 이력 ({total_count}개)\n\n"
            
            for i, notification in enumerate(notifications[:10], 1):
                title = notification.get("title", "제목없음")
                recipient = notification.get("recipient", "수신자불명")
                channel = notification.get("channel", "채널불명")
                status = notification.get("status", "알 수 없음")
                timestamp = notification.get("timestamp", "시간불명")
                
                # 상태별 아이콘
                status_icons = {
                    "sent": "✅",
                    "failed": "❌", 
                    "pending": "⏳",
                    "delivered": "📨"
                }
                status_icon = status_icons.get(status, "❓")
                
                response += f"{status_icon} **{title}**\n"
                response += f"   👤 {recipient} | 📢 {channel}\n"
                response += f"   🕐 {timestamp[:16]}\n\n"
            
            if total_count > 10:
                response += f"... 및 {total_count-10}개 이력 더"
            
            return response
            
        elif request_type == "stats":
            # 알림 통계 포맷
            total_sent = data.get("total_sent", 0)
            success_rate = data.get("success_rate", 0)
            by_channel = data.get("by_channel", {})
            by_type = data.get("by_type", {})
            recent_activity = data.get("recent_activity", {})
            
            response = f"📊 알림 발송 통계\n\n"
            response += f"📨 총 발송: {total_sent:,}개\n"
            response += f"✅ 성공률: {success_rate:.1f}%\n\n"
            
            if by_channel:
                response += "📈 채널별 발송량:\n"
                for channel, count in by_channel.items():
                    percentage = (count / total_sent * 100) if total_sent > 0 else 0
                    response += f"   • {channel}: {count}개 ({percentage:.1f}%)\n"
                response += "\n"
            
            if by_type:
                response += "🏷️ 타입별 분포:\n"
                for noti_type, count in by_type.items():
                    percentage = (count / total_sent * 100) if total_sent > 0 else 0
                    response += f"   • {noti_type}: {count}개 ({percentage:.1f}%)\n"
                response += "\n"
            
            if recent_activity:
                today = recent_activity.get("today", 0)
                this_week = recent_activity.get("this_week", 0)
                response += f"📅 최근 활동:\n"
                response += f"   • 오늘: {today}개\n"
                response += f"   • 이번 주: {this_week}개\n"
            
            return response
        
        return "알림 정보를 처리할 수 없습니다."
    
    def process_notification_request(self, parameters: Dict) -> Dict:
        """알림 요청 처리 (Intent Classifier에서 호출)"""
        try:
            # 매개변수 추출
            action = parameters.get("action", "send")  # send, history, stats
            channel = parameters.get("channel", parameters.get("type", ""))
            title = parameters.get("title", parameters.get("subject", ""))
            message = parameters.get("message", parameters.get("content", parameters.get("text", "")))
            recipient = parameters.get("recipient", parameters.get("to", ""))
            priority = parameters.get("priority", "normal")
            notification_type = parameters.get("notification_type", "info")
            
            # 채널이 명시되지 않은 경우 사용자 입력에서 자동 감지
            if not channel:
                user_input = parameters.get("user_input", "")
                channel = self.detect_channel_from_input(user_input)
                print(f"🔄 [자동 채널 감지] '{channel}' (입력: {user_input})")
            
            print(f"알림 요청 처리: {action} - {channel}")
            
            # 정규화
            normalized_channel = self.normalize_channel(channel)
            normalized_priority = self.normalize_priority(priority)
            normalized_type = self.normalize_notification_type(notification_type)
            
            # 액션에 따른 처리
            if action == "send":
                # 알림 발송
                if not message:
                    # 다른 Agent의 결과로 메시지 자동 생성
                    collected_data = parameters.get("collected_data", {})
                    auto_message = self.create_notification_from_collected_data(collected_data, parameters)
                    
                    if auto_message:
                        message = auto_message
                        print(f"🔄 [자동 메시지 생성] {message[:50]}...")
                    else:
                        return {
                            "success": False,
                            "agent": self.name,
                            "response": "발송할 메시지를 알려주세요.",
                            "error": "메시지 누락"
                        }
                
                # 채널별 전용 API 호출
                if normalized_channel == "slack":
                    slack_data = {
                        "channel": recipient or "#general",
                        "text": message,
                        "username": "AI Assistant",
                        "icon": ":robot_face:"
                    }
                    result = self.send_slack_message(slack_data)
                    
                elif normalized_channel == "email":
                    if not recipient:
                        return {
                            "success": False,
                            "agent": self.name,
                            "response": "이메일 수신자를 알려주세요.",
                            "error": "수신자 누락"
                        }
                    
                    email_data = {
                        "to": recipient,
                        "subject": title or "AI Assistant 알림",
                        "body": message
                    }
                    result = self.send_email(email_data)
                    
                elif normalized_channel == "sms":
                    if not recipient:
                        return {
                            "success": False,
                            "agent": self.name,
                            "response": "SMS 수신자 전화번호를 알려주세요.",
                            "error": "전화번호 누락"
                        }
                    
                    sms_data = {
                        "phone_number": recipient,
                        "message": message
                    }
                    result = self.send_sms(sms_data)
                    
                else:
                    # 일반 알림
                    notification_data = {
                        "title": title or "AI Assistant 알림",
                        "message": message,
                        "recipient": recipient or "전체",
                        "type": normalized_type,
                        "channel": normalized_channel,
                        "priority": normalized_priority
                    }
                    result = self.send_general_notification(notification_data)
                
                formatted_response = self.format_notification_response(result, "send")
                
            elif action == "broadcast":
                # 브로드캐스트 발송
                if not message:
                    return {
                        "success": False,
                        "agent": self.name,
                        "response": "브로드캐스트할 메시지를 알려주세요.",
                        "error": "메시지 누락"
                    }
                
                broadcast_data = {
                    "title": title or "중요 공지",
                    "message": message,
                    "channels": parameters.get("channels", ["slack", "email"]),
                    "recipients": parameters.get("recipients", []),
                    "priority": normalized_priority
                }
                result = self.send_broadcast(broadcast_data)
                formatted_response = self.format_notification_response(result, "send")
                
            elif action == "history":
                # 알림 이력 조회
                limit = parameters.get("limit", 10)
                result = self.get_notification_history(limit)
                formatted_response = self.format_notification_response(result, "history")
                
            elif action == "stats":
                # 알림 통계 조회
                result = self.get_notification_stats()
                formatted_response = self.format_notification_response(result, "stats")
                
            else:
                # 기본값: 일반 발송
                notification_data = {
                    "title": title or "AI Assistant 알림",
                    "message": message or "테스트 메시지",
                    "recipient": recipient or "전체",
                    "type": normalized_type,
                    "channel": normalized_channel,
                    "priority": normalized_priority
                }
                result = self.send_general_notification(notification_data)
                formatted_response = self.format_notification_response(result, "send")
            
            return {
                "success": result.get("success", False),
                "agent": self.name,
                "response": formatted_response,
                "raw_data": result,
                "processed_at": datetime.now().isoformat(),
                "action": action,
                "channel": normalized_channel
            }
            
        except Exception as e:
            return {
                "success": False,
                "agent": self.name,
                "response": f"알림 처리 중 오류가 발생했습니다: {str(e)}",
                "error": str(e),
                "processed_at": datetime.now().isoformat()
            }
    
    def create_notification_from_other_agents(self, weather_info: str = "", calendar_info: str = "", file_info: str = "") -> str:
        """다른 에이전트 결과를 종합해서 알림 메시지 생성"""
        try:
            message_parts = []
            
            if weather_info:
                message_parts.append(f"🌤️ 날씨: {weather_info}")
            
            if calendar_info:
                message_parts.append(f"📅 일정: {calendar_info}")
            
            if file_info:
                message_parts.append(f"📁 파일: {file_info}")
            
            if message_parts:
                return "\n\n".join(message_parts)
            else:
                return "AI Assistant에서 알려드립니다."
                
        except Exception as e:
            return f"알림 메시지 생성 중 오류: {str(e)}"
    
    def create_notification_from_collected_data(self, collected_data: Dict, parameters: Dict) -> str:
        """수집된 데이터로부터 자동 알림 메시지 생성"""
        try:
            message_parts = []
            intent = parameters.get("intent", "")
            user_input = parameters.get("user_input", "")
            
            # 의도에 따른 메시지 헤더
            if "weather" in intent:
                message_parts.append("📢 **날씨 정보 알림**")
            elif "calendar" in intent:
                message_parts.append("📢 **일정 정보 알림**")
            elif "file" in intent:
                message_parts.append("📢 **파일 정보 알림**")
            else:
                message_parts.append("📢 **AI Assistant 알림**")
            
            # 각 Agent 결과 추가
            if collected_data.get("weather_info"):
                weather_info = collected_data["weather_info"]
                # 간단한 요약으로 변환
                weather_summary = self.extract_weather_summary(weather_info)
                message_parts.append(f"🌤️ {weather_summary}")
            
            if collected_data.get("calendar_info"):
                calendar_info = collected_data["calendar_info"]
                calendar_summary = self.extract_calendar_summary(calendar_info)
                message_parts.append(f"📅 {calendar_summary}")
            
            if collected_data.get("file_info"):
                file_info = collected_data["file_info"]
                file_summary = self.extract_file_summary(file_info)
                message_parts.append(f"📁 {file_summary}")
            
            # 메시지가 있으면 조합, 없으면 기본 메시지
            if len(message_parts) > 1:
                return "\n\n".join(message_parts)
            else:
                return f"📢 **AI Assistant 알림**\n\n요청하신 '{user_input}' 작업을 처리했습니다."
                
        except Exception as e:
            print(f"자동 메시지 생성 실패: {e}")
            return "AI Assistant에서 알려드립니다."
    
    def extract_weather_summary(self, weather_info: str) -> str:
        """날씨 정보에서 핵심 요약 추출"""
        try:
            # 온도와 날씨 상태 추출
            if "온도:" in weather_info and "날씨:" in weather_info:
                lines = weather_info.split('\n')
                temp = ""
                condition = ""
                for line in lines:
                    if "온도:" in line:
                        temp = line.split("온도:")[-1].strip()
                    elif "날씨:" in line:
                        condition = line.split("날씨:")[-1].strip()
                
                if temp and condition:
                    return f"현재 {condition}, {temp}"
            
            return weather_info[:50] + "..." if len(weather_info) > 50 else weather_info
        except:
            return "날씨 정보 확인됨"
    
    def extract_calendar_summary(self, calendar_info: str) -> str:
        """일정 정보에서 핵심 요약 추출"""
        try:
            # 일정 개수나 주요 일정 추출
            if "일정" in calendar_info:
                return calendar_info[:100] + "..." if len(calendar_info) > 100 else calendar_info
            return calendar_info[:50] + "..." if len(calendar_info) > 50 else calendar_info
        except:
            return "일정 정보 확인됨"
    
    def extract_file_summary(self, file_info: str) -> str:
        """파일 정보에서 핵심 요약 추출"""
        try:
            if "검색 결과" in file_info or "파일" in file_info:
                return file_info[:100] + "..." if len(file_info) > 100 else file_info
            return file_info[:50] + "..." if len(file_info) > 50 else file_info
        except:
            return "파일 정보 확인됨"
    
    def detect_channel_from_input(self, user_input: str) -> str:
        """사용자 입력에서 알림 채널 자동 감지"""
        try:
            user_input_lower = user_input.lower()
            
            # 이메일 키워드들
            email_keywords = [
                "이메일로", "메일로", "email로", "이메일에", "메일에", "이메일", "메일"
            ]
            
            # Slack 키워드들  
            slack_keywords = [
                "슬랙에", "슬랙으로", "slack에", "slack으로", "슬랙", "채널에", "팀에게"
            ]
            
            # SMS 키워드들
            sms_keywords = [
                "sms로", "문자로", "휴대폰으로", "전화로", "문자", "핸드폰으로"
            ]
            
            # 우선순위 순서로 검사 (더 구체적인 것부터)
            for keyword in email_keywords:
                if keyword in user_input_lower:
                    print(f"🔍 이메일 키워드 감지: '{keyword}'")
                    return "email"
            
            for keyword in sms_keywords:
                if keyword in user_input_lower:
                    print(f"🔍 SMS 키워드 감지: '{keyword}'")
                    return "sms"
            
            for keyword in slack_keywords:
                if keyword in user_input_lower:
                    print(f"🔍 Slack 키워드 감지: '{keyword}'")
                    return "slack"
            
            # 수신자 정보로 추론
            if "팀에게" in user_input_lower or "모두에게" in user_input_lower:
                print(f"🔍 수신자 정보로 Slack 추론")
                return "slack"
            
            # 기본값
            print(f"🔍 기본 채널 사용: slack")
            return "slack"
            
        except Exception as e:
            print(f"채널 감지 실패: {e}")
            return "slack"
    
    def get_capabilities(self) -> Dict:
        """에이전트 능력 정보"""
        return {
            "name": self.name,
            "version": self.version,
            "description": "Notification API 전문 호출 및 알림 발송",
            "supported_operations": [
                "Slack 메시지 발송",
                "이메일 발송",
                "SMS 발송",
                "브로드캐스트 발송",
                "알림 이력 조회",
                "알림 통계 조회"
            ],
            "supported_channels": list(self.channel_mapping.values()),
            "supported_priorities": list(self.priority_mapping.values()),
            "supported_types": list(self.notification_type_mapping.values()),
            "api_endpoint": self.api_base_url
        }
    
    def __del__(self):
        """소멸자 - HTTP 클라이언트 정리"""
        try:
            self.client.close()
        except:
            pass

def test_notification_agent():
    """Notification Agent 테스트"""
    print("=" * 60)
    print("Notification Agent 테스트")
    print("=" * 60)
    
    # Notification Agent 초기화
    agent = NotificationAgent()
    
    # 테스트 케이스들
    test_cases = [
        {"action": "send", "channel": "slack", "message": "테스트 메시지입니다", "recipient": "#general"},
        {"action": "send", "channel": "email", "title": "테스트 이메일", "message": "이메일 테스트", "recipient": "test@example.com"},
        {"action": "send", "channel": "sms", "message": "SMS 테스트", "recipient": "010-1234-5678"},
        {"action": "broadcast", "title": "중요 공지", "message": "전체 공지사항입니다"},
        {"action": "history", "limit": 5},
        {"action": "stats"}
    ]
    
    print(f"\n{len(test_cases)}개 테스트 케이스로 Notification Agent 테스트:")
    print("-" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}] 테스트: {test_case}")
        
        # Notification Agent 호출
        result = agent.process_notification_request(test_case)
        
        print(f"성공 여부: {result['success']}")
        print(f"응답:\n{result['response']}")
        
        if not result['success']:
            print(f"오류: {result.get('error', '알 수 없는 오류')}")
    
    # 다중 에이전트 연동 테스트
    print(f"\n" + "=" * 60)
    print("다중 에이전트 연동 알림 테스트:")
    combined_message = agent.create_notification_from_other_agents(
        weather_info="서울 24도 맑음",
        calendar_info="오늘 3개 일정 있음",
        file_info="프로젝트 문서 5개 발견"
    )
    print(f"통합 메시지:\n{combined_message}")
    
    # 에이전트 능력 정보
    print(f"\n📋 Notification Agent 정보:")
    capabilities = agent.get_capabilities()
    print(f"  이름: {capabilities['name']}")
    print(f"  설명: {capabilities['description']}")
    print(f"  지원 기능: {len(capabilities['supported_operations'])}개")
    print(f"  지원 채널: {capabilities['supported_channels']}")
    print(f"  API 엔드포인트: {capabilities['api_endpoint']}")
    
    print(f"\n" + "=" * 60)
    print("Notification Agent 테스트 완료!")

if __name__ == "__main__":
    test_notification_agent() 