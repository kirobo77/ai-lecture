"""
Lab 4 - Mock Notification API Server
알림/메시징을 위한 Mock API 서버
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import uuid

# FastAPI 앱 생성
app = FastAPI(
    title="Mock Notification API",
    description="알림/메시징을 위한 Mock API 서버",
    version="1.0.0"
)

# 데이터 모델 정의
class NotificationCreate(BaseModel):
    title: str
    message: str
    recipient: str  # email, phone, slack_channel 등
    type: str = "info"  # info, warning, error, success
    channel: str = "email"  # email, sms, slack, push
    priority: str = "normal"  # low, normal, high, urgent
    scheduled_at: Optional[str] = None

class Notification(BaseModel):
    id: str
    title: str
    message: str
    recipient: str
    type: str
    channel: str
    priority: str
    status: str  # pending, sent, delivered, failed
    created_at: str
    sent_at: Optional[str] = None
    delivered_at: Optional[str] = None

class SlackMessage(BaseModel):
    channel: str
    text: str
    username: Optional[str] = "ChatBot"
    icon: Optional[str] = ":robot_face:"
    attachments: List[Dict] = []

class EmailMessage(BaseModel):
    to: str
    subject: str
    body: str
    cc: Optional[List[str]] = []
    attachments: Optional[List[str]] = []

class NotificationStats(BaseModel):
    total_sent: int
    success_rate: float
    channel_distribution: Dict[str, int]
    type_distribution: Dict[str, int]

# Mock 데이터 저장소
NOTIFICATIONS = {}
NOTIFICATION_HISTORY = []

@app.get("/")
async def root():
    """API 루트 엔드포인트"""
    return {
        "service": "Mock Notification API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/notifications",
            "/notifications/send",
            "/notifications/slack",
            "/notifications/email",
            "/notifications/sms",
            "/notifications/stats"
        ]
    }

@app.get("/notifications", response_model=List[Notification])
async def get_notifications(status: Optional[str] = None, channel: Optional[str] = None):
    """모든 알림 조회"""
    notifications = list(NOTIFICATIONS.values())
    
    # 필터링
    if status:
        notifications = [n for n in notifications if n.status == status]
    if channel:
        notifications = [n for n in notifications if n.channel == channel]
    
    # 최신순 정렬
    notifications.sort(key=lambda x: x.created_at, reverse=True)
    
    return notifications

@app.get("/notifications/{notification_id}", response_model=Notification)
async def get_notification(notification_id: str):
    """특정 알림 조회"""
    if notification_id not in NOTIFICATIONS:
        raise HTTPException(status_code=404, detail="알림을 찾을 수 없습니다")
    
    return NOTIFICATIONS[notification_id]

@app.post("/notifications/send", response_model=Notification)
async def send_notification(notification_data: NotificationCreate):
    """일반 알림 발송"""
    notification_id = str(uuid.uuid4())[:8]
    
    # 발송 시뮬레이션
    success_rate = simulate_delivery_success(notification_data.channel, notification_data.priority)
    status = "sent" if success_rate > 0.8 else "failed"
    
    notification = Notification(
        id=notification_id,
        title=notification_data.title,
        message=notification_data.message,
        recipient=notification_data.recipient,
        type=notification_data.type,
        channel=notification_data.channel,
        priority=notification_data.priority,
        status=status,
        created_at=datetime.now().isoformat(),
        sent_at=datetime.now().isoformat() if status == "sent" else None,
        delivered_at=(datetime.now() + timedelta(seconds=5)).isoformat() if status == "sent" else None
    )
    
    NOTIFICATIONS[notification_id] = notification
    NOTIFICATION_HISTORY.append(notification)
    
    return notification

@app.post("/notifications/slack")
async def send_slack_message(slack_data: SlackMessage):
    """Slack 메시지 발송"""
    message_id = str(uuid.uuid4())[:8]
    
    # Slack 메시지 시뮬레이션
    notification = Notification(
        id=message_id,
        title=f"Slack: {slack_data.channel}",
        message=slack_data.text,
        recipient=slack_data.channel,
        type="info",
        channel="slack",
        priority="normal",
        status="sent",
        created_at=datetime.now().isoformat(),
        sent_at=datetime.now().isoformat(),
        delivered_at=(datetime.now() + timedelta(seconds=2)).isoformat()
    )
    
    NOTIFICATIONS[message_id] = notification
    NOTIFICATION_HISTORY.append(notification)
    
    return {
        "message_id": message_id,
        "channel": slack_data.channel,
        "text": slack_data.text,
        "username": slack_data.username,
        "status": "sent",
        "timestamp": datetime.now().isoformat(),
        "response": f"메시지가 {slack_data.channel} 채널에 성공적으로 전송되었습니다."
    }

@app.post("/notifications/email")
async def send_email(email_data: EmailMessage):
    """이메일 발송"""
    email_id = str(uuid.uuid4())[:8]
    
    # 이메일 발송 시뮬레이션
    notification = Notification(
        id=email_id,
        title=email_data.subject,
        message=email_data.body,
        recipient=email_data.to,
        type="info",
        channel="email",
        priority="normal",
        status="sent",
        created_at=datetime.now().isoformat(),
        sent_at=datetime.now().isoformat(),
        delivered_at=(datetime.now() + timedelta(seconds=10)).isoformat()
    )
    
    NOTIFICATIONS[email_id] = notification
    NOTIFICATION_HISTORY.append(notification)
    
    return {
        "email_id": email_id,
        "to": email_data.to,
        "subject": email_data.subject,
        "cc": email_data.cc,
        "status": "sent",
        "timestamp": datetime.now().isoformat(),
        "response": f"이메일이 {email_data.to}로 성공적으로 전송되었습니다."
    }

@app.post("/notifications/sms")
async def send_sms(phone: str, message: str, priority: str = "normal"):
    """SMS 발송"""
    sms_id = str(uuid.uuid4())[:8]
    
    # SMS 길이 제한 체크
    if len(message) > 160:
        raise HTTPException(status_code=400, detail="SMS 메시지는 160자를 초과할 수 없습니다")
    
    # SMS 발송 시뮬레이션
    notification = Notification(
        id=sms_id,
        title="SMS 알림",
        message=message,
        recipient=phone,
        type="info",
        channel="sms",
        priority=priority,
        status="sent",
        created_at=datetime.now().isoformat(),
        sent_at=datetime.now().isoformat(),
        delivered_at=(datetime.now() + timedelta(seconds=3)).isoformat()
    )
    
    NOTIFICATIONS[sms_id] = notification
    NOTIFICATION_HISTORY.append(notification)
    
    return {
        "sms_id": sms_id,
        "phone": phone,
        "message": message,
        "priority": priority,
        "status": "sent",
        "timestamp": datetime.now().isoformat(),
        "response": f"SMS가 {phone}로 성공적으로 전송되었습니다."
    }

@app.post("/notifications/broadcast")
async def broadcast_notification(
    title: str,
    message: str,
    channels: List[str],
    recipients: List[str],
    priority: str = "normal"
):
    """다중 채널 브로드캐스트"""
    broadcast_id = str(uuid.uuid4())[:8]
    sent_notifications = []
    
    for channel in channels:
        for recipient in recipients:
            notification_data = NotificationCreate(
                title=f"[브로드캐스트] {title}",
                message=message,
                recipient=recipient,
                type="info",
                channel=channel,
                priority=priority
            )
            
            notification = await send_notification(notification_data)
            sent_notifications.append(notification)
    
    return {
        "broadcast_id": broadcast_id,
        "title": title,
        "channels": channels,
        "recipients": recipients,
        "total_sent": len(sent_notifications),
        "timestamp": datetime.now().isoformat(),
        "notifications": sent_notifications
    }

@app.get("/notifications/stats", response_model=NotificationStats)
async def get_notification_stats():
    """알림 통계 정보"""
    total_notifications = len(NOTIFICATION_HISTORY)
    
    if total_notifications == 0:
        return NotificationStats(
            total_sent=0,
            success_rate=0.0,
            channel_distribution={},
            type_distribution={}
        )
    
    # 성공률 계산
    successful = len([n for n in NOTIFICATION_HISTORY if n.status == "sent"])
    success_rate = successful / total_notifications * 100
    
    # 채널별 분포
    channel_distribution = {}
    for notification in NOTIFICATION_HISTORY:
        channel = notification.channel
        channel_distribution[channel] = channel_distribution.get(channel, 0) + 1
    
    # 타입별 분포
    type_distribution = {}
    for notification in NOTIFICATION_HISTORY:
        notification_type = notification.type
        type_distribution[notification_type] = type_distribution.get(notification_type, 0) + 1
    
    return NotificationStats(
        total_sent=total_notifications,
        success_rate=round(success_rate, 2),
        channel_distribution=channel_distribution,
        type_distribution=type_distribution
    )

@app.get("/notifications/history")
async def get_notification_history(limit: int = 50):
    """알림 이력 조회"""
    # 최신순으로 정렬하여 제한된 수만 반환
    sorted_history = sorted(NOTIFICATION_HISTORY, key=lambda x: x.created_at, reverse=True)
    
    return {
        "notifications": sorted_history[:limit],
        "total_count": len(NOTIFICATION_HISTORY),
        "returned_count": min(limit, len(NOTIFICATION_HISTORY))
    }

@app.delete("/notifications/{notification_id}")
async def delete_notification(notification_id: str):
    """알림 삭제"""
    if notification_id not in NOTIFICATIONS:
        raise HTTPException(status_code=404, detail="알림을 찾을 수 없습니다")
    
    deleted_notification = NOTIFICATIONS[notification_id]
    del NOTIFICATIONS[notification_id]
    
    return {"message": f"알림 '{deleted_notification.title}'이 삭제되었습니다"}

@app.post("/notifications/test")
async def send_test_notifications():
    """테스트 알림 발송 (데모용)"""
    test_notifications = [
        NotificationCreate(
            title="팀 미팅 알림",
            message="15분 후 팀 스탠드업 미팅이 시작됩니다.",
            recipient="#team-dev",
            type="info",
            channel="slack",
            priority="normal"
        ),
        NotificationCreate(
            title="시스템 점검 안내",
            message="금일 23:00-24:00 시스템 점검이 예정되어 있습니다.",
            recipient="team@company.com",
            type="warning",
            channel="email",
            priority="high"
        ),
        NotificationCreate(
            title="긴급 알림",
            message="서버 장애가 발생했습니다. 즉시 확인 바랍니다.",
            recipient="010-1234-5678",
            type="error",
            channel="sms",
            priority="urgent"
        )
    ]
    
    sent_notifications = []
    for notification_data in test_notifications:
        notification = await send_notification(notification_data)
        sent_notifications.append(notification)
    
    return {
        "message": "테스트 알림이 발송되었습니다",
        "sent_notifications": sent_notifications
    }

def simulate_delivery_success(channel: str, priority: str) -> float:
    """채널과 우선순위에 따른 발송 성공률 시뮬레이션"""
    base_success_rates = {
        "email": 0.95,
        "slack": 0.98,
        "sms": 0.92,
        "push": 0.88
    }
    
    priority_multipliers = {
        "low": 0.9,
        "normal": 1.0,
        "high": 1.05,
        "urgent": 1.1
    }
    
    base_rate = base_success_rates.get(channel, 0.85)
    multiplier = priority_multipliers.get(priority, 1.0)
    
    return min(base_rate * multiplier, 1.0)

# 서버 실행 함수
def run_server():
    """Notification API 서버 실행"""
    print(" Notification API 서버 시작 중...")
    print(" URL: http://localhost:8004")
    print(" API 문서: http://localhost:8004/docs")
    
    uvicorn.run(
        "notification_api:app",
        host="0.0.0.0",
        port=8004,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    run_server() 