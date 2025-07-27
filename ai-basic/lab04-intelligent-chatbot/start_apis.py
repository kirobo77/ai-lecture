#!/usr/bin/env python3
"""
Lab 4 - Mock API 서버 실행 스크립트
모든 Mock API 서버를 한 번에 실행합니다.
"""

import subprocess
import time
import sys
import os
from pathlib import Path

# API 서버 설정
API_SERVERS = [
    {
        "name": "Weather API",
        "script": "mock_apis/weather_api.py",
        "port": 8001,
        "description": "날씨 정보 API"
    },
    {
        "name": "Calendar API", 
        "script": "mock_apis/calendar_api.py",
        "port": 8002,
        "description": "일정 관리 API"
    },
    {
        "name": "File Manager API",
        "script": "mock_apis/file_manager_api.py", 
        "port": 8003,
        "description": "파일 관리 API"
    },
    {
        "name": "Notification API",
        "script": "mock_apis/notification_api.py",
        "port": 8004,
        "description": "알림/메시징 API"
    }
]

def check_port_available(port):
    """포트 사용 가능 여부 확인"""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False

def start_api_server(server_info):
    """개별 API 서버 시작"""
    name = server_info["name"]
    script = server_info["script"]
    port = server_info["port"]
    description = server_info["description"]
    
    print(f"🚀 {name} 시작 중... (포트: {port})")
    
    if not check_port_available(port):
        print(f"⚠️  포트 {port}가 이미 사용 중입니다. {name}을 건너뜁니다.")
        return None
    
    try:
        # 서버 프로세스 시작
        process = subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 서버 시작 대기
        time.sleep(2)
        
        if process.poll() is None:
            print(f"✅ {name} 시작 완료! ({description})")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ {name} 시작 실패:")
            print(f"   stdout: {stdout}")
            print(f"   stderr: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ {name} 시작 중 오류: {e}")
        return None

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("🔌 Lab 4 - Mock API 서버 실행 스크립트")
    print("=" * 60)
    
    # 현재 디렉토리 확인
    current_dir = Path.cwd()
    if not (current_dir / "mock_apis").exists():
        print("❌ mock_apis 폴더를 찾을 수 없습니다.")
        print("   lab04-intelligent-chatbot 폴더에서 실행해주세요.")
        sys.exit(1)
    
    print(f"📁 현재 디렉토리: {current_dir}")
    print(f"🔧 Python 실행 파일: {sys.executable}")
    print()
    
    # API 서버들 시작
    processes = []
    
    for server_info in API_SERVERS:
        process = start_api_server(server_info)
        if process:
            processes.append((server_info, process))
        print()
    
    if not processes:
        print("❌ 모든 API 서버 시작에 실패했습니다.")
        sys.exit(1)
    
    # 실행 중인 서버 목록 출력
    print("=" * 60)
    print("📋 실행 중인 Mock API 서버들:")
    print("=" * 60)
    
    for server_info, process in processes:
        print(f"✅ {server_info['name']}")
        print(f"   포트: http://localhost:{server_info['port']}")
        print(f"   설명: {server_info['description']}")
        print()
    
    print("🌐 챗봇 웹 인터페이스 실행:")
    print("   streamlit run web_interface/chatbot_ui.py")
    print()
    print("🛑 서버 중지: Ctrl+C")
    print("=" * 60)
    
    try:
        # 서버들이 실행 중인 동안 대기
        while True:
            time.sleep(1)
            
            # 모든 프로세스가 살아있는지 확인
            for server_info, process in processes:
                if process.poll() is not None:
                    print(f"⚠️  {server_info['name']}이 예기치 않게 종료되었습니다.")
                    return
            
    except KeyboardInterrupt:
        print("\n🛑 서버들을 종료 중...")
        
        # 모든 프로세스 종료
        for server_info, process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ {server_info['name']} 종료 완료")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"⚠️  {server_info['name']} 강제 종료")
            except Exception as e:
                print(f"❌ {server_info['name']} 종료 중 오류: {e}")
        
        print("👋 모든 Mock API 서버가 종료되었습니다.")

if __name__ == "__main__":
    main() 