"""
Lab 4 - Mock File Manager API Server
파일 관리를 위한 Mock API 서버
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import uvicorn
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import uuid
import mimetypes

# FastAPI 앱 생성
app = FastAPI(
    title="Mock File Manager API",
    description="파일 관리를 위한 Mock API 서버",
    version="1.0.0"
)

# 데이터 모델 정의
class FileInfo(BaseModel):
    id: str
    name: str
    path: str
    size: int  # bytes
    type: str  # file, directory
    mime_type: Optional[str] = None
    created_at: str
    modified_at: str
    tags: List[str] = []
    content_preview: Optional[str] = None

class FileCreate(BaseModel):
    name: str
    content: str
    tags: List[str] = []
    directory: str = "/"

class DirectoryInfo(BaseModel):
    path: str
    files: List[FileInfo]
    total_files: int
    total_size: int

class SearchResult(BaseModel):
    files: List[FileInfo]
    total_matches: int
    query: str
    search_time_ms: int

# Mock 파일 시스템 데이터
FILE_SYSTEM = {
    "files": {
        "f001": FileInfo(
            id="f001",
            name="프로젝트_계획서.md",
            path="/documents/프로젝트_계획서.md",
            size=2048,
            type="file",
            mime_type="text/markdown",
            created_at="2024-01-15T09:00:00",
            modified_at="2024-01-20T14:30:00",
            tags=["프로젝트", "계획", "문서"],
            content_preview="# 프로젝트 계획서\n\n## 개요\n이 프로젝트는 AI 챗봇 시스템을 구축하는 것을 목표로 합니다..."
        ),
        "f002": FileInfo(
            id="f002",
            name="API_명세서.json",
            path="/api/API_명세서.json",
            size=1524,
            type="file",
            mime_type="application/json",
            created_at="2024-01-10T11:00:00",
            modified_at="2024-01-18T16:45:00",
            tags=["API", "명세", "개발"],
            content_preview='{\n  "version": "1.0.0",\n  "endpoints": [\n    {"path": "/weather", "method": "GET"}...'
        ),
        "f003": FileInfo(
            id="f003",
            name="회의록_0125.txt",
            path="/meetings/회의록_0125.txt",
            size=892,
            type="file",
            mime_type="text/plain",
            created_at="2024-01-25T15:00:00",
            modified_at="2024-01-25T15:30:00",
            tags=["회의록", "팀미팅"],
            content_preview="2024-01-25 팀 미팅\n\n참석자: 김개발, 박디자인, 이기획\n\n안건:\n1. 프로젝트 진행 상황..."
        ),
        "f004": FileInfo(
            id="f004",
            name="사용자_가이드.pdf",
            path="/documents/사용자_가이드.pdf",
            size=5120,
            type="file",
            mime_type="application/pdf",
            created_at="2024-01-12T10:00:00",
            modified_at="2024-01-19T13:20:00",
            tags=["가이드", "사용법", "문서"],
            content_preview="[PDF 미리보기] 사용자 가이드 문서입니다. 시스템 사용법과 주요 기능에 대한 설명..."
        )
    },
    "directories": {
        "/": ["documents", "api", "meetings", "assets"],
        "/documents": ["프로젝트_계획서.md", "사용자_가이드.pdf"],
        "/api": ["API_명세서.json"],
        "/meetings": ["회의록_0125.txt"],
        "/assets": []
    }
}

@app.get("/")
async def root():
    """API 루트 엔드포인트"""
    return {
        "service": "Mock File Manager API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/files",
            "/files/{file_id}",
            "/files/search",
            "/directories/{path}",
            "/files/upload",
            "/files/content/{file_id}"
        ]
    }

@app.get("/files", response_model=List[FileInfo])
async def get_all_files():
    """모든 파일 목록 조회"""
    return list(FILE_SYSTEM["files"].values())

@app.get("/files/search", response_model=SearchResult)
async def search_files(q: str, tags: Optional[str] = None, file_type: Optional[str] = None):
    """파일 검색"""
    start_time = datetime.now()
    
    matched_files = []
    query_lower = q.lower()
    
    # 검색어를 단어로 분리
    query_words = query_lower.split()
    
    for file_info in FILE_SYSTEM["files"].values():
        match_score = 0
        
        # 파일명 검색
        file_name_lower = file_info.name.lower()
        for word in query_words:
            if word in file_name_lower:
                match_score += 1
        
        # 태그 검색
        for word in query_words:
            if any(word in tag.lower() for tag in file_info.tags):
                match_score += 1
        
        # 내용 검색 (preview 기반)
        if file_info.content_preview:
            content_lower = file_info.content_preview.lower()
            for word in query_words:
                if word in content_lower:
                    match_score += 1
        
        # 하나라도 매치되면 결과에 포함
        if match_score > 0:
            matched_files.append(file_info)
    
    # 태그 필터링
    if tags:
        tag_list = [t.strip() for t in tags.split(',')]
        matched_files = [f for f in matched_files if any(tag in f.tags for tag in tag_list)]
    
    # 파일 타입 필터링
    if file_type:
        matched_files = [f for f in matched_files if file_type.lower() in f.mime_type.lower()]
    
    search_time = int((datetime.now() - start_time).total_seconds() * 1000)
    
    return SearchResult(
        files=matched_files,
        total_matches=len(matched_files),
        query=q,
        search_time_ms=search_time
    )

@app.get("/files/{file_id}", response_model=FileInfo)
async def get_file(file_id: str):
    """특정 파일 정보 조회"""
    if file_id not in FILE_SYSTEM["files"]:
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다")
    
    return FILE_SYSTEM["files"][file_id]

@app.get("/files/content/{file_id}")
async def get_file_content(file_id: str):
    """파일 내용 조회"""
    if file_id not in FILE_SYSTEM["files"]:
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다")
    
    file_info = FILE_SYSTEM["files"][file_id]
    
    # Mock 파일 내용 생성
    mock_content = generate_mock_content(file_info)
    
    return {
        "file_id": file_id,
        "name": file_info.name,
        "content": mock_content,
        "size": len(mock_content),
        "mime_type": file_info.mime_type
    }

@app.get("/directories", response_model=Dict[str, List[str]])
async def get_directory_structure():
    """전체 디렉토리 구조 조회"""
    return FILE_SYSTEM["directories"]

@app.get("/directories/{path:path}", response_model=DirectoryInfo)
async def get_directory_contents(path: str):
    """특정 디렉토리 내용 조회"""
    if not path.startswith('/'):
        path = '/' + path
    
    if path not in FILE_SYSTEM["directories"]:
        raise HTTPException(status_code=404, detail="디렉토리를 찾을 수 없습니다")
    
    # 해당 디렉토리의 파일들 찾기
    directory_files = []
    total_size = 0
    
    for file_info in FILE_SYSTEM["files"].values():
        if file_info.path.startswith(path) and file_info.path.count('/') == path.count('/') + 1:
            directory_files.append(file_info)
            total_size += file_info.size
    
    return DirectoryInfo(
        path=path,
        files=directory_files,
        total_files=len(directory_files),
        total_size=total_size
    )

@app.post("/files", response_model=FileInfo)
async def create_file(file_data: FileCreate):
    """새 파일 생성"""
    file_id = str(uuid.uuid4())[:8]
    
    # MIME 타입 추측
    mime_type, _ = mimetypes.guess_type(file_data.name)
    if not mime_type:
        mime_type = "text/plain"
    
    new_file = FileInfo(
        id=file_id,
        name=file_data.name,
        path=f"{file_data.directory.rstrip('/')}/{file_data.name}",
        size=len(file_data.content),
        type="file",
        mime_type=mime_type,
        created_at=datetime.now().isoformat(),
        modified_at=datetime.now().isoformat(),
        tags=file_data.tags,
        content_preview=file_data.content[:200] + "..." if len(file_data.content) > 200 else file_data.content
    )
    
    FILE_SYSTEM["files"][file_id] = new_file
    
    return new_file

@app.put("/files/{file_id}", response_model=FileInfo)
async def update_file(file_id: str, file_data: FileCreate):
    """파일 정보 수정"""
    if file_id not in FILE_SYSTEM["files"]:
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다")
    
    existing_file = FILE_SYSTEM["files"][file_id]
    
    updated_file = FileInfo(
        id=file_id,
        name=file_data.name,
        path=f"{file_data.directory.rstrip('/')}/{file_data.name}",
        size=len(file_data.content),
        type=existing_file.type,
        mime_type=existing_file.mime_type,
        created_at=existing_file.created_at,
        modified_at=datetime.now().isoformat(),
        tags=file_data.tags,
        content_preview=file_data.content[:200] + "..." if len(file_data.content) > 200 else file_data.content
    )
    
    FILE_SYSTEM["files"][file_id] = updated_file
    
    return updated_file

@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """파일 삭제"""
    if file_id not in FILE_SYSTEM["files"]:
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다")
    
    deleted_file = FILE_SYSTEM["files"][file_id]
    del FILE_SYSTEM["files"][file_id]
    
    return {"message": f"파일 '{deleted_file.name}'이 삭제되었습니다"}

@app.get("/files/tags/popular")
async def get_popular_tags():
    """인기 태그 목록"""
    tag_counts = {}
    
    for file_info in FILE_SYSTEM["files"].values():
        for tag in file_info.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    # 빈도순 정렬
    popular_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "popular_tags": [{"tag": tag, "count": count} for tag, count in popular_tags[:10]],
        "total_tags": len(tag_counts)
    }

@app.get("/files/stats")
async def get_file_stats():
    """파일 통계 정보"""
    total_files = len(FILE_SYSTEM["files"])
    total_size = sum(f.size for f in FILE_SYSTEM["files"].values())
    
    # 파일 타입별 분포
    type_distribution = {}
    for file_info in FILE_SYSTEM["files"].values():
        ext = file_info.name.split('.')[-1] if '.' in file_info.name else 'unknown'
        type_distribution[ext] = type_distribution.get(ext, 0) + 1
    
    return {
        "total_files": total_files,
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "file_type_distribution": type_distribution,
        "average_file_size": round(total_size / total_files if total_files > 0 else 0, 2)
    }

def generate_mock_content(file_info: FileInfo) -> str:
    """파일 타입에 따른 Mock 콘텐츠 생성"""
    if file_info.mime_type == "text/markdown":
        return f"""# {file_info.name.replace('.md', '')}

## 개요
이 문서는 {file_info.name} 파일의 Mock 콘텐츠입니다.

## 주요 내용
- 프로젝트 목표 및 범위
- 기술 스택 및 아키텍처
- 일정 및 마일스톤
- 팀 구성 및 역할

## 상세 설명
{file_info.content_preview}

생성일: {file_info.created_at}
수정일: {file_info.modified_at}
"""
    
    elif file_info.mime_type == "application/json":
        return """{
  "version": "1.0.0",
  "title": "API 명세서",
  "description": "Mock API 서버들의 엔드포인트 정의",
  "servers": [
    {"url": "http://localhost:8001", "description": "Weather API"},
    {"url": "http://localhost:8002", "description": "Calendar API"},
    {"url": "http://localhost:8003", "description": "File Manager API"}
  ],
  "endpoints": [
    {"path": "/weather/{city}", "method": "GET", "description": "날씨 정보 조회"},
    {"path": "/calendar/today", "method": "GET", "description": "오늘 일정 조회"}
  ]
}"""
    
    elif file_info.mime_type == "text/plain":
        return f"""회의록: {file_info.name}

작성일: {file_info.created_at}

참석자:
- 김개발 (백엔드 개발자)
- 박디자인 (UI/UX 디자이너)  
- 이기획 (프로덕트 매니저)

주요 안건:
1. 프로젝트 진행 상황 점검
2. 다음 스프린트 계획 수립
3. 기술적 이슈 및 해결 방안 논의

결정 사항:
- Mock API 서버 구축 완료
- Multi-Agent 시스템 설계 시작
- 다음 주 프로토타입 데모 예정

액션 아이템:
- 김개발: Agent 시스템 개발 (마감: 다음 주 금요일)
- 박디자인: UI 프로토타입 제작 (마감: 이번 주 수요일)
- 이기획: 요구사항 명세서 업데이트 (마감: 내일)
"""
    
    else:
        return f"[{file_info.mime_type}] {file_info.name} 파일의 Mock 콘텐츠입니다.\n\n{file_info.content_preview}"

# 서버 실행 함수
def run_server():
    """File Manager API 서버 실행"""
    print(" File Manager API 서버 시작 중...")
    print(" URL: http://localhost:8003")
    print(" API 문서: http://localhost:8003/docs")
    
    uvicorn.run(
        "file_manager_api:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    run_server() 