"""
Lab 4 - Database Mock API
데이터 조회 및 분석을 위한 Mock Database API 서버
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import json
import os
import uuid
from datetime import datetime, timedelta
import random

app = FastAPI(
    title="Database Mock API",
    description="데이터 조회 및 분석을 위한 Mock Database API",
    version="1.0.0"
)

# 데이터 모델 정의
class QueryRequest(BaseModel):
    query: str
    table: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    limit: Optional[int] = 100

class QueryResult(BaseModel):
    status: str
    message: str
    data: List[Dict[str, Any]]
    total_rows: int
    execution_time: float

class TableInfo(BaseModel):
    table_name: str
    columns: List[str]
    row_count: int
    description: str

# Mock 데이터베이스 테이블들
mock_tables = {
    "users": {
        "description": "사용자 정보 테이블",
        "columns": ["user_id", "name", "email", "department", "created_at", "last_login"],
        "data": [
            {"user_id": 1, "name": "김철수", "email": "kim@company.com", "department": "개발팀", "created_at": "2024-01-15", "last_login": "2024-12-15"},
            {"user_id": 2, "name": "이영희", "email": "lee@company.com", "department": "마케팅", "created_at": "2024-02-20", "last_login": "2024-12-14"},
            {"user_id": 3, "name": "박민수", "email": "park@company.com", "department": "개발팀", "created_at": "2024-03-10", "last_login": "2024-12-13"},
            {"user_id": 4, "name": "최정연", "email": "choi@company.com", "department": "디자인", "created_at": "2024-04-05", "last_login": "2024-12-12"},
            {"user_id": 5, "name": "홍길동", "email": "hong@company.com", "department": "영업팀", "created_at": "2024-05-12", "last_login": "2024-12-11"}
        ]
    },
    "projects": {
        "description": "프로젝트 정보 테이블",
        "columns": ["project_id", "name", "status", "start_date", "end_date", "manager", "budget"],
        "data": [
            {"project_id": 1, "name": "AI 챗봇 개발", "status": "진행중", "start_date": "2024-10-01", "end_date": "2024-12-31", "manager": "김철수", "budget": 5000000},
            {"project_id": 2, "name": "웹사이트 리뉴얼", "status": "완료", "start_date": "2024-08-01", "end_date": "2024-11-30", "manager": "이영희", "budget": 3000000},
            {"project_id": 3, "name": "모바일 앱 개발", "status": "계획", "start_date": "2025-01-15", "end_date": "2025-06-30", "manager": "박민수", "budget": 8000000},
            {"project_id": 4, "name": "데이터 분석 플랫폼", "status": "진행중", "start_date": "2024-11-01", "end_date": "2025-03-31", "manager": "최정연", "budget": 6000000}
        ]
    },
    "sales": {
        "description": "매출 데이터 테이블",
        "columns": ["sale_id", "product", "amount", "customer", "sale_date", "region"],
        "data": [
            {"sale_id": 1, "product": "프리미엄 서비스", "amount": 1200000, "customer": "A회사", "sale_date": "2024-12-01", "region": "서울"},
            {"sale_id": 2, "product": "기본 서비스", "amount": 800000, "customer": "B회사", "sale_date": "2024-12-02", "region": "부산"},
            {"sale_id": 3, "product": "프리미엄 서비스", "amount": 1200000, "customer": "C회사", "sale_date": "2024-12-03", "region": "대구"},
            {"sale_id": 4, "product": "엔터프라이즈", "amount": 2500000, "customer": "D회사", "sale_date": "2024-12-04", "region": "서울"},
            {"sale_id": 5, "product": "기본 서비스", "amount": 800000, "customer": "E회사", "sale_date": "2024-12-05", "region": "인천"}
        ]
    },
    "logs": {
        "description": "시스템 로그 테이블",
        "columns": ["log_id", "timestamp", "level", "message", "user_id", "ip_address"],
        "data": [
            {"log_id": 1, "timestamp": "2024-12-15 10:30:00", "level": "INFO", "message": "사용자 로그인", "user_id": 1, "ip_address": "192.168.1.100"},
            {"log_id": 2, "timestamp": "2024-12-15 10:31:15", "level": "INFO", "message": "파일 업로드", "user_id": 1, "ip_address": "192.168.1.100"},
            {"log_id": 3, "timestamp": "2024-12-15 10:35:22", "level": "WARN", "message": "로그인 실패", "user_id": None, "ip_address": "192.168.1.200"},
            {"log_id": 4, "timestamp": "2024-12-15 10:40:10", "level": "ERROR", "message": "데이터베이스 연결 실패", "user_id": 2, "ip_address": "192.168.1.150"},
            {"log_id": 5, "timestamp": "2024-12-15 10:45:33", "level": "INFO", "message": "보고서 생성", "user_id": 3, "ip_address": "192.168.1.120"}
        ]
    }
}

@app.get("/", summary="Database API 정보")
async def get_api_info():
    """Database API 기본 정보"""
    return {
        "service": "Database Mock API",
        "version": "1.0.0",
        "description": "데이터 조회 및 분석을 위한 Mock Database API",
        "endpoints": {
            "/tables": "사용 가능한 테이블 목록 조회",
            "/tables/{table_name}": "특정 테이블 정보 조회",
            "/query": "SQL 스타일 데이터 조회",
            "/analytics/{table_name}": "테이블 분석 데이터",
            "/export/{table_name}": "테이블 데이터 내보내기"
        },
        "available_tables": list(mock_tables.keys()),
        "status": "운영중",
        "last_updated": datetime.now().isoformat()
    }

@app.get("/tables", response_model=List[TableInfo], summary="테이블 목록 조회")
async def get_tables():
    """사용 가능한 테이블 목록 조회"""
    tables = []
    
    for table_name, table_data in mock_tables.items():
        tables.append(TableInfo(
            table_name=table_name,
            columns=table_data["columns"],
            row_count=len(table_data["data"]),
            description=table_data["description"]
        ))
    
    return tables

@app.get("/tables/{table_name}", summary="테이블 정보 조회")
async def get_table_info(table_name: str):
    """특정 테이블의 상세 정보 조회"""
    if table_name not in mock_tables:
        raise HTTPException(status_code=404, detail=f"테이블 '{table_name}'을 찾을 수 없습니다")
    
    table_data = mock_tables[table_name]
    
    return {
        "table_name": table_name,
        "description": table_data["description"],
        "columns": table_data["columns"],
        "row_count": len(table_data["data"]),
        "sample_data": table_data["data"][:3],  # 샘플 데이터 3개
        "created_at": "2024-01-01T00:00:00",
        "last_updated": datetime.now().isoformat()
    }

@app.post("/query", response_model=QueryResult, summary="데이터 조회")
async def execute_query(query_request: QueryRequest):
    """SQL 스타일 쿼리로 데이터 조회"""
    start_time = datetime.now()
    
    query = query_request.query.lower()
    table_name = query_request.table
    filters = query_request.filters or {}
    limit = min(query_request.limit or 100, 1000)  # 최대 1000개로 제한
    
    # 간단한 쿼리 파싱
    if "select" in query and "from" in query:
        # FROM 절에서 테이블 이름 추출
        try:
            from_index = query.index("from")
            table_part = query[from_index + 4:].strip().split()[0]
            if table_part in mock_tables:
                table_name = table_part
        except:
            pass
    
    # 테이블 지정이 없으면 기본 테이블 사용
    if not table_name:
        if "user" in query or "사용자" in query:
            table_name = "users"
        elif "project" in query or "프로젝트" in query:
            table_name = "projects"
        elif "sale" in query or "매출" in query:
            table_name = "sales"
        elif "log" in query or "로그" in query:
            table_name = "logs"
        else:
            table_name = "users"  # 기본값
    
    if table_name not in mock_tables:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 테이블: {table_name}")
    
    # 데이터 가져오기
    data = mock_tables[table_name]["data"].copy()
    
    # 필터 적용
    if filters:
        filtered_data = []
        for row in data:
            match = True
            for key, value in filters.items():
                if key in row:
                    if isinstance(value, str) and value.lower() not in str(row[key]).lower():
                        match = False
                        break
                    elif isinstance(value, (int, float)) and row[key] != value:
                        match = False
                        break
            if match:
                filtered_data.append(row)
        data = filtered_data
    
    # 키워드 필터링 (간단한 검색)
    if "where" in query:
        keywords = []
        # WHERE 절 파싱 (매우 간단한 버전)
        where_part = query.split("where")[1] if "where" in query else ""
        if where_part:
            keywords = [w.strip("'\"") for w in where_part.split() if len(w) > 2]
        
        if keywords:
            filtered_data = []
            for row in data:
                for keyword in keywords:
                    if any(keyword.lower() in str(value).lower() for value in row.values()):
                        filtered_data.append(row)
                        break
            data = filtered_data
    
    # 제한 적용
    total_rows = len(data)
    data = data[:limit]
    
    # 실행 시간 계산
    execution_time = (datetime.now() - start_time).total_seconds()
    
    return QueryResult(
        status="success",
        message=f"쿼리 실행 완료: {len(data)}개 행 반환",
        data=data,
        total_rows=total_rows,
        execution_time=round(execution_time, 3)
    )

@app.get("/analytics/{table_name}", summary="테이블 분석")
async def get_table_analytics(table_name: str):
    """테이블 데이터 분석 정보"""
    if table_name not in mock_tables:
        raise HTTPException(status_code=404, detail=f"테이블 '{table_name}'을 찾을 수 없습니다")
    
    data = mock_tables[table_name]["data"]
    
    analytics = {
        "table_name": table_name,
        "total_rows": len(data),
        "columns": mock_tables[table_name]["columns"],
        "analysis": {}
    }
    
    # 컬럼별 분석
    for column in mock_tables[table_name]["columns"]:
        values = [row.get(column) for row in data if row.get(column) is not None]
        
        column_analysis = {
            "type": "unknown",
            "null_count": len(data) - len(values),
            "unique_count": len(set(str(v) for v in values))
        }
        
        # 데이터 타입 분석
        if values:
            sample_value = values[0]
            if isinstance(sample_value, (int, float)):
                column_analysis["type"] = "numeric"
                column_analysis["min"] = min(values)
                column_analysis["max"] = max(values)
                column_analysis["avg"] = sum(values) / len(values)
            elif isinstance(sample_value, str):
                column_analysis["type"] = "text"
                column_analysis["avg_length"] = sum(len(str(v)) for v in values) / len(values)
                
                # 가장 빈번한 값들
                from collections import Counter
                counter = Counter(values)
                column_analysis["top_values"] = dict(counter.most_common(5))
        
        analytics["analysis"][column] = column_analysis
    
    return analytics

@app.get("/export/{table_name}", summary="테이블 데이터 내보내기")
async def export_table_data(table_name: str, 
                           format: str = Query("json", description="내보내기 형식 (json, csv)")):
    """테이블 데이터를 지정된 형식으로 내보내기"""
    if table_name not in mock_tables:
        raise HTTPException(status_code=404, detail=f"테이블 '{table_name}'을 찾을 수 없습니다")
    
    data = mock_tables[table_name]["data"]
    
    if format.lower() == "csv":
        # CSV 형식으로 변환
        columns = mock_tables[table_name]["columns"]
        csv_content = ",".join(columns) + "\n"
        
        for row in data:
            csv_row = ",".join(str(row.get(col, "")) for col in columns)
            csv_content += csv_row + "\n"
        
        return {
            "format": "csv",
            "content": csv_content,
            "filename": f"{table_name}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        }
    
    else:  # JSON 형식 (기본값)
        return {
            "format": "json",
            "content": data,
            "metadata": {
                "table_name": table_name,
                "total_rows": len(data),
                "export_timestamp": datetime.now().isoformat(),
                "columns": mock_tables[table_name]["columns"]
            }
        }

@app.get("/search", summary="전체 데이터 검색")
async def search_data(q: str = Query(..., description="검색 키워드"),
                     tables: Optional[str] = Query(None, description="검색할 테이블 (쉼표로 구분)")):
    """모든 테이블에서 키워드 검색"""
    search_keyword = q.lower()
    search_tables = tables.split(",") if tables else list(mock_tables.keys())
    
    results = {}
    
    for table_name in search_tables:
        if table_name not in mock_tables:
            continue
        
        table_results = []
        data = mock_tables[table_name]["data"]
        
        for row in data:
            # 모든 필드에서 키워드 검색
            if any(search_keyword in str(value).lower() for value in row.values()):
                table_results.append(row)
        
        if table_results:
            results[table_name] = {
                "count": len(table_results),
                "data": table_results[:10]  # 최대 10개만 반환
            }
    
    return {
        "search_keyword": q,
        "total_tables_searched": len(search_tables),
        "tables_with_results": len(results),
        "results": results
    }

@app.get("/stats", summary="데이터베이스 통계")
async def get_database_stats():
    """전체 데이터베이스 통계 정보"""
    total_rows = sum(len(table_data["data"]) for table_data in mock_tables.values())
    total_columns = sum(len(table_data["columns"]) for table_data in mock_tables.values())
    
    table_stats = {}
    for table_name, table_data in mock_tables.items():
        table_stats[table_name] = {
            "rows": len(table_data["data"]),
            "columns": len(table_data["columns"]),
            "description": table_data["description"]
        }
    
    return {
        "database_name": "Mock Database",
        "total_tables": len(mock_tables),
        "total_rows": total_rows,
        "total_columns": total_columns,
        "table_statistics": table_stats,
        "last_updated": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    print("  Database Mock API 서버 시작...")
    print(" 사용 가능한 테이블:", list(mock_tables.keys()))
    print(" API 문서: http://localhost:8005/docs")
    uvicorn.run(app, host="0.0.0.0", port=8005) 