"""Pydantic models for ARI API requests and responses"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str = Field(..., description="Health status")
    mcp_connected: bool = Field(..., description="Whether MCP client is connected")
    tools_available: int = Field(..., description="Number of available tools")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional health details")

class TaskStatus(str, Enum):
    """Task status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# ARI API Models
class StructuredTableRow(BaseModel):
    """구조화된 테이블 행 데이터"""
    data: Dict[str, str] = Field(..., description="컬럼명: 값 매핑")

class StructuredTable(BaseModel):
    """구조화된 테이블 모델"""
    table_name: str = Field(..., description="테이블 이름")
    columns: List[str] = Field(..., description="컬럼 목록")
    rows: List[StructuredTableRow] = Field(..., description="테이블 행 데이터")
    is_merged: bool = Field(False, description="병합된 셀이 있는지 여부")

class AriCrawlResult(BaseModel):
    """ARI crawling result model"""
    title: Optional[str] = Field(None, description="Page title")
    breadcrumbs: Optional[List[Dict[str, str]]] = Field(None, description="Breadcrumb navigation")
    content: Dict[str, Any] = Field(
        ..., description="Extracted content data (e.g., contents array)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Extracted metadata (img, urls, pagetree, etc.)"
    )

class AriCrawlResponse(BaseModel):
    """Response model for ARI crawling endpoint (Task-like schema)"""
    taskId: str = Field(..., description="Task identifier")
    status: TaskStatus = Field(..., description="Task status")
    result: List[AriCrawlResult] = Field(..., description="List of extracted results per file")
    error: Optional[str] = Field(None, description="Error message if failed")
    createdAt: str = Field(..., description="Task creation timestamp")
    completedAt: Optional[str] = Field(None, description="Task completion timestamp")
    message: str = Field(..., description="Processing result message")
    total_files: int = Field(..., description="Number of processed files")
    total_size: int = Field(..., description="Total file size in bytes")

