"""API routes for ARI Processing"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import Response, FileResponse
import tempfile
import os
import logging
import json
from typing import List
from datetime import datetime

from pydantic import BaseModel, Field
from app.models import (
    HealthResponse, TaskStatus, AriCrawlResponse
)
from app.infrastructure.mcp.mcp_service import mcp_service
from app.infrastructure.llm.llm_service import llm_service
from app.application.ari.ari_service import ari_service

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()

@router.get("/", tags=["root"])
async def root():
    """Root endpoint"""
    return {"message": "ARI Processing Server is running"}

@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint"""
    try:
        health_data = await mcp_service.health_check()
        
        return HealthResponse(
            status="healthy" if health_data["connected"] else "unhealthy",
            mcp_connected=health_data["connected"],
            tools_available=health_data["tools_available"],
            details=health_data
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="error",
            mcp_connected=False,
            tools_available=0,
            details={"error": str(e)}
        )

# === ARI HTML Processing Endpoints ===

@router.post("/ari/crawl", response_model=AriCrawlResponse, tags=["ari"])
async def ari_crawl_endpoint(
    files: List[UploadFile] = File(..., description="HTML 파일들 (복수 파일 지원)")
):
    """Process HTML files from ARI for RAG conversion"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="업로드할 HTML 파일이 없습니다")
        
        # 통합된 HTML 파일 처리 (마크다운 + 구조화된 JSON까지)
        result = await ari_service.process_html_files_complete(files)
        if not result['success']:
            raise HTTPException(status_code=500, detail=result['message'])

        # 새로운 구조로 응답 데이터 구성
        structured_results = []
        for info in result['processed_files']:
            processed_data = info.get('processed_data', {})
            metadata = processed_data.get('metadata', {})
            
            structured_results.append({
                'title': processed_data.get('title', ''),
                'breadcrumbs': processed_data.get('breadcrumbs', []),
                'content': {
                    'contents': info['contents']
                },
                'metadata': {
                    'img': metadata.get('img', []),
                    'urls': metadata.get('urls', []),
                    'pagetree': metadata.get('pagetree', []),
                    'content_length': metadata.get('content_length', 0),
                    'extracted_at': metadata.get('extracted_at', ''),
                    'markdown_length': metadata.get('markdown_length', 0),
                    'contents_count': metadata.get('contents_count', 0)
                }
            })

        response_data = AriCrawlResponse(
            taskId=f"ari_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            status=TaskStatus.COMPLETED,
            result=structured_results,
            error=None,
            createdAt=result['processed_files'][0]['upload_time'] if result['processed_files'] else datetime.now().isoformat(),
            completedAt=datetime.now().isoformat(),
            message=result['message'],
            total_files=result['total_files'],
            total_size=result['total_size']
        )
        
        logger.info(f"✅ ARI HTML 완전 처리 완료: {result['total_files']}개 파일, {result['total_size']} bytes")
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ARI HTML 처리 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"ARI HTML 처리 중 오류가 발생했습니다: {str(e)}")


class AriProcessRequest(BaseModel):
    """HTML 콘텐츠 기반 ARI 처리 요청 (내부망: URL 크롤링 비사용)"""
    htmls: List[str] = Field(..., description="직접 전달할 HTML 콘텐츠 목록")


@router.post("/ari/process", tags=["ari"])
async def ari_process_endpoint(request: AriProcessRequest):
    """
    HTML 콘텐츠를 받아 Confluence HTML을 전처리/파싱하여 주요 내용만 추출합니다.
    - htmls가 제공되면 그대로 처리
    반환: { success, processed, total_inputs, results: [processed_data...], errors: [...] }
    """
    try:
        errors: List[dict] = []
        candidate_htmls: List[str] = []
        
        # HTML 직접 전달만 처리
        for html in request.htmls:
            if isinstance(html, str) and html.strip():
                candidate_htmls.append(html)

        if not candidate_htmls:
            raise HTTPException(status_code=400, detail="처리할 HTML이 비어있습니다")

        # ARI 파이프라인 수행: HTML -> Markdown -> JSON(contents)
        results: List[dict] = []
        for html in candidate_htmls:
            try:
                # 통합된 처리: 마크다운 + 구조화된 JSON
                md = ari_service.extract_markdown(html)
                json_result = ari_service.ari_markdown_to_json(md)
                contents = json_result.get('contents', []) if json_result.get('success') else []
                if not contents:
                    contents = [{"id": 1, "type": "text", "title": "", "data": md}]

                results.append({
                    'content': {
                        'contents': contents
                    }
                })
            except Exception as pe:
                logger.error(f"ARI process item 실패: {pe}")
                errors.append({"error": f"parse failed: {str(pe)}"})

        return {
            "success": True,
            "total_inputs": len(candidate_htmls),
            "processed": len(results),
            "results": results,
            "errors": errors,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ARI process failed: {e}")
        raise HTTPException(status_code=500, detail=f"ARI 처리 실패: {str(e)}")


@router.post("/ari/markdown", tags=["ari"])
async def ari_markdown_endpoint(
    files: List[UploadFile] = File(..., description="HTML 파일들 (복수 파일 지원)")
):
    """
    HTML을 받아 마크다운으로 변환하여 반환합니다.
    - 반환: 마크다운 텍스트 (text/markdown)
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="업로드할 HTML 파일이 없습니다")
        
        fragments: List[str] = []
        for file in files:
            if not file.filename.endswith('.html'):
                continue
            content = await file.read()
            html = content.decode('utf-8', errors='ignore')
            md = ari_service.extract_markdown(html)
            fragments.append(md)
        
        final_md = "\n\n".join(fragments)
        return Response(content=final_md, media_type="text/markdown; charset=utf-8")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ARI markdown failed: {e}")
        raise HTTPException(status_code=500, detail=f"ARI 마크다운 생성 실패: {str(e)}")

# === LLM Query Endpoints ===

class LLMQueryRequest(BaseModel):
    question: str = Field(..., description="사용자 질문")

class LLMQueryResponse(BaseModel):
    success: bool = Field(..., description="처리 성공 여부")
    answer: str = Field(..., description="LLM 응답")
    tools_used: List[str] = Field(default=[], description="사용된 도구 목록")
    error: str = Field(default="", description="오류 메시지")

@router.post("/llm/query", response_model=LLMQueryResponse, tags=["llm"])
async def llm_query_endpoint(request: LLMQueryRequest):
    """
    LLM을 사용하여 사용자 질문에 답변 (텍스트만)
    - 의도 분류를 통해 적절한 MCP 도구를 선택하고 실행
    - 도구 실행 결과를 바탕으로 최종 답변 생성
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="질문이 비어있습니다")
        
        # MCP 서비스에서 사용 가능한 도구 목록 가져오기
        available_tools = mcp_service.available_tools
        
        if not available_tools:
            logger.warning("사용 가능한 MCP 도구가 없습니다")
        
        # LLM 쿼리 실행
        answer = await llm_service.query(request.question, available_tools)
        
        return LLMQueryResponse(
            success=True,
            answer=answer,
            tools_used=[],  # 실제 사용된 도구는 로그에서 확인 가능
            error=""
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        return LLMQueryResponse(
            success=False,
            answer="",
            tools_used=[],
            error=f"LLM 쿼리 실행 중 오류가 발생했습니다: {str(e)}"
        )

# === 통합 LLM + 파일 처리 엔드포인트 ===

@router.post("/llm/query-with-files", response_model=LLMQueryResponse, tags=["llm"])
async def llm_query_with_files_endpoint(
    question: str = Form(..., description="사용자 질문"),
    files: List[UploadFile] = File(default=[], description="HTML 파일들 (선택사항)")
):
    """
    파일과 함께 LLM 질문 처리 (새로운 플로우)
    1. API 진입
    2. 질문과 파일 수신
    3. 질문이 html 내용을 추출해줘와 업로드된 파일이 html이라면 의도를 파악하여 도구 조회
    4. mcp-server에 있는 html을 마크다운으로 변환 도구 호출
    5. 마크다운 데이터를 처리
    6. 처리된 결과를 응답 -> 응답은 모델에게 주지않고 사용자는 받은 응답 결과만 봄
    
    모델의 역할: 의도를 분류하여 mcp-server에 있는 도구만 호출
    """
    try:
        if not question.strip():
            raise HTTPException(status_code=400, detail="질문이 비어있습니다")
        
        # 1. 파일 정보 확인 (전처리 없이 메타데이터만)
        file_info = []
        html_content = None
        
        if files and len(files) > 0:
            # HTML 파일들만 필터링
            html_files = [f for f in files if f.filename and f.filename.endswith('.html')]
            
            if html_files:
                logger.info(f"📁 {len(html_files)}개 HTML 파일 감지")
                
                # 첫 번째 HTML 파일만 처리 (단순화)
                first_file = html_files[0]
                try:
                    content = await first_file.read()
                    html_content = content.decode('utf-8', errors='ignore')
                    
                    file_info.append({
                        "filename": first_file.filename,
                        "content_length": len(html_content),
                        "file_type": "HTML"
                    })
                    
                    logger.info(f"📄 파일 정보: {first_file.filename} ({len(html_content)} bytes)")
                    
                except Exception as e:
                    logger.error(f"파일 {first_file.filename} 읽기 실패: {e}")
                    file_info.append({
                        "filename": first_file.filename,
                        "error": str(e)
                    })
        
        # 2. 의도 분류를 위한 질문 구성
        enhanced_question = question
        if file_info and html_content:
            enhanced_question = f"""사용자 질문: {question}

파일 업로드 정보:
- 업로드된 파일: {file_info[0]['filename']}
- 파일 타입: HTML
- 파일 크기: {file_info[0]['content_length']} bytes

HTML 파일이 업로드되었습니다. 사용자의 질문을 분석하여 적절한 도구를 호출해주세요."""
        
        # 3. MCP 도구 목록 가져오기
        available_tools = mcp_service.available_tools
        
        if not available_tools:
            logger.warning("사용 가능한 MCP 도구가 없습니다")
            return LLMQueryResponse(
                success=False,
                answer="",
                tools_used=[],
                error="사용 가능한 MCP 도구가 없습니다"
            )
        
        # 4. 모델이 의도 분류하여 도구 호출 (파일 내용은 도구 호출 시 전달)
        if html_content:
            # HTML 파일이 있는 경우: 모델이 의도 분류 후 도구 호출
            logger.info("🔧 HTML 파일이 있으므로 모델이 의도 분류 후 도구 호출")
            
            # MCP 서비스 상태 사전 확인
            logger.info(f"🔍 MCP 서비스 연결 상태: {mcp_service.is_connected}")
            logger.info(f"🔍 사용 가능한 도구 수: {len(available_tools)}")
            
            # 모델이 의도 분류하고 도구 호출 (실제 HTML 내용은 도구 호출 시 전달)
            answer, tools_used = await llm_service.query_with_raw_result_and_html(enhanced_question, available_tools, html_content)
            
            return LLMQueryResponse(
                success=True,
                answer=answer,
                tools_used=tools_used,
                error=""
            )
        else:
            # 파일이 없는 경우: 일반 질문 처리
            logger.info("💬 파일이 없으므로 일반 질문 처리")
            answer = await llm_service.query_with_raw_result(question, available_tools)
            
            return LLMQueryResponse(
                success=True,
                answer=answer,
                tools_used=["general_query"],
                error=""
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LLM query with files failed: {e}")
        return LLMQueryResponse(
            success=False,
            answer="",
            tools_used=[],
            error=f"파일 포함 LLM 쿼리 실행 중 오류가 발생했습니다: {str(e)}"
        )

# === MCP 도구 사용 엔드포인트 (선택사항) ===

@router.post("/llm/query-with-tools", response_model=LLMQueryResponse, tags=["llm"])
async def llm_query_with_tools_endpoint(request: LLMQueryRequest):
    """
    MCP 도구를 사용한 LLM 질문 처리
    - 시스템 상태 확인 등 MCP 도구가 필요한 경우
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="질문이 비어있습니다")
        
        # MCP 서비스에서 사용 가능한 도구 목록 가져오기
        available_tools = mcp_service.available_tools
        
        if not available_tools:
            logger.warning("사용 가능한 MCP 도구가 없습니다")
            # 도구가 없으면 일반 대화로 폴백
            answer = await llm_service.generate_response(request.question)
        else:
            # 모든 도구를 사용하여 LLM이 적절히 선택
            logger.info("🔧 MCP 도구를 사용한 질문 처리")
            answer = await llm_service.query(request.question, available_tools)
        
        return LLMQueryResponse(
            success=True,
            answer=answer,
            tools_used=[],
            error=""
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MCP 도구 사용 LLM query failed: {e}")
        return LLMQueryResponse(
            success=False,
            answer="",
            tools_used=[],
            error=f"MCP 도구 사용 LLM 쿼리 실행 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/llm/query-with-files-download", tags=["llm"])
async def llm_query_with_files_download_endpoint(
    background_tasks: BackgroundTasks,
    question: str = Form(..., description="사용자 질문"),
    files: List[UploadFile] = File(default=[], description="HTML 파일들 (선택사항)")
):
    """
    파일과 함께 LLM 질문 처리 후 JSON 파일로 다운로드
    - 기존 query-with-files와 동일한 처리 과정
    - 응답을 JSON 파일로 다운로드 제공
    - 대용량 콘텐츠 처리에 적합
    """
    try:
        if not question.strip():
            raise HTTPException(status_code=400, detail="질문이 비어있습니다")
        
        # 1. 파일 정보 확인 (전처리 없이 메타데이터만)
        file_info = []
        html_content = None
        
        if files and len(files) > 0:
            # HTML 파일들만 필터링
            html_files = [f for f in files if f.filename and f.filename.endswith('.html')]
            
            if html_files:
                logger.info(f"📁 {len(html_files)}개 HTML 파일 감지")
                
                # 첫 번째 HTML 파일만 처리 (단순화)
                first_file = html_files[0]
                try:
                    content = await first_file.read()
                    html_content = content.decode('utf-8', errors='ignore')
                    
                    file_info.append({
                        "filename": first_file.filename,
                        "content_length": len(html_content),
                        "file_type": "HTML"
                    })
                    
                    logger.info(f"📄 파일 정보: {first_file.filename} ({len(html_content)} bytes)")
                    
                except Exception as e:
                    logger.error(f"파일 {first_file.filename} 읽기 실패: {e}")
                    file_info.append({
                        "filename": first_file.filename,
                        "error": str(e)
                    })
        
        # 2. LLM 서비스 호출
        if html_content:
            # HTML 파일이 있는 경우 - 새로운 플로우 사용
            logger.info("🔄 HTML 파일과 함께 LLM 쿼리 실행 (새로운 플로우)")
            
            # MCP 도구 목록 가져오기
            available_tools = mcp_service.available_tools
            
            # LLM에 질문과 HTML 콘텐츠 전달 (새로운 메서드)
            answer, tools_used = await llm_service.query_with_raw_result_and_html(
                question=question,
                available_tools=available_tools,
                html_content=html_content
            )
            
            # 응답 데이터 구성
            response_data = {
                "success": True,
                "question": question,
                "file_info": file_info,
                "answer": answer,
                "tools_used": tools_used,
                "error": "",
                "processed_at": datetime.now().isoformat(),
                "content_length": len(answer),
                "download_type": "json_file"
            }
            
        else:
            # HTML 파일이 없는 경우 - 기본 플로우
            logger.info("🔄 일반 LLM 쿼리 실행")
            
            # MCP 도구 목록 가져오기
            available_tools = mcp_service.available_tools
            
            # LLM에 질문 전달
            answer = await llm_service.query(question, available_tools)
            tools_used = ["general_query"]  # 일반 쿼리의 경우
            
            # 응답 데이터 구성
            response_data = {
                "success": True,
                "question": question,
                "file_info": file_info,
                "answer": answer,
                "tools_used": tools_used,
                "error": "",
                "processed_at": datetime.now().isoformat(),
                "content_length": len(answer),
                "download_type": "json_file"
            }
        
        # 3. 임시 JSON 파일 생성
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as temp_file:
            json.dump(response_data, temp_file, ensure_ascii=False, indent=2)
            temp_file_path = temp_file.name
        
        # 4. 파일명 생성 (타임스탬프 포함)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"html_content_extracted_{timestamp}.json"
        
        logger.info(f"📥 JSON 파일 다운로드 준비 완료: {filename} ({len(json.dumps(response_data, ensure_ascii=False)):,} bytes)")
        
        # 5. 파일 응답 반환 (다운로드 후 임시 파일 자동 삭제)
        def cleanup_temp_file():
            try:
                os.unlink(temp_file_path)
                logger.info(f"🗑️ 임시 파일 삭제 완료: {temp_file_path}")
            except Exception as e:
                logger.warning(f"임시 파일 삭제 실패: {e}")
        
        # BackgroundTasks에 정리 작업 추가
        background_tasks.add_task(cleanup_temp_file)
        
        return FileResponse(
            path=temp_file_path,
            filename=filename,
            media_type='application/json'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"파일 다운로드 처리 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"파일 다운로드 처리 중 오류가 발생했습니다: {str(e)}")

