"""LLM Service - Handles OpenAI API interactions"""
from typing import List, Dict, Any, AsyncGenerator, Optional
import json
import logging
import asyncio
from datetime import datetime
import tiktoken
import numpy as np
from openai import AsyncOpenAI
from app.config import settings
from app.infrastructure.mcp.mcp_service import mcp_service
from app.exceptions.base import LLMQueryError

logger = logging.getLogger(__name__)

# 임베딩 모델 사용하지 않음 - OpenAI API만 사용
EMBEDDING_AVAILABLE = False
logger.info("임베딩 모델 사용 안함 - OpenAI API만으로 의도 분류")

class LLMService:
    """Service class for managing LLM interactions"""
    
    def __init__(self):
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # 간단한 키워드 기반 의도 분류만 사용
        self._intent_examples = self._get_intent_examples()
        
    
    def _get_intent_examples(self) -> Dict[str, List[str]]:
        """의도별 예시 문장들"""
        return {
            'web_crawling': [
                "웹사이트를 크롤링해주세요",
                "이 URL의 내용을 가져와주세요",
                "페이지를 스크래핑해서 분석해주세요",
                "사이트 데이터를 수집해주세요",
                "웹페이지 정보를 추출해주세요"
            ],
            'system_status': [
                "시스템 상태를 확인해주세요",
                "서버가 정상 작동하는지 체크해주세요",
                "헬스체크를 실행해주세요",
                "현재 시스템 통계를 보여주세요",
                "서비스 상태를 알려주세요"
            ],
            'text_analysis': [
                "이 텍스트를 요약해주세요",
                "내용을 분석해서 정리해주세요",
                "문서의 핵심 내용을 추출해주세요",
                "텍스트에서 주요 정보를 찾아주세요",
                "내용을 간단히 요약해주세요"
            ],
            'html_processing': [
                "HTML 파일을 처리해주세요",
                "컨플루언스 문서를 변환해주세요",
                "업로드한 파일을 분석해주세요",
                "HTML에서 메인 컨텐츠를 추출해주세요",
                "ARI 파일을 파싱해주세요"
            ],
            'db_query': [
                "메뉴 정보를 조회해주세요",
                "링크 데이터를 찾아주세요",
                "담당자 정보를 검색해주세요",
                "데이터베이스에서 정보를 가져와주세요",
                "매니저 목록을 보여주세요"
            ],
            'database_schema': [
                "메뉴 테이블 구조를 보여주세요",
                "데이터베이스 스키마를 알려주세요",
                "컬럼명을 보여주세요",
                "테이블 필드를 알려주세요",
                "메뉴 컬럼 정보를 확인해주세요",
                "데이터 구조를 설명해주세요"
            ],
            'rag_query': [
                "문서에서 검색해주세요",
                "지식베이스에서 찾아주세요",
                "RAG로 질문에 답변해주세요",
                "등록된 문서에서 정보를 찾아주세요",
                "벡터 검색을 실행해주세요"
            ]
        }
    
    
    
    def _format_tools_for_openai(self, available_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert MCP tools format to OpenAI tools format
        
        MCP format might be: {"name": "tool_name", "description": "...", "parameters": {...}}
        OpenAI expects: {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
        """
        formatted_tools = []
        for tool in available_tools:
            # Check if already in OpenAI format
            if "type" in tool and tool["type"] == "function" and "function" in tool:
                formatted_tools.append(tool)
            else:
                # Convert from MCP format to OpenAI format
                formatted_tool = {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {
                            "type": "object",
                            "properties": {},
                            "required": []
                        })
                    }
                }
                formatted_tools.append(formatted_tool)
        return formatted_tools
    
    async def query(self, question: str, available_tools: List[Dict[str, Any]]) -> str:
        """Execute a query against the LLM with available tools"""
        try:
            # 간단한 로직: 전달받은 도구를 그대로 사용
            formatted_tools = self._format_tools_for_openai(available_tools)
            
            # Log for debugging
            logger.info(f"🔍 도구 준비 완료: {len(formatted_tools)}개 도구")
            if formatted_tools:
                logger.info(f"🔧 사용할 도구: {[t['function']['name'] for t in formatted_tools]}")
            
            tool_catalog = self._get_tool_categories_description(formatted_tools)
            system_prompt = f"""You are an AI assistant specialized in web analysis and data processing.

CRITICAL: Always use tools when available. Execute the appropriate tools to perform real work; avoid generic explanations without using tools.

Available tools (dynamic):
{tool_catalog}

Tool usage principles:
1) If a URL is present → use crawl4ai_scrape or crawl_urls_sequential
2) For system status → use health_check
3) For structured data conversion → use convert_to_json_format
4) For menu/DB queries → use menu_search
5) For HTML/Confluence content → use ari_extract_main_blocks and/or ari_markdown_to_json and/or convert_to_json_format

Response policy:
- After using tools, produce a concise, high-signal answer based on the results.
- IMPORTANT: Write all final answers to the user in Korean.
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            
            # First API call with tools
            try:
                logger.info(f"🚀 OpenAI API 호출 시작: model={settings.openai_model}, tools={len(formatted_tools)}개")
                logger.info(f"📝 메시지 내용: {messages[0]['content'][:200]}...")
                logger.info(f"🔧 도구 목록: {[t['function']['name'] for t in formatted_tools]}")
                
                response = await self._client.chat.completions.create(
                    model=settings.openai_model,
                    messages=messages,
                    tools=formatted_tools if formatted_tools else None,
                    tool_choice="auto" if formatted_tools else None,
                    timeout=60,  # 60초로 증가
                    max_tokens=1000  # 토큰 제한 추가
                )
                logger.info(f"✅ OpenAI API 응답 받음")
            except asyncio.TimeoutError:
                logger.error(f"⏰ OpenAI API 타임아웃 (60초)")
                raise LLMQueryError("OpenAI API 응답 시간 초과")
            except Exception as api_error:
                logger.error(f"❌ OpenAI API 호출 실패: {api_error}")
                raise
            
            # Check if tools were called
            message = response.choices[0].message
            
            if not message.tool_calls:
                return message.content or ""
            
            # Process tool calls
            messages.append(message.model_dump())  # Add assistant's message with tool calls
            
            # Execute each tool call
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                logger.info(f"Executing tool: {function_name} with args: {function_args}")
                
                try:
                    # Call your MCP service
                    tool_result = await mcp_service.call_tool(function_name, function_args)
                    
                    # Format the result
                    if hasattr(tool_result, 'structured_content'):
                        result_content = json.dumps(tool_result.structured_content, ensure_ascii=False)
                    elif hasattr(tool_result, 'data'):
                        result_content = json.dumps(tool_result.data, ensure_ascii=False)
                    elif isinstance(tool_result, dict):
                        result_content = json.dumps(tool_result, ensure_ascii=False)
                    else:
                        result_content = str(tool_result)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_content
                    })
                except Exception as e:
                    logger.error(f"Tool execution failed for {function_name}: {e}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps({"error": str(e)}, ensure_ascii=False)
                    })
            
            # Second API call for final response
            final_response = await self._client.chat.completions.create(
                model=settings.openai_model,
                messages=messages
            )
            
            return final_response.choices[0].message.content or ""
            
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            # 폴백: 도구 없이 간단한 응답 시도
            try:
                logger.info("🔄 도구 없이 폴백 응답 시도...")
                fallback_response = await self.generate_response(question)
                logger.info("✅ 폴백 응답 성공")
                return fallback_response
            except Exception as fallback_error:
                logger.error(f"❌ 폴백도 실패: {fallback_error}")
                raise LLMQueryError(f"LLM 쿼리 실행 중 오류: {str(e)}")
    
    async def query_tool_only(self, question: str, available_tools: List[Dict[str, Any]]) -> str:
        """모델이 도구만 호출하고 결과를 그대로 반환 (추가 해석 없음)"""
        try:
            formatted_tools = self._format_tools_for_openai(available_tools)
            
            logger.info(f"🔧 도구 호출 전용 모드: {len(formatted_tools)}개 도구")
            
            system_prompt = """You are a tool execution assistant. Your job is to:
1. Analyze the user's request
2. Call the appropriate tool with the provided content
3. Return ONLY the raw tool result without any additional interpretation

CRITICAL: Do not add explanations or interpretations. Just execute the tool and return the raw result."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            
            # OpenAI API 호출
            response = await self._client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                tools=formatted_tools,
                tool_choice="required"  # 도구 호출 강제
            )
            
            message = response.choices[0].message
            
            if not message.tool_calls:
                return json.dumps({"success": False, "error": "도구가 호출되지 않았습니다"}, ensure_ascii=False)
            
            # 첫 번째 도구 호출만 처리
            tool_call = message.tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            logger.info(f"🔧 도구 호출: {function_name}")
            
            # MCP 도구 실행
            tool_result = await mcp_service.call_tool(function_name, function_args)
            
            # 결과를 그대로 반환
            if hasattr(tool_result, 'structured_content') and tool_result.structured_content:
                result_data = tool_result.structured_content
            elif hasattr(tool_result, 'data') and tool_result.data:
                result_data = tool_result.data
            else:
                result_data = {"result": str(tool_result)}
            
            return json.dumps(result_data, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"도구 호출 전용 모드 실패: {e}")
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)
    
    async def query_with_raw_result(self, question: str, available_tools: List[Dict[str, Any]]) -> str:
        """모델이 도구를 호출하고 원본 결과를 그대로 반환 (추가 해석 없음)"""
        try:
            formatted_tools = self._format_tools_for_openai(available_tools)
            
            logger.info(f"🔧 모델 의도분류 + 원본 결과 반환 모드: {len(formatted_tools)}개 도구")
            
            system_prompt = """You are an AI assistant that analyzes user requests and calls appropriate tools.

CRITICAL INSTRUCTIONS:
1. Always analyze the user's request and call the most appropriate tool
2. After calling a tool, return ONLY the raw tool result without any additional interpretation or explanation
3. Do not add summaries, explanations, or your own analysis
4. Just execute the tool and return the raw JSON result

Available tools will help you process files, extract data, and perform various tasks."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            
            # OpenAI API 호출 (도구 호출 필수)
            response = await self._client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                tools=formatted_tools,
                tool_choice="auto",  # 자동으로 도구 선택
                timeout=60,
                max_tokens=1000
            )
            
            message = response.choices[0].message
            
            if not message.tool_calls:
                # 도구 호출이 없으면 일반 응답 반환
                return message.content or ""
            
            # 도구 호출 처리
            messages.append(message.model_dump())
            
            # 각 도구 호출 실행
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                logger.info(f"🔧 모델이 선택한 도구: {function_name}")
                
                try:
                    # MCP 도구 실행
                    tool_result = await mcp_service.call_tool(function_name, function_args)
                    
                    # 도구 결과를 원본 그대로 반환
                    if hasattr(tool_result, 'structured_content') and tool_result.structured_content:
                        return json.dumps(tool_result.structured_content, ensure_ascii=False, indent=2)
                    elif hasattr(tool_result, 'data') and tool_result.data:
                        return json.dumps(tool_result.data, ensure_ascii=False, indent=2)
                    else:
                        return str(tool_result)
                        
                except Exception as e:
                    logger.error(f"도구 실행 실패 {function_name}: {e}")
                    return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)
            
            # 여기까지 오면 도구 호출은 있었지만 결과 반환에 실패
            return json.dumps({"success": False, "error": "도구 호출 결과 처리 실패"}, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"모델 의도분류 + 원본 결과 반환 실패: {e}")
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)
    
    async def query_with_file_content(self, question: str, available_tools: List[Dict[str, Any]], file_content: str) -> str:
        """모델이 의도 분류 후 도구 호출, 실제 파일 내용은 도구 호출 시 전달"""
        try:
            formatted_tools = self._format_tools_for_openai(available_tools)
            
            logger.info(f"🔧 모델 의도분류 (파일 내용 별도 처리): {len(formatted_tools)}개 도구")
            
            system_prompt = """You are an AI assistant that analyzes user requests and calls appropriate tools.

CRITICAL INSTRUCTIONS:
1. Analyze the user's request and determine if a tool should be called
2. If an HTML file is uploaded, call the ari_parse_html tool
3. Do not worry about the file content - it will be provided automatically to the tool
4. Just decide which tool to call based on the user's intent

Your job is ONLY to decide which tool to call, not to process the content yourself."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            
            # OpenAI API 호출 (도구 호출 결정만)
            response = await self._client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                tools=formatted_tools,
                tool_choice="auto",
                timeout=60,
                max_tokens=500  # 의도 분류만 하므로 토큰 수 줄임
            )
            
            message = response.choices[0].message
            
            if not message.tool_calls:
                # 도구 호출이 없으면 일반 응답 반환
                return message.content or "도구 호출이 필요하지 않은 요청입니다."
            
            # 도구 호출 처리 (실제 파일 내용 전달)
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                logger.info(f"🔧 모델이 선택한 도구: {function_name}")
                
                # HTML 처리 도구인 경우 실제 파일 내용 전달
                if function_name == "ari_parse_html":
                    function_args["html_content"] = file_content
                    logger.info(f"📄 실제 파일 내용 전달: {len(file_content)}자")
                
                try:
                    # MCP 도구 실행
                    tool_result = await mcp_service.call_tool(function_name, function_args)
                    
                    # 도구 결과를 원본 그대로 반환
                    if hasattr(tool_result, 'structured_content') and tool_result.structured_content:
                        return json.dumps(tool_result.structured_content, ensure_ascii=False, indent=2)
                    elif hasattr(tool_result, 'data') and tool_result.data:
                        return json.dumps(tool_result.data, ensure_ascii=False, indent=2)
                    else:
                        return str(tool_result)
                        
                except Exception as e:
                    logger.error(f"도구 실행 실패 {function_name}: {e}")
                    return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)
            
            return json.dumps({"success": False, "error": "도구 호출 결과 처리 실패"}, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"파일 내용 별도 처리 모드 실패: {e}")
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)
    
    async def query_with_raw_result_and_html(self, question: str, available_tools: List[Dict[str, Any]], html_content: str) -> tuple[str, List[str]]:
        """
        새로운 플로우: 모델이 의도 분류 후 HTML → 마크다운 → JSON 구조화를 순차 수행
        - 모델이 사용자 질문을 분석하여 적절한 도구 워크플로우 결정
        - HTML 파일이 있으면 HTML → 마크다운 → JSON 순서로 처리
        - 모델 해석 없이 최종 JSON 결과만 반환
        """
        try:
            logger.info(f"🔧 새로운 플로우 - 모델 의도 분류 후 HTML 처리 시작")
            
            # 사용된 도구들 추적
            used_tools = []
            
            # 1단계: 모델이 의도 분류하여 워크플로우 결정
            formatted_tools = self._format_tools_for_openai(available_tools)
            
            system_prompt = """You are a tool execution assistant. When a user uploads an HTML file and asks to process it, you MUST call the ari_html_to_markdown tool.

MANDATORY WORKFLOW FOR HTML FILES:
1. ALWAYS call ari_html_to_markdown first (required for any HTML processing request)
2. You will then be asked to call ari_markdown_to_json in the next step

CRITICAL: You MUST call ari_html_to_markdown tool for ANY request involving HTML files. Do not provide text responses - only call the tool.

Available tools:
- ari_html_to_markdown: Convert HTML to markdown (REQUIRED for HTML files)
- ari_markdown_to_json: Convert markdown to JSON structure  
- ari_parse_html: Alternative HTML parser
- health_check: System health check

REMEMBER: For HTML processing requests, immediately call ari_html_to_markdown tool."""

            # 사용자 메시지를 HTML 처리 요청으로 명확화
            user_message = f"I have uploaded an HTML file. {question}. Please call ari_html_to_markdown tool to process it."
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # 모델이 의도 분류 및 첫 번째 도구 선택
            logger.info("🤖 모델이 의도 분류 및 도구 워크플로우 결정 중...")
            response = await self._client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                tools=formatted_tools,
                tool_choice="auto",
                timeout=60,
                max_tokens=500
            )
            
            message = response.choices[0].message
            
            if not message.tool_calls:
                logger.error("❌ 모델이 도구를 선택하지 않음")
                return json.dumps({"success": False, "error": "모델이 적절한 도구를 선택하지 못했습니다"}, ensure_ascii=False), []
            
            # 2단계: 모델이 선택한 도구들을 순차 실행
            logger.info(f"🤖 모델이 {len(message.tool_calls)}개 도구 호출 결정")
            
            # HTML → 마크다운 변환 (첫 번째 도구 호출)
            first_tool_call = message.tool_calls[0]
            function_name = first_tool_call.function.name
            function_args = json.loads(first_tool_call.function.arguments)
            
            logger.info(f"📄 1단계: 모델이 선택한 도구 - {function_name}")
            used_tools.append(function_name)
            
            # HTML 처리 도구에 실제 HTML 내용 전달
            if function_name in ["ari_html_to_markdown", "ari_parse_html"]:
                function_args["html_content"] = html_content
                logger.info(f"📄 HTML 내용 전달: {len(html_content)} bytes")
            
            # 첫 번째 도구 실행 (HTML → 마크다운)
            try:
                markdown_result = await mcp_service.call_tool(function_name, function_args)
                
                # fastmcp CallToolResult 구조 파싱
                markdown_content = ""
                first_success = False
                
                if hasattr(markdown_result, 'content') and markdown_result.content:
                    first_content = markdown_result.content[0]
                    if hasattr(first_content, 'text'):
                        json_data = json.loads(first_content.text)
                        logger.info(f"📄 1단계 JSON 파싱 성공: {json_data.get('success')}")
                        
                        if json_data.get('success') and 'result' in json_data:
                            result_data = json_data['result']
                            if 'markdown' in result_data:
                                first_success = True
                                markdown_content = result_data['markdown']
                                logger.info(f"✅ 1단계 완료 - 마크다운 변환: {len(markdown_content)}자")
                
                if not first_success:
                    logger.error("❌ 1단계 실패 - HTML → 마크다운 변환 실패")
                    return json.dumps({"success": False, "error": "HTML → 마크다운 변환 실패"}, ensure_ascii=False)
                    
            except Exception as e:
                logger.error(f"❌ 1단계 도구 실행 오류: {e}")
                return json.dumps({"success": False, "error": f"HTML → 마크다운 변환 오류: {str(e)}"}, ensure_ascii=False)
            
            # 3단계: 모델이 2번째 도구 선택 (1단계 결과 기반 컨텍스트 제공)
            logger.info("🤖 2단계: 모델이 1단계 결과를 바탕으로 다음 도구 선택")
            
            # 1단계 결과를 바탕으로 한 명확한 지침 제공
            second_system_prompt = """You are an AI assistant that continues a multi-step workflow based on previous results.

WORKFLOW CONTEXT:
- Step 1 was completed: HTML content was successfully converted to markdown
- Step 1 result: Markdown content is now available for further processing

CRITICAL INSTRUCTIONS for Step 2:
1. Since Step 1 (ari_html_to_markdown) was successful, you must now call ari_markdown_to_json
2. The ari_markdown_to_json tool converts markdown content into structured JSON format
3. This is the logical next step in the HTML → Markdown → JSON workflow
4. Do NOT call ari_html_to_markdown again - that step is already complete

Available tools:
- ari_markdown_to_json: Convert markdown to structured JSON (THIS IS WHAT YOU NEED)
- ari_html_to_markdown: Convert HTML to markdown (ALREADY COMPLETED)
- ari_parse_html: Simple HTML parsing (NOT NEEDED)
- health_check: Check system status (NOT NEEDED)

Your task: Call ari_markdown_to_json to process the markdown content from Step 1."""

            # 2번째 도구 호출을 위한 메시지 구성 (1단계 결과 포함)
            second_messages = [
                {"role": "system", "content": second_system_prompt},
                {"role": "user", "content": f"""Original request: {question}

Step 1 Status: ✅ COMPLETED
- Tool used: {function_name}
- Result: HTML successfully converted to markdown ({len(markdown_content)} characters)

Step 2 Task: Please use ari_markdown_to_json to convert the markdown content into structured JSON format.

This completes the workflow: HTML → Markdown → JSON"""}
            ]
            
            # 모델이 2번째 도구 선택
            logger.info("🤖 모델이 1단계 완료 컨텍스트를 바탕으로 2단계 도구 결정 중...")
            second_response = await self._client.chat.completions.create(
                model=settings.openai_model,
                messages=second_messages,
                tools=formatted_tools,
                tool_choice="auto",
                timeout=60,
                max_tokens=500
            )
            
            second_message = second_response.choices[0].message
            
            if not second_message.tool_calls:
                logger.error("❌ 모델이 2단계 도구를 선택하지 않음")
                return json.dumps({"success": False, "error": "모델이 2단계 도구를 선택하지 못했습니다"}, ensure_ascii=False)
            
            # 2번째 도구 실행
            second_tool_call = second_message.tool_calls[0]
            second_function_name = second_tool_call.function.name
            second_function_args = json.loads(second_tool_call.function.arguments)
            
            logger.info(f"🤖 2단계: 모델이 선택한 도구 - {second_function_name}")
            used_tools.append(second_function_name)
            
            # 올바른 도구 선택 검증
            if second_function_name != "ari_markdown_to_json":
                logger.warning(f"⚠️ 모델이 예상과 다른 도구 선택: {second_function_name} (예상: ari_markdown_to_json)")
            
            # 마크다운 내용 전달
            if second_function_name == "ari_markdown_to_json":
                second_function_args["markdown_content"] = markdown_content
                logger.info(f"📊 마크다운 내용 전달: {len(markdown_content)} chars")
            elif "markdown_content" in second_function_args:
                # 모델이 다른 도구를 선택했지만 markdown_content 파라미터가 있으면 전달
                second_function_args["markdown_content"] = markdown_content
                logger.info(f"📊 마크다운 내용 전달 (다른 도구): {len(markdown_content)} chars")
            
            try:
                json_result = await mcp_service.call_tool(second_function_name, second_function_args)
                
                # fastmcp CallToolResult 구조 파싱
                contents = []
                json_success = False
                
                if hasattr(json_result, 'content') and json_result.content:
                    first_content = json_result.content[0]
                    if hasattr(first_content, 'text'):
                        json_data = json.loads(first_content.text)
                        logger.info(f"📊 2단계 JSON 파싱 성공: {json_data.get('success')}")
                        
                        if json_data.get('success') and 'contents' in json_data:
                            contents = json_data['contents']
                            json_success = True
                            logger.info(f"✅ 2단계 완료 - JSON 구조화: {len(contents)}개 컨텐츠 블록")
                
                if json_success:
                    
                    # 최종 결과 구성
                    # 실제 HTML 콘텐츠 전체를 깔끔하게 정리해서 반환
                    logger.info("📝 HTML 콘텐츠 전체 정리 중...")
                    
                    # 모든 콘텐츠를 깔끔하게 정리
                    content_parts = []
                    current_section = ""
                    processed_count = 0
                    
                    logger.info(f"📄 총 {len(contents)}개 콘텐츠 블록 처리 시작")
                    
                    for i, item in enumerate(contents):
                        if item.get('type') == 'text' and item.get('data'):
                            title = item.get('title', '').strip()
                            data = item.get('data', '').strip()
                            
                            # 제목이 있고 이전 섹션과 다르면 새 섹션으로 처리
                            if title and title != current_section:
                                if title not in data:  # 제목이 데이터에 포함되지 않은 경우만
                                    content_parts.append(f"\n## {title}\n")
                                current_section = title
                            
                            # 실제 데이터 추가 (의미있는 내용만)
                            if data and len(data.strip()) > 2:
                                content_parts.append(data)
                                processed_count += 1
                        
                        # 진행 상황 로그 (매 50개마다)
                        if (i + 1) % 50 == 0:
                            logger.info(f"📄 진행 상황: {i + 1}/{len(contents)} 블록 처리됨")
                        
                        elif item.get('type') == 'table':
                            # 테이블 데이터 처리
                            headers = item.get('headers', [])
                            rows = item.get('rows', [])
                            
                            if headers and rows:
                                content_parts.append(f"\n### 표 데이터\n")
                                # 테이블 헤더
                                content_parts.append("| " + " | ".join(headers) + " |")
                                content_parts.append("| " + " | ".join(["---"] * len(headers)) + " |")
                                
                                # 테이블 행 (최대 10개만)
                                for row in rows[:10]:
                                    row_data = row.get('data', {})
                                    row_values = [str(row_data.get(header, '')) for header in headers]
                                    content_parts.append("| " + " | ".join(row_values) + " |")
                                
                                if len(rows) > 10:
                                    content_parts.append(f"\n... (총 {len(rows)}개 행 중 10개만 표시)")
                    
                    # 최종 콘텐츠 조합
                    if content_parts:
                        full_content = "\n".join(content_parts)
                        # 연속된 빈 줄 정리
                        import re
                        full_content = re.sub(r'\n\s*\n\s*\n', '\n\n', full_content)
                        full_content = full_content.strip()
                        
                        logger.info(f"✅ HTML 콘텐츠 전체 정리 완료")
                        logger.info(f"📊 처리 통계: {processed_count}개 텍스트 블록, 최종 길이: {len(full_content):,} characters")
                        logger.info(f"📄 전체 콘텐츠 미리보기 (처음 200자): {full_content[:200]}...")
                        logger.info(f"📄 전체 콘텐츠 미리보기 (마지막 200자): ...{full_content[-200:]}")
                        
                        # 응답 크기 체크 (1MB 제한)
                        max_response_size = 1024 * 1024  # 1MB
                        if len(full_content.encode('utf-8')) > max_response_size:
                            logger.warning(f"⚠️ 응답이 너무 큽니다 ({len(full_content.encode('utf-8')):,} bytes). 1MB로 제한합니다.")
                            # UTF-8 바이트 기준으로 자르기
                            content_bytes = full_content.encode('utf-8')
                            truncated_bytes = content_bytes[:max_response_size-100]  # 여유분 100바이트
                            try:
                                full_content = truncated_bytes.decode('utf-8')
                            except UnicodeDecodeError:
                                # 마지막 불완전한 UTF-8 문자 제거
                                for i in range(10):  # 최대 10바이트 뒤로
                                    try:
                                        full_content = truncated_bytes[:-i].decode('utf-8')
                                        break
                                    except UnicodeDecodeError:
                                        continue
                            full_content += "\n\n... (내용이 너무 길어 일부만 표시됩니다)"
                            logger.info(f"📏 응답 크기 조정됨: {len(full_content):,} characters")
                        
                        return full_content, used_tools
                    else:
                        logger.warning("⚠️ 추출된 콘텐츠가 없습니다")
                        return "HTML 파일에서 텍스트 콘텐츠를 찾을 수 없습니다.", used_tools
                    
                else:
                    logger.error(f"❌ 2단계 실패 - JSON 구조화 실패")
                    return json.dumps({"success": False, "error": "마크다운 → JSON 구조화 실패"}, ensure_ascii=False), used_tools
                    
            except Exception as e:
                logger.error(f"❌ 2단계 도구 실행 오류: {e}")
                return json.dumps({"success": False, "error": f"마크다운 → JSON 구조화 오류: {str(e)}"}, ensure_ascii=False), used_tools
            
        except Exception as e:
            logger.error(f"HTML 파일 처리 플로우 실패: {e}")
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False), used_tools if 'used_tools' in locals() else []
    
    def _get_tool_categories_description(self, tools: List[Dict[str, Any]]) -> str:
        """
        도구 카테고리별 설명 생성
        """
        categories = {
            'web': [],
            'system': [],
            'text': [],
            'html': [],
            'other': []
        }
        
        for tool in tools:
            tool_name = tool.get('function', {}).get('name', '')
            tool_desc = tool.get('function', {}).get('description', '')
            
            if 'crawl' in tool_name or 'scrape' in tool_name:
                categories['web'].append(f"- {tool_name}: {tool_desc}")
            elif 'health' in tool_name or 'status' in tool_name:
                categories['system'].append(f"- {tool_name}: {tool_desc}")
            elif 'summarize' in tool_name or 'extract' in tool_name:
                categories['text'].append(f"- {tool_name}: {tool_desc}")
            elif 'html' in tool_name or 'table' in tool_name or 'ari' in tool_name:
                categories['html'].append(f"- {tool_name}: {tool_desc}")
            else:
                categories['other'].append(f"- {tool_name}: {tool_desc}")
        
        description = ""
        if categories['web']:
            description += "**웹 크롤링 도구:**\n" + "\n".join(categories['web']) + "\n\n"
        if categories['system']:
            description += "**시스템 상태 도구:**\n" + "\n".join(categories['system']) + "\n\n"
        if categories['text']:
            description += "**텍스트 분석 도구:**\n" + "\n".join(categories['text']) + "\n\n"
        if categories['html']:
            description += "**HTML 처리 도구:**\n" + "\n".join(categories['html']) + "\n\n"
        if categories['other']:
            description += "**기타 도구:**\n" + "\n".join(categories['other']) + "\n\n"
        
        return description
    
    async def generate_response(self, prompt: str) -> str:
        """
        Generate a simple response using OpenAI Chat Completions API
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Generated response as string
        """
        try:
            # 토큰 수 계산 및 제한 확인
            tokens = self.tokenizer.encode(prompt)
            token_count = len(tokens)
            
            # RAG용으로는 gpt-4o-mini 사용 (더 저렴하고 토큰 제한이 큼)
            model = "gpt-4o-mini"  # RAG 응답 생성용
            
            # 토큰 제한 확인 (gpt-4o-mini는 128k context)
            max_tokens = 20000  # 더 안전한 제한
            if token_count > max_tokens:
                logger.warning(f"Prompt too long ({token_count} tokens), truncating to {max_tokens}")
                tokens = tokens[:max_tokens]
                prompt = self.tokenizer.decode(tokens)
                token_count = max_tokens
            
            logger.info(f"LLM request: {token_count} tokens, model: {model}")
            
            response = await self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다. 주어진 정보를 바탕으로 정확하고 유용한 답변을 제공합니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,  # 응답 토큰 수 증가
                temperature=0.7
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise LLMQueryError(f"Failed to generate response: {str(e)}")

# Global service instance
llm_service = LLMService()
