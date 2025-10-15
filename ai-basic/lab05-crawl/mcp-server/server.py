from fastmcp import FastMCP
import asyncio
import logging
from typing import Dict, Any
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 불필요한 디버그 로그 숨기기
logging.getLogger("mcp.server").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("sse_starlette").setLevel(logging.WARNING)
logging.getLogger("mcp.server.lowlevel").setLevel(logging.WARNING)

mcp = FastMCP(name="AriProcessingServer")

# Health check endpoint (MCP tool)
@mcp.tool
def health_check() -> Dict[str, Any]:
    """
    ARI Processing Server 헬스체크
    - BeautifulSoup 임포트 가능 여부 확인
    """
    logger.info("[MCP] health_check called")
    soup_ok = False

    # BeautifulSoup import 확인
    try:
        import importlib
        importlib.import_module("bs4")
        soup_ok = True
    except Exception as e:
        logger.warning(f"BeautifulSoup import failed: {e}")

    status = "healthy" if soup_ok else "unhealthy"
    return {
        "success": soup_ok,
        "status": status,
        "service": "ari-processing-server",
        "dependencies": {
            "beautifulsoup": soup_ok,
        }
    }


# ============================================================================
# ARI CONTENT PROCESSING TOOLS (HTML 구조화 및 전용 파싱)
# ============================================================================

@mcp.tool  
def ari_parse_html(html_content: str) -> Dict[str, Any]:
    """
    ARI 전용 HTML 파싱: 순수 HTML 파싱 및 구조화된 JSON 반환
    - 필터링 없이 모든 텍스트 및 이미지 추출
    - ARI 모델 스키마에 맞춘 구조화된 결과 반환
    - RAG 크롤링과는 다른 목적의 전용 파서
    """
    try:
        from bs4 import BeautifulSoup
        from datetime import datetime
        
        soup = BeautifulSoup(html_content or "", 'html.parser')
        title_el = soup.find('title')
        title_text = title_el.get_text(strip=True) if title_el else ""
        text = soup.get_text(separator=' ', strip=True)

        images = []
        for img in soup.find_all('img', src=True):
            images.append({'alt': (img.get('alt') or '').strip(), 'src': img['src']})

        return {
            'success': True,
            'result': {
                'content': {'text': text},
                'metadata': {
                    'title': title_text,
                    'extracted_at': datetime.now().isoformat(),
                    'content_length': len(text),
                    'images': images
                }
            }
        }
    except Exception as e:
        logger.error(f"ARI 파싱 실패: {e}")
        return {'success': False, 'error': str(e)}


@mcp.tool
def ari_html_to_markdown(html_content: str) -> Dict[str, Any]:
    """
    HTML을 마크다운으로 변환하는 도구
    - Confluence HTML을 정제하여 마크다운으로 변환
    - 테이블, 헤더, 링크 등을 마크다운 형식으로 보존
    - ARI 전용 HTML 전처리 로직 포함
    """
    logger.info(f"[MCP] ari_html_to_markdown 호출됨 - HTML 크기: {len(html_content)} chars")
    try:
        from bs4 import BeautifulSoup
        from datetime import datetime
        import re
        import tempfile
        import os
        
        # markdownify 라이브러리 시도
        try:
            from markdownify import markdownify as md_convert
        except ImportError:
            md_convert = None
            
        # pymupdf4llm 라이브러리 시도
        try:
            import pymupdf4llm
        except ImportError:
            pymupdf4llm = None
        
        def extract_clean_html(html_content: str) -> str:
            """exclude 요소 제거 후 페이지 제목, 브레드크럼, main-content 원문 HTML을 그대로 반환"""
            soup = BeautifulSoup(html_content, 'html.parser')
            elements_to_remove = [
                'header', 'footer', 'nav', 'aside', 'sidebar',
                '.header', '.footer', '.nav', '.aside', '.sidebar',
                '.navigation', '.menu',
                'div.aui-page-header-actions', 'div.page-actions', 'div.aui-toolbar2', 
                'div.comment-container', 'div.like-button-container', 'div.page-labels', 
                'div.comment-actions', 'span.st-table-filter', 'svg',
                'div.confluence-information-macro', 'div.aui-message', 'div.page-metadata-modification-info',
                '.aui-page-header-actions', '.like-button-container', '.page-labels',
                # Confluence UI 메뉴 요소들 추가 제거
                'aui-item-link', 'aui-dropdown2-trigger', 'aui-dropdown2-section',
                '.table-filter-inserter', '.pivot-table-inserter', '.table-chart-inserter',
                '[data-macro="table-filter"]', '[data-macro="pivot-table"]', '[data-macro="table-chart"]',
                '.aui-icon', '.aui-iconfont-configure', '.aui-icon-wait',
                'div#page-metadata-banner', 'ul.banner',
            ]
            for selector in elements_to_remove:
                for el in soup.select(selector):
                    el.decompose()
            
            # 특정 UI 텍스트가 포함된 요소들 제거
            ui_texts_to_remove = [
                "Filter table data", "Create a pivot table", 
                "Create a chart from data series", "Configure buttons visibility",
                "table-filter", "pivot-table", "table-chart"
            ]
            
            for text in ui_texts_to_remove:
                # 해당 텍스트를 포함한 모든 요소 찾아서 제거
                for el in soup.find_all(text=lambda t: t and text in str(t)):
                    parent = el.parent
                    if parent:
                        parent.decompose()
                
                # 속성값으로도 찾아서 제거
                for el in soup.find_all(attrs={"data-macro": text}):
                    el.decompose()
                    
                for el in soup.find_all(class_=lambda x: x and text in str(x)):
                    el.decompose()

            # 페이지 제목과 브레드크럼을 포함한 컨테이너 생성
            result_parts = []
            
            # 1. 페이지 제목 추가
            title_element = soup.find('h1', {'id': 'title-text'})
            if title_element:
                result_parts.append(f"<h1>{title_element.decode_contents()}</h1>")
            
            # 2. 브레드크럼 추가
            breadcrumb_element = soup.find('ol', {'id': 'breadcrumbs'})
            if breadcrumb_element:
                result_parts.append(f"<nav aria-label='이동 경로'>{breadcrumb_element.decode_contents()}</nav>")
            else:
                # 대안: breadcrumbs 클래스가 있는 요소 찾기
                breadcrumb_alt = soup.find('div', class_='breadcrumbs')
                if breadcrumb_alt:
                    result_parts.append(f"<nav aria-label='이동 경로'>{breadcrumb_alt.decode_contents()}</nav>")
            
            # 3. 메인 콘텐츠 추가
            main_content = soup.find('div', {'id': 'main-content'})
            if not main_content:
                main_content = soup.find('div', {'class': 'wiki-content'})
            if not main_content:
                main_content = soup.find('main') or soup.find('body') or soup

            if main_content:
                try:
                    main_html = main_content.decode_contents() if hasattr(main_content, 'decode_contents') else str(main_content)
                    result_parts.append(main_html)
                except Exception:
                    result_parts.append(str(main_content))

            return '\n'.join(result_parts)
        
        # HTML 정제
        cleaned_html = extract_clean_html(html_content)
        
        # 마크다운 변환 시도
        markdown_content = ""
        
        # 1. pymupdf4llm 시도
        if pymupdf4llm is not None and cleaned_html.strip():
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as temp_html:
                    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
</head>
<body>
    {cleaned_html}
</body>
</html>"""
                    temp_html.write(full_html)
                    temp_html_path = temp_html.name
                
                markdown_result = pymupdf4llm.to_markdown(temp_html_path)
                
                try:
                    os.unlink(temp_html_path)
                except:
                    pass
                
                if markdown_result and markdown_result.strip():
                    markdown_content = markdown_result
                    
            except Exception as e:
                logger.warning(f"pymupdf4llm conversion failed: {e}")
                try:
                    if 'temp_html_path' in locals():
                        os.unlink(temp_html_path)
                except:
                    pass
        
        # 2. markdownify 폴백
        if not markdown_content and md_convert is not None:
            try:
                markdown_content = md_convert(
                    cleaned_html,
                    heading_style="ATX",
                    strip=['style', 'script']
                )
            except Exception as e:
                logger.warning(f"markdownify failed: {e}")
        
        # 3. 최종 폴백 - BeautifulSoup으로 텍스트 추출
        if not markdown_content:
            try:
                soup_remaining = BeautifulSoup(cleaned_html, 'html.parser')
                markdown_content = soup_remaining.get_text('\n', strip=True)
            except Exception:
                markdown_content = cleaned_html
        
        result = {
            'success': True,
            'result': {
                'markdown': markdown_content,
                'original_length': len(html_content),
                'markdown_length': len(markdown_content),
                'converted_at': datetime.now().isoformat()
            }
        }
        logger.info(f"[MCP] ari_html_to_markdown 완료 - 마크다운 크기: {len(markdown_content)} chars")
        return result
        
    except Exception as e:
        logger.error(f"HTML to Markdown 변환 실패: {e}")
        return {'success': False, 'error': str(e)}


@mcp.tool
def ari_markdown_to_json(markdown_content: str) -> Dict[str, Any]:
    """
    마크다운을 구조화된 JSON 형태로 변환
    - 헤더(#..)는 이후 text 블록의 title로 사용
    - 마크다운 테이블(|...| + | --- |)을 table 항목으로 파싱
    - 그 외 연속 텍스트를 하나의 text 항목으로 누적하여 추가
    """
    logger.info(f"[MCP] ari_markdown_to_json 호출됨 - 마크다운 크기: {len(markdown_content)} chars")
    try:
        import re
        from typing import List, Dict, Any
        
        if markdown_content is None:
            return {"success": False, "error": "마크다운이 비어있습니다"}

        lines = markdown_content.splitlines()
        contents: List[Dict[str, Any]] = []
        buffer: List[str] = []  # 텍스트 누적 버퍼
        current_title: str = ""
        idx = 0
        i = 0

        def flush_text_buffer():
            nonlocal idx, buffer
            if buffer and any(s.strip() for s in buffer):
                text = "\n".join([s.rstrip() for s in buffer]).strip()
                if text:
                    idx += 1
                    contents.append({
                        "id": idx,
                        "type": "text",
                        "title": current_title,
                        "data": text
                    })
            buffer = []

        while i < len(lines):
            line = lines[i]

            # 제목 라인
            if re.match(r"^\s*#{1,6}\s+", line):
                # 기존 텍스트 버퍼를 먼저 비움
                flush_text_buffer()
                # 현재 제목 갱신
                current_title = re.sub(r"^\s*#{1,6}\s+", "", line).strip()
                i += 1
                continue

            # 테이블 감지: 현재 줄이 헤더 라인, 다음 줄이 구분선
            if '|' in line:
                # 테이블 헤더 후보와 구분선 검사
                header_candidate = line.strip()
                if i + 1 < len(lines):
                    separator = lines[i + 1].strip()
                    if re.match(r"^\|\s*:?\-+\s*(\|\s*:?\-+\s*)+\|$", separator):
                        # 텍스트 버퍼를 먼저 비움
                        flush_text_buffer()

                        # 헤더 파싱
                        raw_headers = [h.strip() for h in header_candidate.strip('|').split('|')]
                        headers = [h for h in raw_headers if h != ""]

                        # 데이터 행 수집
                        i += 2  # 헤더와 구분선 건너뜀
                        rows = []
                        row_id = 0
                        while i < len(lines) and '|' in lines[i] and not re.match(r"^\s*#", lines[i]):
                            row_line = lines[i].strip()
                            if not row_line:
                                break
                            raw_cells = [c.strip() for c in row_line.strip('|').split('|')]
                            # 셀 수가 헤더보다 적으면 보정
                            while len(raw_cells) < len(headers):
                                raw_cells.append("")
                            data = {headers[j]: raw_cells[j] if j < len(raw_cells) else "" for j in range(len(headers))}
                            row_id += 1
                            rows.append({"row_id": row_id, "data": data})
                            i += 1

                        idx += 1
                        contents.append({
                            "id": idx,
                            "type": "table",
                            "headers": headers,
                            "rows": rows
                        })
                        continue

            # 빈 줄이면 텍스트 버퍼를 플러시
            if not line.strip():
                flush_text_buffer()
                i += 1
                continue

            # 그 외는 텍스트 버퍼에 누적
            buffer.append(line)
            i += 1

        # 루프 종료 후 남은 텍스트 반영
        flush_text_buffer()

        result = {"success": True, "contents": contents}
        logger.info(f"[MCP] ari_markdown_to_json 완료 - {len(contents)}개 컨텐츠 블록 생성")
        return result
        
    except Exception as e:
        logger.error(f"마크다운 to JSON 변환 실패: {e}")
        return {"success": False, "error": str(e)}

async def main():
    # Start ARI Processing MCP server
    logger.info("🚀 ARI Processing MCP Server 시작 중...")
    logger.info("📍 서버 주소: http://0.0.0.0:4200/my-custom-path")
    logger.info("🔧 사용 가능한 도구: health_check, ari_parse_html, ari_html_to_markdown, ari_markdown_to_json")
    
    await mcp.run_async(
        transport="http",
        host="0.0.0.0",
        port=4200,
        path="/my-custom-path",
        log_level="info",
    )

if __name__ == "__main__":
    asyncio.run(main())