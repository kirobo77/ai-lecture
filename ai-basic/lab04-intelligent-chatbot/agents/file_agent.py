"""
Lab 4 - File Agent
File Manager API 전문 호출 및 파일 관리 에이전트
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import httpx
from datetime import datetime
from typing import Dict, List, Optional, Union
import json
import urllib.parse

class FileAgent:
    """File Manager API 전문 호출 에이전트"""
    
    def __init__(self, api_base_url: str = "http://localhost:8003"):
        """File Agent 초기화"""
        self.name = "File Agent"
        self.version = "1.0.0"
        self.api_base_url = api_base_url.rstrip('/')
        
        # HTTP 클라이언트 설정
        self.client = httpx.Client(timeout=30.0)
        
        # 파일 타입 매핑
        self.file_type_mapping = {
            '문서': 'document',
            '이미지': 'image',
            '비디오': 'video',
            '코드': 'code',
            '기타': 'other',
            'document': 'document',
            'image': 'image',
            'video': 'video',
            'code': 'code',
            'other': 'other'
        }
        
        # 확장자별 타입 매핑
        self.extension_mapping = {
            # 문서
            'pdf': 'document', 'doc': 'document', 'docx': 'document',
            'txt': 'document', 'md': 'document', 'rtf': 'document',
            'ppt': 'document', 'pptx': 'document', 'xls': 'document', 'xlsx': 'document',
            # 이미지
            'jpg': 'image', 'jpeg': 'image', 'png': 'image', 'gif': 'image',
            'bmp': 'image', 'svg': 'image', 'tiff': 'image',
            # 비디오
            'mp4': 'video', 'avi': 'video', 'mov': 'video', 'mkv': 'video',
            'wmv': 'video', 'flv': 'video', 'webm': 'video',
            # 코드
            'py': 'code', 'js': 'code', 'html': 'code', 'css': 'code',
            'java': 'code', 'cpp': 'code', 'c': 'code', 'php': 'code',
            'go': 'code', 'rs': 'code', 'swift': 'code'
        }
        
        print(f"{self.name} 초기화 완료 (API: {self.api_base_url})")
    
    def normalize_file_type(self, file_type: str) -> str:
        """파일 타입 정규화"""
        if not file_type:
            return 'other'
        
        file_type_lower = file_type.lower().strip()
        
        # 직접 매핑
        if file_type_lower in self.file_type_mapping:
            return self.file_type_mapping[file_type_lower]
        
        # 확장자로 추론
        if file_type_lower.startswith('.'):
            extension = file_type_lower[1:]
        else:
            extension = file_type_lower
        
        return self.extension_mapping.get(extension, 'other')
    
    def search_files(self, query: str, file_type: Optional[str] = None, tags: Optional[str] = None) -> Dict:
        """파일 검색"""
        try:
            # 파라미터 구성
            params = {"q": query}
            
            if file_type:
                params["file_type"] = self.normalize_file_type(file_type)
            
            if tags:
                params["tags"] = tags
            
            response = self.client.get(
                f"{self.api_base_url}/files/search",
                params=params
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json(),
                    "action": "search",
                    "query": query
                }
            else:
                return {
                    "success": False,
                    "error": f"파일 검색 실패: {response.status_code}",
                    "query": query
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"파일 검색 실패: {str(e)}",
                "query": query
            }
    
    def get_file_content(self, file_id: str) -> Dict:
        """파일 내용 조회"""
        try:
            response = self.client.get(f"{self.api_base_url}/files/{file_id}/content")
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json(),
                    "action": "get_content",
                    "file_id": file_id
                }
            else:
                return {
                    "success": False,
                    "error": f"파일 내용 조회 실패: {response.status_code}",
                    "file_id": file_id
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"파일 내용 조회 실패: {str(e)}",
                "file_id": file_id
            }
    
    def get_file_list(self, directory: Optional[str] = None, file_type: Optional[str] = None) -> Dict:
        """파일 목록 조회"""
        try:
            # 파라미터 구성
            params = {}
            
            if directory:
                params["directory"] = directory
            
            if file_type:
                params["file_type"] = self.normalize_file_type(file_type)
            
            response = self.client.get(
                f"{self.api_base_url}/files",
                params=params
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json(),
                    "action": "list",
                    "directory": directory,
                    "file_type": file_type
                }
            else:
                return {
                    "success": False,
                    "error": f"파일 목록 조회 실패: {response.status_code}",
                    "directory": directory
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"파일 목록 조회 실패: {str(e)}",
                "directory": directory
            }
    
    def get_directory_structure(self, path: Optional[str] = None) -> Dict:
        """디렉토리 구조 조회"""
        try:
            params = {}
            if path:
                params["path"] = path
            
            response = self.client.get(
                f"{self.api_base_url}/files/directory",
                params=params
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json(),
                    "action": "directory",
                    "path": path
                }
            else:
                return {
                    "success": False,
                    "error": f"디렉토리 구조 조회 실패: {response.status_code}",
                    "path": path
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"디렉토리 구조 조회 실패: {str(e)}",
                "path": path
            }
    
    def get_file_stats(self) -> Dict:
        """파일 통계 정보 조회"""
        try:
            response = self.client.get(f"{self.api_base_url}/files/stats")
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json(),
                    "action": "stats"
                }
            else:
                return {
                    "success": False,
                    "error": f"파일 통계 조회 실패: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"파일 통계 조회 실패: {str(e)}"
            }
    
    def get_popular_tags(self) -> Dict:
        """인기 태그 조회"""
        try:
            response = self.client.get(f"{self.api_base_url}/files/tags/popular")
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json(),
                    "action": "popular_tags"
                }
            else:
                return {
                    "success": False,
                    "error": f"인기 태그 조회 실패: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"인기 태그 조회 실패: {str(e)}"
            }
    
    def format_file_response(self, file_result: Dict, request_type: str = "search") -> str:
        """파일 정보를 사용자 친화적 메시지로 변환"""
        if not file_result.get("success"):
            error_msg = file_result.get("error", "알 수 없는 오류")
            return f"죄송합니다. 파일 정보를 가져올 수 없습니다. ({error_msg})"
        
        data = file_result["data"]
        
        if request_type == "search":
            # 파일 검색 결과 포맷
            files = data.get("files", [])
            total_matches = data.get("total_matches", 0)
            query = file_result.get("query", "검색어")
            search_time = data.get("search_time_ms", 0)
            
            if total_matches == 0:
                return f"🔍 '{query}' 검색 결과\n\n검색된 파일이 없습니다."
            
            response = f"🔍 '{query}' 검색 결과 ({total_matches}개, {search_time}ms)\n\n"
            
            for i, file in enumerate(files[:10], 1):  # 최대 10개까지만 표시
                name = file.get("name", "이름없음")
                file_type = file.get("type", "unknown")
                size = file.get("size", 0)
                modified_at = file.get("modified_at", "날짜불명")
                tags = file.get("tags", [])
                preview = file.get("content_preview", "")
                
                # 파일 크기 포맷팅
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size/1024:.1f}KB"
                else:
                    size_str = f"{size/(1024*1024):.1f}MB"
                
                # 파일 타입별 아이콘
                type_icons = {
                    "document": "📄",
                    "image": "🖼️",
                    "video": "🎥",
                    "code": "💻",
                    "other": "📁"
                }
                icon = type_icons.get(file_type, "📁")
                
                response += f"{icon} **{name}**\n"
                response += f"   📊 {size_str} | {file_type} | {modified_at[:10]}\n"
                
                if tags:
                    response += f"   🏷️ {', '.join(tags[:3])}"
                    if len(tags) > 3:
                        response += f" +{len(tags)-3}"
                    response += "\n"
                
                if preview and len(preview) > 10:
                    preview_text = preview[:100] + "..." if len(preview) > 100 else preview
                    response += f"   💭 {preview_text}\n"
                
                response += "\n"
            
            if total_matches > 10:
                response += f"... 및 {total_matches-10}개 파일 더"
            
            return response
            
        elif request_type == "get_content":
            # 파일 내용 조회 포맷
            name = data.get("name", "파일")
            content = data.get("content", "")
            size = data.get("size", 0)
            file_type = data.get("type", "unknown")
            
            response = f"📄 {name} 내용\n\n"
            
            if len(content) > 500:
                response += f"```\n{content[:500]}...\n```\n\n"
                response += f"💡 전체 {size}글자 중 500글자만 표시"
            else:
                response += f"```\n{content}\n```"
            
            return response
            
        elif request_type == "list":
            # 파일 목록 포맷
            files = data.get("files", [])
            total_files = data.get("total_files", 0)
            directory = file_result.get("directory", "루트")
            
            if total_files == 0:
                return f"📁 {directory}\n\n파일이 없습니다."
            
            response = f"📁 {directory} ({total_files}개 파일)\n\n"
            
            # 타입별 그룹화
            type_groups = {}
            for file in files:
                file_type = file.get("type", "other")
                if file_type not in type_groups:
                    type_groups[file_type] = []
                type_groups[file_type].append(file)
            
            type_icons = {
                "document": "📄",
                "image": "🖼️", 
                "video": "🎥",
                "code": "💻",
                "other": "📁"
            }
            
            for file_type, files_in_type in type_groups.items():
                icon = type_icons.get(file_type, "📁")
                response += f"{icon} **{file_type.title()}** ({len(files_in_type)}개)\n"
                
                for file in files_in_type[:5]:  # 각 타입당 최대 5개
                    name = file.get("name", "이름없음")
                    size = file.get("size", 0)
                    modified_at = file.get("modified_at", "")
                    
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024 * 1024:
                        size_str = f"{size/1024:.1f}KB"
                    else:
                        size_str = f"{size/(1024*1024):.1f}MB"
                    
                    response += f"   • {name} ({size_str})\n"
                
                if len(files_in_type) > 5:
                    response += f"   ... 및 {len(files_in_type)-5}개 더\n"
                
                response += "\n"
            
            return response
            
        elif request_type == "directory":
            # 디렉토리 구조 포맷
            structure = data.get("structure", {})
            total_items = data.get("total_items", 0)
            
            response = f"📂 디렉토리 구조 ({total_items}개 항목)\n\n"
            
            def format_directory_tree(items, indent=0):
                tree_str = ""
                for item in items:
                    name = item.get("name", "이름없음")
                    is_directory = item.get("is_directory", False)
                    children = item.get("children", [])
                    
                    prefix = "  " * indent
                    icon = "📁" if is_directory else "📄"
                    tree_str += f"{prefix}{icon} {name}\n"
                    
                    if children and indent < 3:  # 최대 3단계까지만
                        tree_str += format_directory_tree(children, indent + 1)
                
                return tree_str
            
            if "items" in structure:
                response += format_directory_tree(structure["items"])
            
            return response
            
        elif request_type == "stats":
            # 파일 통계 포맷
            total_files = data.get("total_files", 0)
            total_size = data.get("total_size_mb", 0)
            by_type = data.get("by_type", {})
            popular_tags = data.get("popular_tags", [])
            
            response = f"📊 파일 통계\n\n"
            response += f"📁 전체 파일: {total_files:,}개\n"
            response += f"💾 총 용량: {total_size:.1f}MB\n\n"
            
            if by_type:
                response += "📈 타입별 분포:\n"
                for file_type, count in by_type.items():
                    percentage = (count / total_files * 100) if total_files > 0 else 0
                    response += f"   • {file_type}: {count}개 ({percentage:.1f}%)\n"
                response += "\n"
            
            if popular_tags:
                response += "🏷️ 인기 태그:\n"
                for tag in popular_tags[:5]:
                    response += f"   • {tag}\n"
            
            return response
        
        return "파일 정보를 처리할 수 없습니다."
    
    def process_file_request(self, parameters: Dict) -> Dict:
        """파일 요청 처리 (Intent Classifier에서 호출)"""
        try:
            # 매개변수 추출
            action = parameters.get("action", "search")  # search, get_content, list, directory, stats
            query = parameters.get("query", parameters.get("search_term", ""))
            file_type = parameters.get("file_type", parameters.get("type", ""))
            tags = parameters.get("tags", "")
            directory = parameters.get("directory", parameters.get("path", ""))
            file_id = parameters.get("file_id", "")
            
            print(f"파일 요청 처리: {action} - {query or directory or file_id}")
            
            # 액션에 따른 처리
            if action == "search":
                # 파일 검색
                if not query:
                    # 사용자 입력에서 키워드 자동 추출 시도
                    user_input = parameters.get("user_input", "")
                    auto_query = self.extract_search_keywords(user_input)
                    
                    if auto_query:
                        query = auto_query
                        print(f"🔄 [자동 검색어 추출] '{query}'")
                    else:
                        return {
                            "success": False,
                            "agent": self.name,
                            "response": "검색어를 알려주세요.",
                            "error": "검색어 누락"
                        }
                
                result = self.search_files(query, file_type, tags)
                formatted_response = self.format_file_response(result, "search")
                
            elif action == "get_content":
                # 파일 내용 조회
                if not file_id:
                    return {
                        "success": False,
                        "agent": self.name,
                        "response": "파일 ID를 알려주세요.",
                        "error": "파일 ID 누락"
                    }
                
                result = self.get_file_content(file_id)
                formatted_response = self.format_file_response(result, "get_content")
                
            elif action == "list":
                # 파일 목록 조회
                result = self.get_file_list(directory, file_type)
                formatted_response = self.format_file_response(result, "list")
                
            elif action == "directory":
                # 디렉토리 구조 조회
                result = self.get_directory_structure(directory)
                formatted_response = self.format_file_response(result, "directory")
                
            elif action == "stats":
                # 파일 통계 조회
                result = self.get_file_stats()
                formatted_response = self.format_file_response(result, "stats")
                
            else:
                # 기본값: 검색
                if query:
                    result = self.search_files(query, file_type, tags)
                    formatted_response = self.format_file_response(result, "search")
                else:
                    result = self.get_file_list(directory, file_type)
                    formatted_response = self.format_file_response(result, "list")
            
            return {
                "success": result.get("success", False),
                "agent": self.name,
                "response": formatted_response,
                "raw_data": result,
                "processed_at": datetime.now().isoformat(),
                "action": action
            }
            
        except Exception as e:
            return {
                "success": False,
                "agent": self.name,
                "response": f"파일 처리 중 오류가 발생했습니다: {str(e)}",
                "error": str(e),
                "processed_at": datetime.now().isoformat()
            }
    
    def get_file_summary_for_notification(self, query: str) -> str:
        """알림용 간단한 파일 요약"""
        try:
            result = self.search_files(query)
            
            if result.get("success"):
                data = result["data"]
                total_matches = data.get("total_matches", 0)
                
                if total_matches == 0:
                    return f"'{query}' 관련 파일을 찾을 수 없습니다."
                else:
                    files = data.get("files", [])
                    file_names = [f.get("name", "파일") for f in files[:3]]
                    summary = f"'{query}' 검색 결과 {total_matches}개: "
                    summary += ", ".join(file_names)
                    if total_matches > 3:
                        summary += f" 외 {total_matches-3}개"
                    return summary
            else:
                return f"'{query}' 파일 검색 중 오류가 발생했습니다."
                
        except Exception as e:
            return f"파일 요약 중 오류: {str(e)}"
    
    def extract_search_keywords(self, user_input: str) -> str:
        """사용자 입력에서 검색 키워드 자동 추출"""
        try:
            # 일반적인 파일 관련 키워드들
            file_keywords = [
                "프로젝트", "문서", "API", "명세서", "회의록", "보고서", "계획서",
                "가이드", "매뉴얼", "readme", "코드", "파일", "데이터", "스크립트",
                "이미지", "사진", "영상", "비디오", "음악", "소스", "설정", "config"
            ]
            
            # 키워드 추출 패턴들
            patterns = [
                r'(.+?)\s*(?:문서|파일)',  # "프로젝트 문서", "API 파일"
                r'(.+?)\s*(?:찾아|검색)',  # "프로젝트 찾아", "API 검색"
                r'(.+?)\s*(?:보여|알려)',  # "문서 보여", "파일 알려"
            ]
            
            user_input_lower = user_input.lower()
            
            # 패턴 매칭으로 키워드 추출
            for pattern in patterns:
                import re
                matches = re.findall(pattern, user_input, re.IGNORECASE)
                for match in matches:
                    match = match.strip()
                    if len(match) > 1 and match not in ["이", "그", "저", "것", "거"]:
                        print(f"🔍 패턴 매칭으로 키워드 추출: '{match}'")
                        return match
            
            # 알려진 키워드 직접 검색
            for keyword in file_keywords:
                if keyword in user_input_lower:
                    print(f"🔍 키워드 사전에서 추출: '{keyword}'")
                    return keyword
            
            # 마지막 수단: 명사 추출 (간단한 휴리스틱)
            words = user_input.split()
            for word in words:
                word = word.strip('.,!?')
                if len(word) > 2 and word not in ["찾아서", "보내줘", "알려줘", "확인해줘"]:
                    if any(char in word for char in "프로젝트문서API명세서보고서"):
                        print(f"🔍 휴리스틱으로 키워드 추출: '{word}'")
                        return word
            
            return ""
            
        except Exception as e:
            print(f"키워드 추출 실패: {e}")
            return ""
    
    def get_capabilities(self) -> Dict:
        """에이전트 능력 정보"""
        return {
            "name": self.name,
            "version": self.version,
            "description": "File Manager API 전문 호출 및 파일 관리",
            "supported_operations": [
                "파일 검색 (키워드, 태그, 타입별)",
                "파일 내용 조회",
                "파일 목록 조회",
                "디렉토리 구조 탐색",
                "파일 통계 정보",
                "자동 검색어 추출"
            ],
            "supported_file_types": list(self.file_type_mapping.values()),
            "supported_extensions": list(self.extension_mapping.keys()),
            "api_endpoint": self.api_base_url
        }
    
    def __del__(self):
        """소멸자 - HTTP 클라이언트 정리"""
        try:
            self.client.close()
        except:
            pass

def test_file_agent():
    """File Agent 테스트"""
    print("=" * 60)
    print("File Agent 테스트")
    print("=" * 60)
    
    # File Agent 초기화
    agent = FileAgent()
    
    # 테스트 케이스들
    test_cases = [
        {"action": "search", "query": "프로젝트"},
        {"action": "search", "query": "API", "file_type": "document"},
        {"action": "list", "directory": "documents"},
        {"action": "directory"},
        {"action": "stats"},
        {"action": "get_content", "file_id": "doc_001"}  # 테스트용 ID
    ]
    
    print(f"\n{len(test_cases)}개 테스트 케이스로 File Agent 테스트:")
    print("-" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}] 테스트: {test_case}")
        
        # File Agent 호출
        result = agent.process_file_request(test_case)
        
        print(f"성공 여부: {result['success']}")
        print(f"응답:\n{result['response'][:300]}...")  # 처음 300자만
        
        if not result['success']:
            print(f"오류: {result.get('error', '알 수 없는 오류')}")
    
    # 알림용 요약 테스트
    print(f"\n" + "=" * 60)
    print("알림용 파일 요약 테스트:")
    summary = agent.get_file_summary_for_notification("API")
    print(f"  API 파일 요약: {summary}")
    
    # 에이전트 능력 정보
    print(f"\n📋 File Agent 정보:")
    capabilities = agent.get_capabilities()
    print(f"  이름: {capabilities['name']}")
    print(f"  설명: {capabilities['description']}")
    print(f"  지원 기능: {len(capabilities['supported_operations'])}개")
    print(f"  지원 파일 타입: {capabilities['supported_file_types']}")
    print(f"  API 엔드포인트: {capabilities['api_endpoint']}")
    
    print(f"\n" + "=" * 60)
    print("File Agent 테스트 완료!")

if __name__ == "__main__":
    test_file_agent() 