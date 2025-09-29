"""
Lab 4 - RAG 지식베이스 관리자
의도 분석, 대화 컨텍스트, 도메인 지식을 위한 벡터 저장소 관리
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import chromadb
from chromadb.utils import embedding_functions
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from shared.config import OPENAI_API_KEY, CHAT_MODEL
from shared.utils import EmbeddingUtils, ChromaUtils

class KnowledgeBase:
    """RAG 지식베이스 관리자"""
    
    def __init__(self, persist_directory: str = "./data/rag_db"):
        """지식베이스 초기화"""
        self.persist_directory = persist_directory
        
        # ChromaDB 클라이언트 설정
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # OpenAI 임베딩 함수 설정 (SSL 검증 비활성화)
        self.openai_ef = ChromaUtils.create_openai_embedding_function()
        
        # 컬렉션들
        self.collections = {}
        
        self.initialize_collections()
        print(f"RAG 지식베이스 초기화 완료: {persist_directory}")
    
    def initialize_collections(self):
        """필요한 컬렉션들 초기화"""
        collection_configs = [
            {
                "name": "intent_patterns",
                "description": "사용자 의도 분석을 위한 패턴 컬렉션"
            },
            {
                "name": "conversation_history",
                "description": "과거 대화 기록 컬렉션"
            },
            {
                "name": "domain_knowledge",
                "description": "도메인 특화 지식 컬렉션"
            },
            {
                "name": "api_documentation",
                "description": "API 사용법 및 문서 컬렉션"
            },
            {
                "name": "user_preferences",
                "description": "사용자 선호도 및 설정 컬렉션"
            }
        ]
        
        for config in collection_configs:
            try:
                # 기존 컬렉션이 있는지 확인 후 재사용 또는 생성
                try:
                    collection = self.chroma_client.get_collection(
                        name=config["name"],
                        embedding_function=self.openai_ef
                    )
                    print(f"  🔄 기존 컬렉션 재사용: {config['name']}")
                except:
                    # 컬렉션이 없으면 새로 생성
                    collection = self.chroma_client.create_collection(
                        name=config["name"],
                        embedding_function=self.openai_ef,
                        metadata={"description": config["description"]}
                    )
                    print(f"  📚 새 컬렉션 생성: {config['name']}")
                
                self.collections[config["name"]] = collection
                
            except Exception as e:
                print(f"  ❌ 컬렉션 초기화 실패: {config['name']} - {e}")
                # 실패해도 계속 진행 (다른 컬렉션들은 정상 작동하도록)
    
    def add_intent_patterns(self, patterns: List[Dict]):
        """의도 분석 패턴 추가"""
        try:
            if "intent_patterns" not in self.collections:
                print("의도 패턴 컬렉션이 없습니다. 건너뜁니다.")
                return
                
            collection = self.collections["intent_patterns"]
            
            documents = []
            metadatas = []
            ids = []
            
            for i, pattern in enumerate(patterns):
                doc_id = f"intent_pattern_{i}_{uuid.uuid4().hex[:8]}"
                
                documents.append(pattern["text"])
                metadatas.append({
                    "intent": pattern["intent"],
                    "apis": json.dumps(pattern["apis"]),
                    "confidence": pattern["confidence"],
                    "pattern_type": pattern.get("type", "example"),
                    "created_at": datetime.now().isoformat()
                })
                ids.append(doc_id)
            
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"의도 패턴 {len(patterns)}개 추가됨")
            
        except Exception as e:
            print(f"의도 패턴 추가 실패: {e}")
            # 실패해도 계속 진행
    
    def search_intent_patterns(self, query: str, top_k: int = 5) -> List[Dict]:
        """의도 패턴 검색"""
        try:
            if "intent_patterns" not in self.collections:
                print("의도 패턴 컬렉션이 없습니다. 빈 결과 반환.")
                return []
                
            collection = self.collections["intent_patterns"]
            
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            patterns = []
            for i in range(len(results["documents"][0])):
                patterns.append({
                    "text": results["documents"][0][i],
                    "intent": results["metadatas"][0][i]["intent"],
                    "apis": json.loads(results["metadatas"][0][i]["apis"]),
                    "confidence": float(results["metadatas"][0][i]["confidence"]),
                    "similarity": 1 - results["distances"][0][i],
                    "type": results["metadatas"][0][i]["pattern_type"]
                })
            
            return patterns
            
        except Exception as e:
            print(f"의도 패턴 검색 실패: {e}")
            return []
    
    def add_conversation_record(self, user_input: str, assistant_response: str, 
                              intent: str, success: bool, metadata: Dict = None):
        """대화 기록 추가"""
        try:
            if "conversation_history" not in self.collections:
                print("대화 기록 컬렉션이 없습니다. 건너뜁니다.")
                return
                
            collection = self.collections["conversation_history"]
            
            # 대화 텍스트 구성
            conversation_text = f"User: {user_input}\nAssistant: {assistant_response}"
            
            doc_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            record_metadata = {
                "user_input": user_input,
                "assistant_response": assistant_response[:500],  # 응답 길이 제한
                "intent": intent,
                "success": success,
                "timestamp": datetime.now().isoformat(),
                "response_length": len(assistant_response)
            }
            
            # 추가 메타데이터 병합
            if metadata:
                record_metadata.update(metadata)
            
            collection.add(
                documents=[conversation_text],
                metadatas=[record_metadata],
                ids=[doc_id]
            )
            
        except Exception as e:
            print(f"대화 기록 추가 실패: {e}")
    
    def search_conversation_history(self, query: str, intent_filter: str = None, 
                                  success_only: bool = False, top_k: int = 5) -> List[Dict]:
        """대화 기록 검색"""
        try:
            if "conversation_history" not in self.collections:
                print("대화 기록 컬렉션이 없습니다. 빈 결과 반환.")
                return []
                
            collection = self.collections["conversation_history"]
            
            # 필터 조건 구성
            where_filter = {}
            if intent_filter:
                where_filter["intent"] = intent_filter
            if success_only:
                where_filter["success"] = True
            
            query_params = {
                "query_texts": [query],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"]
            }
            
            if where_filter:
                query_params["where"] = where_filter
            
            results = collection.query(**query_params)
            
            conversations = []
            for i in range(len(results["documents"][0])):
                conversations.append({
                    "conversation": results["documents"][0][i],
                    "user_input": results["metadatas"][0][i]["user_input"],
                    "assistant_response": results["metadatas"][0][i]["assistant_response"],
                    "intent": results["metadatas"][0][i]["intent"],
                    "success": results["metadatas"][0][i]["success"],
                    "timestamp": results["metadatas"][0][i]["timestamp"],
                    "similarity": 1 - results["distances"][0][i]
                })
            
            return conversations
            
        except Exception as e:
            print(f"대화 기록 검색 실패: {e}")
            return []
    
    def add_domain_knowledge(self, knowledge_items: List[Dict]):
        """도메인 지식 추가"""
        try:
            if "domain_knowledge" not in self.collections:
                print("도메인 지식 컬렉션이 없습니다. 건너뜁니다.")
                return
                
            collection = self.collections["domain_knowledge"]
            
            documents = []
            metadatas = []
            ids = []
            
            for item in knowledge_items:
                doc_id = f"knowledge_{uuid.uuid4().hex[:8]}"
                
                documents.append(item["content"])
                metadatas.append({
                    "title": item.get("title", ""),
                    "category": item.get("category", "general"),
                    "source": item.get("source", "manual"),
                    "importance": item.get("importance", 1.0),
                    "created_at": datetime.now().isoformat(),
                    "tags": json.dumps(item.get("tags", []))
                })
                ids.append(doc_id)
            
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"도메인 지식 {len(knowledge_items)}개 추가됨")
            
        except Exception as e:
            print(f"도메인 지식 추가 실패: {e}")
    
    def search_domain_knowledge(self, query: str, category: str = None, 
                               min_importance: float = 0.0, top_k: int = 5) -> List[Dict]:
        """도메인 지식 검색"""
        try:
            if "domain_knowledge" not in self.collections:
                print("도메인 지식 컬렉션이 없습니다. 빈 결과 반환.")
                return []
                
            collection = self.collections["domain_knowledge"]
            
            # 필터 조건
            where_filter = {}
            if category:
                where_filter["category"] = category
            if min_importance > 0:
                where_filter["importance"] = {"$gte": min_importance}
            
            query_params = {
                "query_texts": [query],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"]
            }
            
            if where_filter:
                query_params["where"] = where_filter
            
            results = collection.query(**query_params)
            
            knowledge_items = []
            for i in range(len(results["documents"][0])):
                knowledge_items.append({
                    "content": results["documents"][0][i],
                    "title": results["metadatas"][0][i]["title"],
                    "category": results["metadatas"][0][i]["category"],
                    "source": results["metadatas"][0][i]["source"],
                    "importance": results["metadatas"][0][i]["importance"],
                    "tags": json.loads(results["metadatas"][0][i]["tags"]),
                    "similarity": 1 - results["distances"][0][i]
                })
            
            return knowledge_items
            
        except Exception as e:
            print(f"도메인 지식 검색 실패: {e}")
            return []
    
    def add_api_documentation(self, api_docs: List[Dict]):
        """API 문서 추가"""
        try:
            if "api_documentation" not in self.collections:
                print("API 문서 컬렉션이 없습니다. 건너뜁니다.")
                return
                
            collection = self.collections["api_documentation"]
            
            documents = []
            metadatas = []
            ids = []
            
            for doc in api_docs:
                doc_id = f"api_doc_{uuid.uuid4().hex[:8]}"
                
                # API 문서 텍스트 구성
                doc_text = f"API: {doc['name']}\n"
                doc_text += f"Description: {doc['description']}\n"
                if doc.get('parameters'):
                    doc_text += f"Parameters: {json.dumps(doc['parameters'])}\n"
                if doc.get('examples'):
                    doc_text += f"Examples: {doc['examples']}"
                
                documents.append(doc_text)
                metadatas.append({
                    "api_name": doc["name"],
                    "endpoint": doc.get("endpoint", ""),
                    "method": doc.get("method", "GET"),
                    "category": doc.get("category", "general"),
                    "version": doc.get("version", "1.0"),
                    "deprecated": doc.get("deprecated", False),
                    "created_at": datetime.now().isoformat()
                })
                ids.append(doc_id)
            
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"API 문서 {len(api_docs)}개 추가됨")
            
        except Exception as e:
            print(f"API 문서 추가 실패: {e}")
    
    def search_api_documentation(self, query: str, api_category: str = None, 
                                method: str = None, top_k: int = 3) -> List[Dict]:
        """API 문서 검색"""
        try:
            if "api_documentation" not in self.collections:
                print("API 문서 컬렉션이 없습니다. 빈 결과 반환.")
                return []
                
            collection = self.collections["api_documentation"]
            
            # 필터 조건
            where_filter = {"deprecated": False}  # 비추천 API 제외
            if api_category:
                where_filter["category"] = api_category
            if method:
                where_filter["method"] = method.upper()
            
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            api_docs = []
            for i in range(len(results["documents"][0])):
                api_docs.append({
                    "content": results["documents"][0][i],
                    "api_name": results["metadatas"][0][i]["api_name"],
                    "endpoint": results["metadatas"][0][i]["endpoint"],
                    "method": results["metadatas"][0][i]["method"],
                    "category": results["metadatas"][0][i]["category"],
                    "similarity": 1 - results["distances"][0][i]
                })
            
            return api_docs
            
        except Exception as e:
            print(f"API 문서 검색 실패: {e}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """컬렉션 통계 정보"""
        stats = {}
        
        for name, collection in self.collections.items():
            try:
                count = collection.count()
                stats[name] = {
                    "document_count": count,
                    "status": "active" if count > 0 else "empty"
                }
            except Exception as e:
                stats[name] = {
                    "document_count": 0,
                    "status": f"error: {str(e)}"
                }
        
        return stats
    
    def cleanup_old_conversations(self, days_old: int = 30):
        """오래된 대화 기록 정리"""
        try:
            from datetime import timedelta
            
            cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
            
            collection = self.collections["conversation_history"]
            
            # 오래된 기록 조회
            old_records = collection.query(
                query_texts=["dummy"],  # 더미 쿼리
                n_results=1000,  # 대량 조회
                where={"timestamp": {"$lt": cutoff_date}},
                include=["metadatas"]
            )
            
            # 삭제 실행 (실제로는 ChromaDB의 delete 기능 사용)
            # 여기서는 로그만 출력
            count = len(old_records.get("ids", []))
            print(f"정리할 오래된 대화 기록: {count}개 ({days_old}일 이전)")
            
        except Exception as e:
            print(f"대화 기록 정리 실패: {e}")

def initialize_default_knowledge():
    """기본 지식베이스 초기화"""
    kb = KnowledgeBase()
    
    # 기본 의도 패턴 추가
    default_patterns = [
        {"text": "날씨 어때?", "intent": "weather_query", "apis": ["weather"], "confidence": 0.9, "type": "simple"},
        {"text": "일정 확인해줘", "intent": "calendar_query", "apis": ["calendar"], "confidence": 0.9, "type": "simple"},
        {"text": "파일 찾아줘", "intent": "file_search", "apis": ["file"], "confidence": 0.9, "type": "simple"},
        {"text": "알림 보내줘", "intent": "notification_send", "apis": ["notification"], "confidence": 0.9, "type": "simple"},
        {"text": "날씨 확인하고 알려줘", "intent": "weather_and_notify", "apis": ["weather", "notification"], "confidence": 0.8, "type": "complex"}
    ]
    
    kb.add_intent_patterns(default_patterns)
    
    # 기본 도메인 지식 추가
    default_knowledge = [
        {
            "title": "날씨 API 사용법",
            "content": "날씨 API는 도시명을 받아 현재 날씨와 예보를 제공합니다. 지원 도시: 서울, 부산, 인천, 대구, 광주",
            "category": "api_usage",
            "importance": 0.8,
            "tags": ["weather", "api", "cities"]
        },
        {
            "title": "일정 관리 기능",
            "content": "일정 API는 오늘/내일 일정 조회, 새 일정 생성, 빈 시간 확인 등의 기능을 제공합니다.",
            "category": "api_usage",
            "importance": 0.8,
            "tags": ["calendar", "scheduling", "events"]
        },
        {
            "title": "파일 검색 가이드",
            "content": "파일 API는 키워드 기반 검색, 파일 타입 필터링, 디렉토리 탐색 기능을 제공합니다.",
            "category": "api_usage",
            "importance": 0.7,
            "tags": ["files", "search", "documents"]
        }
    ]
    
    kb.add_domain_knowledge(default_knowledge)
    
    # API 문서 추가
    api_docs = [
        {
            "name": "get_weather",
            "description": "지정된 도시의 현재 날씨 정보를 조회합니다",
            "endpoint": "/weather/{city}",
            "method": "GET",
            "category": "weather",
            "parameters": {"city": "도시명 (한글 또는 영문)"},
            "examples": "서울 날씨, 부산 날씨"
        },
        {
            "name": "get_schedule",
            "description": "사용자의 일정 정보를 조회합니다",
            "endpoint": "/calendar/today",
            "method": "GET", 
            "category": "calendar",
            "examples": "오늘 일정, 내일 스케줄"
        },
        {
            "name": "search_files",
            "description": "키워드로 파일을 검색합니다",
            "endpoint": "/files/search",
            "method": "GET",
            "category": "file",
            "parameters": {"query": "검색 키워드", "file_type": "파일 타입 (선택)"},
            "examples": "프로젝트 문서, API 명세서"
        },
        {
            "name": "send_notification",
            "description": "지정된 채널로 알림을 발송합니다",
            "endpoint": "/notifications/slack",
            "method": "POST",
            "category": "notification",
            "parameters": {"channel": "채널명", "message": "메시지 내용"},
            "examples": "팀에게 알림, 이메일 발송"
        }
    ]
    
    kb.add_api_documentation(api_docs)
    
    print("기본 지식베이스 초기화 완료!")
    return kb

def test_knowledge_base():
    """지식베이스 테스트"""
    print("=" * 60)
    print("RAG 지식베이스 테스트")
    print("=" * 60)
    
    # 지식베이스 초기화
    kb = initialize_default_knowledge()
    
    # 통계 정보 출력
    print(f"\n📊 컬렉션 통계:")
    stats = kb.get_collection_stats()
    for name, stat in stats.items():
        print(f"  {name}: {stat['document_count']}개 문서 ({stat['status']})")
    
    # 테스트 검색들
    test_queries = [
        ("의도 패턴", "날씨 확인해줘", "intent_patterns"),
        ("도메인 지식", "API 사용법", "domain_knowledge"),
        ("API 문서", "날씨 정보", "api_documentation")
    ]
    
    print(f"\n🔍 검색 테스트:")
    for test_name, query, search_type in test_queries:
        print(f"\n[{test_name}] '{query}' 검색:")
        
        if search_type == "intent_patterns":
            results = kb.search_intent_patterns(query, top_k=3)
            for result in results:
                print(f"  - {result['text']} (의도: {result['intent']}, 유사도: {result['similarity']:.3f})")
        
        elif search_type == "domain_knowledge":
            results = kb.search_domain_knowledge(query, top_k=3)
            for result in results:
                print(f"  - {result['title']} (카테고리: {result['category']}, 유사도: {result['similarity']:.3f})")
        
        elif search_type == "api_documentation":
            results = kb.search_api_documentation(query, top_k=3)
            for result in results:
                print(f"  - {result['api_name']} (엔드포인트: {result['endpoint']}, 유사도: {result['similarity']:.3f})")
    
    print(f"\n" + "=" * 60)
    print("지식베이스 테스트 완료!")

if __name__ == "__main__":
    test_knowledge_base() 