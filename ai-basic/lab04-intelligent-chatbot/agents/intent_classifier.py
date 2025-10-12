"""
Lab 4 - Intent Classification Agent
RAG 기반 사용자 의도 분석 및 API 라우팅 에이전트
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.config import validate_api_keys, OPENAI_API_KEY, CHAT_MODEL
from shared.utils import ChatUtils, ChromaUtils
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re
import chromadb
from chromadb.utils import embedding_functions

class IntentClassifier:
    """RAG 기반 사용자 의도 분석 에이전트"""
    
    def __init__(self):
        """Intent Classifier 초기화"""
        self.name = "Intent Classification Agent"
        self.version = "1.0.0"
        
        # OpenAI 클라이언트 초기화
        self.chat_utils = ChatUtils()
        
        # ChromaDB 클라이언트 설정
        self.chroma_client = chromadb.PersistentClient(path="./data/intent_db")
        
        # OpenAI 임베딩 함수 설정 (SSL 검증 비활성화)
        self.openai_ef = ChromaUtils.create_openai_embedding_function()
        
        # 의도 분석 지식베이스 초기화
        self.initialize_knowledge_base()
        
        print(f"{self.name} 초기화 완료")
    
    def initialize_knowledge_base(self):
        """의도 분석을 위한 지식베이스 구축"""
        try:
            # 기존 컬렉션 삭제 후 재생성
            try:
                self.chroma_client.delete_collection("intent_patterns")
            except:
                pass
            
            # 새 컬렉션 생성
            self.collection = self.chroma_client.create_collection(
                name="intent_patterns",
                embedding_function=self.openai_ef
            )
            
            # 훈련 패턴 데이터
            training_patterns = self.get_training_patterns()
            
            # 패턴들을 벡터 DB에 저장
            documents = []
            metadatas = []
            ids = []
            
            for pattern in training_patterns:
                documents.append(pattern["text"])
                metadatas.append({
                    "intent": pattern["intent"],
                    "apis": json.dumps(pattern["apis"]),
                    "confidence": pattern["confidence"],
                    "example_type": pattern["type"]
                })
                ids.append(f"pattern_{len(ids)}")
            
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"의도 분석 지식베이스 구축 완료: {len(training_patterns)}개 패턴")
            
        except Exception as e:
            print(f"지식베이스 초기화 실패: {e}")
    
    def get_training_patterns(self) -> List[Dict]:
        """의도 분석용 훈련 패턴 데이터"""
        return [
            # 날씨 관련 패턴
            {"text": "오늘 날씨 어때?", "intent": "weather_query", "apis": ["weather"], "confidence": 0.95, "type": "simple"},
            {"text": "서울 날씨 확인해줘", "intent": "weather_query", "apis": ["weather"], "confidence": 0.9, "type": "simple"},
            {"text": "비 올까?", "intent": "weather_query", "apis": ["weather"], "confidence": 0.85, "type": "simple"},
            {"text": "내일 우산 가져가야 할까?", "intent": "weather_query", "apis": ["weather"], "confidence": 0.8, "type": "implicit"},
            
            # 일정 관련 패턴  
            {"text": "오늘 일정 확인해줘", "intent": "calendar_query", "apis": ["calendar"], "confidence": 0.95, "type": "simple"},
            {"text": "내일 회의 있어?", "intent": "calendar_query", "apis": ["calendar"], "confidence": 0.9, "type": "simple"},
            {"text": "다음 주 스케줄", "intent": "calendar_query", "apis": ["calendar"], "confidence": 0.85, "type": "simple"},
            {"text": "15시에 미팅 잡아줘", "intent": "calendar_create", "apis": ["calendar"], "confidence": 0.9, "type": "action"},
            
            # 파일 관련 패턴
            {"text": "프로젝트 문서 찾아줘", "intent": "file_search", "apis": ["file"], "confidence": 0.9, "type": "simple"},
            {"text": "회의록 어디 있어?", "intent": "file_search", "apis": ["file"], "confidence": 0.85, "type": "simple"},
            {"text": "API 명세서 보여줘", "intent": "file_search", "apis": ["file"], "confidence": 0.88, "type": "simple"},
            {"text": "문서 요약해줘", "intent": "file_process", "apis": ["file"], "confidence": 0.8, "type": "action"},
            
            # 알림 관련 패턴
            {"text": "팀에게 알려줘", "intent": "notification_send", "apis": ["notification"], "confidence": 0.9, "type": "action"},
            {"text": "슬랙에 메시지 보내줘", "intent": "notification_send", "apis": ["notification"], "confidence": 0.95, "type": "action"},
            {"text": "이메일 발송해줘", "intent": "notification_send", "apis": ["notification"], "confidence": 0.9, "type": "action"},
            {"text": "모든 팀원에게 공지해줘", "intent": "notification_broadcast", "apis": ["notification"], "confidence": 0.85, "type": "action"},
            
            # 복합 의도 패턴
            {"text": "날씨 확인하고 팀에게 알려줘", "intent": "weather_and_notify", "apis": ["weather", "notification"], "confidence": 0.9, "type": "complex"},
            {"text": "오늘 일정 보고 회의실 예약해줘", "intent": "calendar_and_action", "apis": ["calendar", "notification"], "confidence": 0.85, "type": "complex"},
            {"text": "프로젝트 문서 찾아서 요약해서 팀에게 공유해줘", "intent": "file_and_notify", "apis": ["file", "notification"], "confidence": 0.8, "type": "complex"},
            {"text": "내일 날씨 확인하고 일정 조정 필요하면 알려줘", "intent": "weather_calendar_notify", "apis": ["weather", "calendar", "notification"], "confidence": 0.75, "type": "complex"},
            
            # 일반적인 대화 패턴
            {"text": "안녕하세요", "intent": "greeting", "apis": [], "confidence": 0.95, "type": "social"},
            {"text": "고마워", "intent": "thanks", "apis": [], "confidence": 0.9, "type": "social"},
            {"text": "도움말", "intent": "help", "apis": [], "confidence": 0.95, "type": "system"},
            {"text": "뭘 할 수 있어?", "intent": "capability_query", "apis": [], "confidence": 0.9, "type": "system"}
        ]
    
    def analyze_intent(self, user_input: str, context: Optional[Dict] = None) -> Dict:
        """사용자 입력의 의도 분석"""
        try:
            # 1. RAG 기반 유사 패턴 검색
            similar_patterns = self.find_similar_patterns(user_input, top_k=5)
            
            # 2. LLM 기반 의도 분석
            llm_analysis = self.llm_intent_analysis(user_input, similar_patterns, context)
            
            # 3. 종합 결과 생성
            final_result = self.combine_analysis_results(user_input, similar_patterns, llm_analysis)
            
            # 4. 분석 결과 저장 (학습용)
            self.save_analysis_result(user_input, final_result)
            
            return final_result
            
        except Exception as e:
            print(f"의도 분석 실패: {e}")
            return {
                "intent": "unknown",
                "apis": [],
                "confidence": 0.0,
                "reasoning": f"분석 중 오류 발생: {str(e)}",
                "suggested_response": "죄송합니다. 요청을 이해하지 못했습니다. 다시 말씀해 주세요."
            }
    
    def find_similar_patterns(self, user_input: str, top_k: int = 5) -> List[Dict]:
        """RAG 기반 유사 패턴 검색"""
        try:
            results = self.collection.query(
                query_texts=[user_input],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            similar_patterns = []
            for i in range(len(results["documents"][0])):
                similar_patterns.append({
                    "text": results["documents"][0][i],
                    "intent": results["metadatas"][0][i]["intent"],
                    "apis": json.loads(results["metadatas"][0][i]["apis"]),
                    "confidence": float(results["metadatas"][0][i]["confidence"]),
                    "similarity": 1 - results["distances"][0][i],  # 거리를 유사도로 변환
                    "type": results["metadatas"][0][i]["example_type"]
                })
            
            return similar_patterns
            
        except Exception as e:
            print(f"유사 패턴 검색 실패: {e}")
            return []
    
    def llm_intent_analysis(self, user_input: str, similar_patterns: List[Dict], context: Optional[Dict] = None) -> Dict:
        """LLM 기반 의도 분석"""
        
        # 유사 패턴 정보 구성
        pattern_info = "\n".join([
            f"- '{p['text']}' → 의도: {p['intent']}, API: {p['apis']}, 유사도: {p['similarity']:.2f}"
            for p in similar_patterns[:3]
        ])
        
        # 컨텍스트 정보
        context_info = ""
        if context:
            context_info = f"대화 컨텍스트: {json.dumps(context, ensure_ascii=False, indent=2)}"
        
        prompt = f"""
사용자 입력을 분석하여 의도와 필요한 API, 그리고 구체적인 매개변수를 추출해주세요.

사용자 입력: "{user_input}"

{context_info}

유사한 패턴들:
{pattern_info}

사용 가능한 API:
- weather: 날씨 정보 조회
- calendar: 일정 관리 (조회/생성/수정)
- file: 파일 검색 및 관리
- notification: 알림 발송 (Slack, Email, SMS)

매개변수 추출 가이드라인:
1. **검색어/키워드**: "프로젝트 문서", "API 명세서", "회의록" 등
2. **채널 유형**: "이메일로", "슬랙에", "SMS로" 등에서 channel 추출
3. **도시명**: "서울", "부산", "인천" 등
4. **시간 정보**: "오늘", "내일", "15시", "3시" 등
5. **액션**: "찾아서", "생성해서", "확인하고" 등에서 action 추출
6. **수신자**: "팀에게", "관리자에게", "전체에게" 등

다음 JSON 형식으로 응답해주세요:
{{
    "intent": "분석된 의도 (예: weather_query, calendar_create, file_search, notification_send, file_and_notify 등)",
    "apis": ["필요한 API 목록"],
    "confidence": 0.0-1.0 사이의 신뢰도,
    "reasoning": "분석 근거 설명",
    "parameters": {{
        "query": "검색어 또는 키워드",
        "channel": "slack/email/sms 중 하나",
        "action": "search/create/send/query 중 하나",
        "city": "도시명",
        "time": "시간 정보",
        "recipient": "수신자 정보",
        "title": "제목 또는 이벤트명",
        "file_type": "문서/이미지/코드 등",
        "date": "날짜 정보"
    }},
    "action_sequence": ["수행할 작업 순서"]
}}

예시:
입력: "프로젝트 문서 찾아서 요약해서 이메일로 보내줘"
→ parameters: {{"query": "프로젝트 문서", "channel": "email", "action": "search", "recipient": "전체"}}

입력: "내일 3시에 회의 잡아줘"  
→ parameters: {{"action": "create", "time": "15:00", "date": "내일", "title": "회의"}}

입력: "서울 날씨 확인하고 팀에게 알려줘"
→ parameters: {{"city": "서울", "action": "query", "channel": "slack", "recipient": "팀"}}
"""
        
        try:
            print(f"🤔 LLM 의도 분석 시작... (프롬프트 길이: {len(prompt)} 문자)")
            
            # 간단한 프롬프트로 빠른 분석 (타임아웃 문제 해결)
            simplified_prompt = f"""
사용자 입력을 분석하여 의도와 필요한 API를 추출해주세요.

입력: "{user_input}"

사용 가능한 API:
- weather: 날씨 정보 조회
- calendar: 일정 관리 (조회/생성/수정)
- file: 파일 검색 및 관리
- notification: 알림 발송 (Slack, Email, SMS)

JSON 형식으로 응답:
{{
    "intent": "weather_query|calendar_query|file_search|notification_send|weather_and_notify 중 하나",
    "apis": ["weather", "calendar", "file", "notification 중 필요한 것들"],
    "confidence": 0.8,
    "parameters": {{
        "city": "도시명 (있으면)",
        "channel": "slack|email|sms (있으면)",
        "action": "query|search|send (있으면)",
        "query": "검색어 (있으면)"
    }}
}}
"""
            
            response = self.chat_utils.get_chat_response(simplified_prompt)
            print(f"✅ LLM 응답 받음 (길이: {len(response)} 문자)")
            
            # JSON 파싱
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                print(f"✅ JSON 파싱 성공: {result.get('intent', 'unknown')}")
                return result
            else:
                print(f"❌ JSON 형식 응답을 찾을 수 없음. 응답: {response[:200]}...")
                raise ValueError("JSON 형식 응답을 찾을 수 없음")
                
        except Exception as e:
            print(f"❌ LLM 의도 분석 실패: {e}")
            
            # 폴백: 간단한 키워드 기반 분석
            fallback_result = self.simple_keyword_analysis(user_input)
            print(f"🔄 폴백 분석 결과: {fallback_result.get('intent', 'unknown')}")
            
            return fallback_result
    
    def simple_keyword_analysis(self, user_input: str) -> Dict:
        """키워드 기반 간단한 의도 분석 (폴백용)"""
        user_input_lower = user_input.lower()
        
        # 키워드 매핑
        keyword_patterns = {
            "weather": {
                "keywords": ["날씨", "기온", "비", "눈", "바람", "온도", "weather"],
                "apis": ["weather"]
            },
            "calendar": {
                "keywords": ["일정", "회의", "약속", "미팅", "스케줄", "schedule", "meeting"],
                "apis": ["calendar"]
            },
            "file": {
                "keywords": ["파일", "문서", "자료", "보고서", "데이터", "file", "document"],
                "apis": ["file"]
            },
            "notification": {
                "keywords": ["알림", "알려", "보내", "전송", "메시지", "이메일", "슬랙", "notification"],
                "apis": ["notification"]
            }
        }
        
        detected_intents = []
        detected_apis = []
        
        for intent, pattern in keyword_patterns.items():
            if any(keyword in user_input_lower for keyword in pattern["keywords"]):
                detected_intents.append(intent)
                detected_apis.extend(pattern["apis"])
        
        # 복합 의도 처리
        if len(detected_intents) > 1:
            intent = "_and_".join(detected_intents)
        elif detected_intents:
            intent = detected_intents[0] + "_query"
        else:
            intent = "general_conversation"
        
        # 기본 파라미터 추출
        parameters = {}
        
        # 도시 추출 (확장된 목록)
        cities = [
            "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
            "수원", "고양", "용인", "성남", "안산", "안양", "남양주", "평택",
            "시흥", "김포", "하남", "오산", "구리", "군포", "의왕", "과천",
            "의정부", "동두천", "안성", "포천", "여주", "양평", "가평", "연천",
            "춘천", "원주", "강릉", "태백", "삼척", "정선", "속초", "고성",
            "양양", "인제", "횡성", "영월", "평창", "화천", "양구", "철원",
            "청주", "충주", "제천", "보은", "옥천", "영동", "증평", "진천",
            "괴산", "음성", "단양", "천안", "공주", "보령", "아산", "서산",
            "논산", "계룡", "당진", "금산", "부여", "서천", "청양", "홍성",
            "예산", "태안", "전주", "군산", "익산", "정읍", "남원", "김제",
            "완주", "진안", "무주", "장수", "임실", "순창", "고창", "부안",
            "목포", "여수", "순천", "나주", "광양", "담양", "곡성", "구례",
            "고흥", "보성", "화순", "장흥", "강진", "해남", "영암", "무안",
            "함평", "영광", "장성", "완도", "진도", "신안", "포항", "경주",
            "김천", "안동", "구미", "영주", "영천", "상주", "문경", "경산",
            "군위", "의성", "청송", "영양", "영덕", "청도", "고령", "성주",
            "칠곡", "예천", "봉화", "울진", "울릉", "독도"
        ]
        
        for city in cities:
            if city in user_input:
                parameters["city"] = city
                break
        
        # 도시가 없으면 기본값 설정
        if "weather" in detected_intents and "city" not in parameters:
            parameters["city"] = "서울"  # 기본값
            print(f"🔄 [자동 도시 설정] 기본값 '서울' 사용")
        
        # 채널 추출
        if "이메일" in user_input_lower or "email" in user_input_lower:
            parameters["channel"] = "email"
        elif "슬랙" in user_input_lower or "slack" in user_input_lower:
            parameters["channel"] = "slack"
        elif "문자" in user_input_lower or "sms" in user_input_lower:
            parameters["channel"] = "sms"
        
        # 액션 추출
        if "확인" in user_input or "조회" in user_input or "알려" in user_input:
            parameters["action"] = "query"
        elif "생성" in user_input or "만들" in user_input or "잡아" in user_input:
            parameters["action"] = "create"
        elif "보내" in user_input or "전송" in user_input or "알려" in user_input:
            parameters["action"] = "send"
        
        return {
            "intent": intent,
            "apis": list(set(detected_apis)),
            "confidence": 0.7,  # 키워드 기반이므로 중간 신뢰도
            "reasoning": f"키워드 기반 분석: {detected_intents}",
            "parameters": parameters,
            "action_sequence": detected_apis,
            "user_input": user_input  # 디버깅용
        }
    
    def combine_analysis_results(self, user_input: str, similar_patterns: List[Dict], llm_analysis: Dict) -> Dict:
        """RAG와 LLM 결과를 종합하여 최종 결과 생성"""
        
        # 유사 패턴 기반 가중치 계산
        pattern_weight = 0.0
        if similar_patterns:
            top_pattern = similar_patterns[0]
            pattern_weight = top_pattern["similarity"] * top_pattern["confidence"]
        
        # LLM 분석 가중치
        llm_weight = llm_analysis.get("confidence", 0.0)
        
        # 가중 평균으로 신뢰도 계산
        final_confidence = (pattern_weight * 0.4 + llm_weight * 0.6)
        
        # API 결정 (LLM 우선, 유사 패턴 보조)
        final_apis = llm_analysis.get("apis", [])
        if not final_apis and similar_patterns:
            final_apis = similar_patterns[0]["apis"]
        
        result = {
            "user_input": user_input,
            "intent": llm_analysis.get("intent", "unknown"),
            "apis": final_apis,
            "confidence": final_confidence,
            "reasoning": llm_analysis.get("reasoning", ""),
            "parameters": llm_analysis.get("parameters", {}),
            "action_sequence": llm_analysis.get("action_sequence", []),
            "similar_patterns": [p["text"] for p in similar_patterns[:3]],
            "analysis_timestamp": datetime.now().isoformat(),
            "suggested_response": self.generate_response_template(llm_analysis.get("intent", "unknown"))
        }
        
        return result
    
    def generate_response_template(self, intent: str) -> str:
        """의도에 따른 응답 템플릿 생성"""
        templates = {
            "weather_query": "날씨 정보를 확인하겠습니다.",
            "calendar_query": "일정을 조회하겠습니다.", 
            "calendar_create": "일정을 생성하겠습니다.",
            "file_search": "파일을 검색하겠습니다.",
            "file_process": "파일을 처리하겠습니다.",
            "notification_send": "알림을 발송하겠습니다.",
            "greeting": "안녕하세요! 무엇을 도와드릴까요?",
            "thanks": "천만에요! 또 다른 도움이 필요하시면 말씀해 주세요.",
            "help": "날씨 확인, 일정 관리, 파일 검색, 알림 발송 등을 도와드릴 수 있습니다.",
            "unknown": "죄송합니다. 요청을 이해하지 못했습니다. 다시 말씀해 주세요."
        }
        
        return templates.get(intent, "요청을 처리하겠습니다.")
    
    def save_analysis_result(self, user_input: str, result: Dict):
        """분석 결과를 학습용으로 저장"""
        try:
            # 실제로는 별도 DB나 파일에 저장
            # 여기서는 간단히 로그만 남김
            print(f"[학습 데이터] '{user_input}' → {result['intent']} (신뢰도: {result['confidence']:.2f})")
        except Exception as e:
            print(f"분석 결과 저장 실패: {e}")
    
    def get_capabilities(self) -> Dict:
        """에이전트 능력 정보 반환"""
        return {
            "name": self.name,
            "version": self.version,
            "description": "RAG 기반 사용자 의도 분석 및 API 라우팅",
            "supported_intents": [
                "weather_query", "calendar_query", "calendar_create", 
                "file_search", "file_process", "notification_send",
                "greeting", "thanks", "help"
            ],
            "supported_apis": ["weather", "calendar", "file", "notification"],
            "features": [
                "자연어 의도 분석",
                "다중 API 요청 처리", 
                "RAG 기반 패턴 학습",
                "실시간 컨텍스트 이해"
            ]
        }

def test_intent_classifier():
    """Intent Classifier 테스트"""
    print("=" * 60)
    print("Intent Classification Agent 테스트")
    print("=" * 60)
    
    # API 키 검증
    if not validate_api_keys():
        print("API 키 설정을 확인해주세요.")
        return
    
    # Intent Classifier 초기화
    classifier = IntentClassifier()
    
    # 테스트 케이스들
    test_cases = [
        "오늘 서울 날씨 어때?",
        "내일 회의 일정 확인해줘", 
        "프로젝트 계획서 찾아줘",
        "팀에게 알림 보내줘",
        "날씨 확인하고 일정 조정 필요하면 슬랙에 알려줘",
        "안녕하세요",
        "뭘 할 수 있어?"
    ]
    
    print(f"\n{len(test_cases)}개 테스트 케이스로 의도 분석 테스트:")
    print("-" * 60)
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n[{i}] 입력: '{test_input}'")
        
        result = classifier.analyze_intent(test_input)
        
        print(f"    의도: {result['intent']}")
        print(f"    API: {result['apis']}")
        print(f"    신뢰도: {result['confidence']:.2f}")
        print(f"    근거: {result['reasoning'][:100]}...")
        if result['parameters']:
            print(f"    매개변수: {result['parameters']}")
        print(f"    응답: {result['suggested_response']}")
    
    print(f"\n" + "=" * 60)
    print("Intent Classification Agent 테스트 완료!")
    
    # 에이전트 능력 정보 출력
    capabilities = classifier.get_capabilities()
    print(f"\n📋 에이전트 정보:")
    print(f"  이름: {capabilities['name']}")
    print(f"  버전: {capabilities['version']}")
    print(f"  설명: {capabilities['description']}")
    print(f"  지원 의도: {len(capabilities['supported_intents'])}개")
    print(f"  지원 API: {capabilities['supported_apis']}")

if __name__ == "__main__":
    test_intent_classifier() 