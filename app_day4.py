import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. 페이지 및 데이터 설정
# ==========================================
st.set_page_config(page_title="스마트 호텔 챗봇", page_icon="🤖", layout="wide")

# 실습했던 호텔 FAQ 데이터 15개
faq_data = [
    {"q": "체크인 시간이 어떻게 되나요?", "a": "오후 3시부터입니다."},
    {"q": "체크아웃 시간은 언제인가요?", "a": "오전 11시까지입니다."},
    {"q": "조식은 어디서 먹나요?", "a": "1층 레스토랑에서 제공됩니다."},
    {"q": "와이파이 비밀번호가 뭔가요?", "a": "객실 내 전화기에 적혀있습니다."},
    {"q": "주차장이 있나요?", "a": "지하 1층~3층에 무료 주차 가능합니다."},
    {"q": "수영장은 몇 시까지 이용 가능한가요?", "a": "밤 10시까지 운영합니다."},
    {"q": "객실 내 음식 배달이 가능한가요?", "a": "로비에서 수령 시 가능합니다."},
    {"q": "주변에 편의점이 있나요?", "a": "호텔 정문 우측 도보 1분 거리에 있습니다."},
    {"q": "공항 셔틀 서비스가 있나요?", "a": "오전 8시부터 1시간 간격으로 운행합니다."},
    {"q": "반려동물 동반이 가능한가요?", "a": "전용 객실 예약 시 가능합니다."},
    {"q": "객실 내 흡연이 가능한가요?", "a": "전 객실 금연이며, 1층 흡연구역을 이용해주세요."},
    {"q": "세탁 서비스를 이용하고 싶어요.", "a": "객실 내 세탁물 봉투를 이용해 데스크에 맡겨주세요."},
    {"q": "레이트 체크아웃이 가능한가요?", "a": "추가 요금 지불 시 오후 2시까지 가능합니다."},
    {"q": "객실 내 취사가 가능한가요?", "a": "화재 위험으로 취사는 불가합니다."},
    {"q": "어린이 요금은 어떻게 되나요?", "a": "미취학 아동은 무료입니다."}
]

questions = [item["q"] for item in faq_data]
answers = [item["a"] for item in faq_data]

# ==========================================
# 2. AI 모델 및 FAISS 검색기 로딩 (캐싱)
# ==========================================
# st.cache_resource를 사용해 앱이 새로고침되어도 모델을 다시 불러오지 않게 합니다.
@st.cache_resource
def load_chatbot_engine():
    # SBERT 모델 로드
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    
    # FAQ 질문들을 벡터로 변환 (임베딩)
    embeddings = model.encode(questions).astype('float32')
    
    # 코사인 유사도를 구하기 위해 벡터 정규화(L2) 후 FAISS 인덱스 생성
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) # 내적(Inner Product) 기반 검색
    index.add(embeddings)
    
    return model, index

with st.spinner("AI 챗봇 엔진을 깨우는 중입니다... (약 10~20초 소요)"):
    model, index = load_chatbot_engine()

# ==========================================
# 3. 사이드바 구성 (FAQ 목록)
# ==========================================
with st.sidebar:
    st.header("📋 지원 가능한 FAQ 목록")
    st.info("아래 질문들과 비슷한 의미로 자유롭게 물어보세요!")
    for q in questions:
        st.markdown(f"- {q}")

# ==========================================
# 4. 메인 채팅 UI 구성
# ==========================================
st.title("🤖 스마트 호텔 AI 컨시어지")
st.markdown("SBERT 문장 임베딩과 FAISS 벡터 검색을 활용한 인공지능 챗봇입니다.")

# 세션 상태(Session State)를 이용해 채팅 기록 저장
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 호텔 이용과 관련하여 무엇이든 물어보세요."}
    ]

# 이전 채팅 기록들을 화면에 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# 5. 사용자 입력 및 AI 응답 처리
# ==========================================
# 사용자가 질문을 입력하면 동작
if user_query := st.chat_input("질문을 입력해주세요... (예: 늦게 나가도 되나요?)"):
    
    # 1. 사용자 질문을 화면에 표시하고 기록에 추가
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # 2. AI 응답 생성 (FAISS 검색)
    with st.chat_message("assistant"):
        # 질문을 벡터로 변환
        query_vec = model.encode([user_query]).astype('float32')
        faiss.normalize_L2(query_vec)
        
        # 가장 유사한 질문 1개 찾기
        scores, indices = index.search(query_vec, 1)
        best_score = float(scores[0][0])
        best_idx = indices[0][0]
        
        # 3. 유사도 점수에 따른 분기 처리 (Threshold: 0.4)
        if best_score >= 0.4:
            matched_q = questions[best_idx]
            answer = answers[best_idx]
            
            # 답변과 함께 매칭 정보(유사도 점수)를 구성
            response_text = f"{answer}\n\n---\n*💡 **참고 정보** (매칭률: {best_score:.2f})*\n*매칭된 원본 질문: {matched_q}*"
            st.markdown(response_text)
        else:
            response_text = f"죄송합니다. 해당 질문에 대한 관련 FAQ가 없습니다. (최고 매칭률: {best_score:.2f} < 0.4)\n\n프론트 데스크(내선 0번)로 문의해 주시면 친절히 안내해 드리겠습니다."
            st.warning(response_text)
            
    # AI의 응답을 기록에 추가
    st.session_state.messages.append({"role": "assistant", "content": response_text})