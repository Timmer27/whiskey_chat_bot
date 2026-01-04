import os
import requests
import streamlit as st
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=False)

API_BASE = os.getenv("BACKEND_URL", "http://localhost:8000")  
CHAT_URL = f"{API_BASE}/chat/whiskey"

st.set_page_config(page_title="Whisky CHAT BOT", layout="wide")
st.title("Whisky CHAT BOT")

q = st.text_input(
    "질문을 입력하세요",
    placeholder="예: 짠맛/미네랄 + 과일향도 있는 위스키 추천해줘",
)

def ask_api(question: str) -> str:
    """FastAPI /chat/whiskey 호출 후 answer 문자열만 반환"""
    r = requests.post(
        CHAT_URL,
        json={"question": question},
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    return str(data)

with st.form("ask_form", clear_on_submit=False):
    submitted = st.form_submit_button("질문하기", use_container_width=True)

if submitted:
    if not q.strip():
        st.warning("질문을 입력해 주세요.")
        st.stop()

    with st.spinner("답변 생성 중..."):
        try:
            answer = ask_api(q.strip())
        except requests.exceptions.ConnectionError:
            st.error(f"API 서버에 연결할 수 없습니다: {CHAT_URL}\n\nFastAPI가 실행 중인지 확인하세요.")
            st.stop()
        except requests.exceptions.Timeout:
            st.error("요청 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요.")
            st.stop()
        except requests.HTTPError as e:
            st.error(f"서버 오류: {e}\n\n응답: {getattr(e.response, 'text', '')}")
            st.stop()
        except Exception as e:
            st.error(f"알 수 없는 오류: {e}")
            st.stop()

    st.subheader("답변")
    st.write(answer)
