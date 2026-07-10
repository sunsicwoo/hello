import streamlit as st
from google import genai
from PIL import Image

# 1. 스트림릿 페이지 설정
st.set_page_config(page_title="나사 결함 탐지 AI", layout="centered")
st.title("🔩 나사 결함 탐지 시스템 (Gemini AI)")
st.write("나사 사진을 업로드하면 Gemini AI가 결함(마모, 부러짐, 나사산 손상 등)을 분석합니다.")

# 2. 사이드바에 API 키 입력창 생성
st.sidebar.title("🔑 API 설정")
user_api_key = st.sidebar.text_input(
    "Google Gemini API 키를 입력하세요", 
    type="password", 
    placeholder="AIzaSy..."
)

# AI Studio 링크 안내
st.sidebar.markdown(
    "[Google AI Studio에서 API 키 발급받기](https://aistudio.google.com/)"
)

# 3. API 키 입력 여부 검증 및 클라이언트 초기화
client = None
if user_api_key:
    try:
        # 입력받은 API 키로 Gemini 클라이언트 초기화
        client = genai.Client(api_key=user_api_key)
        st.sidebar.success("API 키가 인식되었습니다!")
    except Exception as e:
        st.sidebar.error(f"클라이언트 초기화 실패: {e}")
else:
    st.sidebar.warning("서비스를 이용하려면 API 키를 입력해주세요.")

# 4. 파일 업로드 컴포넌트
uploaded_file = st.file_uploader("나사 이미지를 업로드하세요...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 업로드된 이미지를 화면에 표시
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드된 나사 이미지", use_container_width=True)
    
    st.write("---")
    st.subheader("🔍 AI 분석 결과")
    
    # 5. API 키가 없을 때 버튼 클릭/분석 막기
    if not client:
        st.error("보안 및 API 호출을 위해 왼쪽 사이드바에 Gemini API 키를 먼저 입력해야 합니다.")
    else:
        # 분석 실행 버튼
        if st.button("나사 결함 분석 시작"):
            with st.spinner("Gemini가 이미지를 정밀 분석하고 있습니다..."):
                try:
                    # AI에게 줄 역할 및 안내 프롬프트
                    prompt = """
                    당신은 정밀 제조 부품 검사 전문가입니다. 
                    제공된 나사(Screw/Bolt) 이미지를 정밀하게 분석하여 다음 양식에 맞춰 한국어로 답변해주세요:
                    
                    1. 결함 여부: [정상 / 결함 발견]
                    2. 결함 유형: (결함이 있다면 마모, 부식, 나사산 뭉개짐, 휘어짐 등 구체적으로 적어주세요. 정상이면 '없음')
                    3. 상세 분석 내용: (이미지에서 관찰되는 나사의 상태를 기술적으로 설명해주세요)
                    4. 조치 권고 사항: (폐기, 재사용 가능, 정밀 재검사 등)
                    """
                    
                    # 수정한 부분: gemini-1.5-flash 모델로 변경
                    response = client.models.generate_content(
                        model='gemini-1.5-flash',
                        contents=[image, prompt]
                    )
                    
                    # 결과 출력
                    st.success("분석 완료!")
                    st.markdown(response.text)
                    
                except Exception as e:
                    st.error(f"분석 중 오류가 발생했습니다. API 키가 올바른지 확인해주세요. (오류 내용: {e})")

else:
    st.info("나사 이미지를 업로드하면 분석 준비가 완료됩니다.")