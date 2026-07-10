ㅊimport os
import streamlit as st
from google import genai
from PIL import Image

# 1. Gemini 클라이언트 초기화
# 환경 변수에 GEMINI_API_KEY가 등록되어 있으면 자동으로 인식합니다.
# 만약 코드가 클라우드(Streamlit Cloud)에서 돈다면 Settings -> Secrets에 등록하세요.
try:
    client = genai.Client()
except Exception as e:
    st.error("Gemini API 키를 설정해주세요. 환경변수 'GEMINI_API_KEY'가 필요합니다.")
    st.stop()

# 2. 스트림릿 UI 구성
st.set_page_config(page_title="나사 결함 탐지 AI", layout="centered")
st.title("🔩 나사 결함 탐지 시스템 (Gemini AI)")
st.write("나사 사진을 업로드하면 Gemini AI가 결함(마모, 부러짐, 나사산 손상 등)을 분석합니다.")

# 3. 파일 업로드 컴포넌트
uploaded_file = st.file_uploader("나사 이미지를 업로드하세요...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 업로드된 이미지를 화면에 표시
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드된 나사 이미지", use_container_width=True)
    
    st.write("---")
    st.subheader("🔍 AI 분석 결과")
    
    # 분석 중 애니메이션 효과
    with st.spinner("Gemini가 이미지를 분석하고 있습니다..."):
        try:
            # 4. 프롬프트 작성 (AI에게 역할을 부여하고 명확한 지시 전달)
            prompt = """
            당신은 정밀 제조 부품 검사 전문가입니다. 
            제공된 나사(Screw/Bolt) 이미지를 정밀하게 분석하여 다음 양식에 맞춰 한국어로 답변해주세요:
            
            1. 결함 여부: [정상 / 결함 발견]
            2. 결함 유형: (결함이 있다면 마모, 부식, 나사산 뭉개짐, 휘어짐 등 구체적으로 적어주세요. 정상이면 '없음')
            3. 상세 분석 내용: (이미지에서 관찰되는 나사의 상태를 기술적으로 설명해주세요)
            4. 조치 권고 사항: (폐기, 재사용 가능, 정밀 재검사 등)
            """
            
            # 5. Gemini 멀티모달 모델 호출 (최신 gemini-2.5-flash 모델 사용)
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[image, prompt]
            )
            
            # 6. 결과 출력
            st.markdown(response.text)
            
        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {e}")

else:
    st.info("나사 이미지를 업로드하면 분석이 시작됩니다.")