import streamlit as st
from ultralytics import YOLO
from google import genai
from PIL import Image
import os

# 1. 스트림릿 페이지 설정 (와이드 모드)
st.set_page_config(page_title="나사 결함 탐지 AI 시스템", layout="wide", initial_sidebar_state="expanded")

# 세션 상태 초기화 (API 인증 여부 저장용)
if "api_authenticated" not in st.session_state:
    st.session_state.api_authenticated = False

# 커스텀 타이틀 스타일 적용
st.markdown("""
    <style>
    .main-title { font-size:32px; font-weight:bold; color:#1E3A8A; margin-bottom:5px; }
    .sub-title { font-size:16px; color:#6B7280; margin-bottom:25px; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-title">🔩 나사 결함 정밀 검사 시스템</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">YOLOv8 Object Detection과 Gemini Vision AI를 결합한 가공 부품 품질 검사 솔루션</div>', unsafe_allow_html=True)

# 2. 사이드바 구성 (API Key 입력 및 [인증] 버튼 추가)
st.sidebar.header("⚙️ 시스템 설정")
user_api_key = st.sidebar.text_input(
    "Google Gemini API Key", 
    type="password", 
    placeholder="AIzaSy..."
)

# [추가 기능] API 키 인증 버튼 및 로직
if st.sidebar.button("인증", use_container_width=True):
    if user_api_key:
        with st.sidebar.spinner("API 키 검증 중..."):
            try:
                # 실제로 작동하는 가벼운 API 호출을 시도하여 키를 검증합니다.
                test_client = genai.Client(api_key=user_api_key)
                # 모델 리스트를 가볍게 호출하여 인증이 유효한지 확인
                test_client.models.get(model='gemini-2.5-flash')
                
                st.session_state.api_authenticated = True
                st.session_state.verified_key = user_api_key # 인증 성공한 키 저장
            except Exception as e:
                st.session_state.api_authenticated = False
                st.sidebar.error("❌ 잘못된 API 키이거나 통신 오류입니다.")
    else:
        st.sidebar.warning("⚠️ API 키를 입력해주세요.")

# 인증 결과 알림 표시
if st.session_state.api_authenticated:
    st.sidebar.success("✅ Gemini API 인증 성공!")
else:
    st.sidebar.info("🔓 API 키를 입력하고 인증 버튼을 눌러주세요.")

st.sidebar.markdown("[🔑 API 키 발급받기](https://aistudio.google.com/)")
st.sidebar.write("---")

# Confidence Threshold 슬라이더
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.00, max_value=1.00, value=0.25, step=0.05
)

# 3. YOLO 모델 로드
model_path = 'best.pt' 
yolo_model = None
if os.path.exists(model_path):
    try:
        yolo_model = YOLO(model_path)
    except Exception as e:
        st.sidebar.error(f"YOLO 로드 실패: {e}")
else:
    st.sidebar.error(f"'{model_path}' 파일을 찾을 수 없습니다.")

# 4. 메인 화면 - 파일 업로드
uploaded_file = st.file_uploader("검사할 나사 이미지를 드래그앤드롭하거나 선택하세요.", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    
    # 5. API 및 모델 검증 후 대시보드 작동
    if not st.session_state.api_authenticated:
        st.info("💡 좌측 사이드바에서 Gemini API 키를 입력하고 [인증] 버튼을 눌러야 정밀 리포트 기능이 활성화됩니다.")
    elif not yolo_model:
        st.error("YOLO 모델이 준비되지 않았습니다.")
    else:
        # 탐지 실행 버튼
        if st.button("⚡ 탐지 및 정밀 분석 실행", use_container_width=True):
            col1, col2 = st.columns(2)
            
            with st.spinner("AI 분석 엔진 가동 중..."):
                try:
                    # 5-1. YOLOv8 객체 탐지
                    results = yolo_model(original_image, conf=conf_threshold)
                    detected_img_array = results[0].plot()
                    detected_image = Image.fromarray(detected_img_array[..., ::-1])

                    # 결과 텍스트 요약
                    detected_info = ""
                    if len(results[0].boxes) > 0:
                        detected_info = "⚠️ 결함 발견:\n"
                        for box in results[0].boxes:
                            class_id = int(box.cls[0])
                            class_name = yolo_model.names[class_id]
                            confidence = float(box.conf[0])
                            detected_info += f"- **{class_name}** (신뢰도: {confidence*100:.1f}%)\n"
                    else:
                        detected_info = "✅ 특이사항 없음: 정상 제품으로 판단됩니다."

                    # 화면 표시 (1단계: 이미지 비교)
                    with col1:
                        st.subheader("📸 원본 이미지")
                        st.image(original_image, use_container_width=True)
                        
                    with col2:
                        st.subheader("🔍 YOLO 결함 탐지 결과")
                        st.image(detected_image, use_container_width=True)
                        st.markdown(detected_info)

                    st.write("---")

                    # 5-2. 2단계: Gemini 상세 분석 (인증 완료된 키 사용)
                    st.subheader("📝 Gemini AI 품질 판정 리포트")
                    
                    client = genai.Client(api_key=st.session_state.verified_key)
                    
                    prompt = f"""
                    당신은 제조 부품 품질 관리 전문가입니다. 
                    제공된 나사 이미지와 YOLOv8 탐지 정보({detected_info.strip()})를 바탕으로 다음 3가지 항목만 한국어로 간단명료하게 작성해주세요. 불필요한 서론은 생략하세요.

                    1. **결함 부위 상태**: (YOLO 결과 검증 및 육안상 보이는 특징 요약)
                    2. **예상 발생 원인**: (마모, 부식 등의 가공/사용상 원인 한 줄 추론)
                    3. **현장 조치 가이드**: (재사용 가능, 즉시 폐기, 재검사 등 명확한 행동 지침)
                    """
                    
                    response = client.models.generate_content(
                        model='gemini-3.1-flash-lite',
                        contents=[original_image, detected_image, prompt]
                    )
                    
                    with st.container(border=True):
                        st.markdown(response.text)
                    
                    st.toast("품질 분석이 성공적으로 완료되었습니다!", icon="🔥")

                except Exception as e:
                    st.error(f"분석 중 오류 발생: {e}")
else:
    st.image("https://images.unsplash.com/photo-1534224039826-c7a0dea0e66a?q=80&w=1000", caption="제조 품질 관리 시스템 데모", use_container_width=True)
