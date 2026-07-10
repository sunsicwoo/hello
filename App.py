import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# 웹 브라우저 탭 설정
st.set_page_config(page_title="AI 너트 결함 탐지 서비스", layout="wide")

st.title("🛠️ 너트 검사 시스템 (콤팩트 다중 검사)")
st.markdown("GitHub에 업로드된 인공지능 모델을 활용해 여러 개의 너트 이미지를 한 번에 검사합니다.")
st.markdown("---")

# 모델 파일 경로
MODEL_PATH = "best.pt" 

@st.cache_resource
def load_yolo_model():
    try:
        return YOLO(MODEL_PATH)
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None

model = load_yolo_model()

if model is not None:
    # 1. 여러 장의 파일 업로드 허용
    uploaded_files = st.file_uploader(
        "검사할 너트 이미지들을 모두 선택하거나 드래그하세요 (.png, .jpg)", 
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.subheader(f"📊 총 {len(uploaded_files)}장의 너트 이미지 분석 결과")
        
        for idx, uploaded_file in enumerate(uploaded_files):
            st.markdown(f"### 🔍 {idx+1}. 파일명: `{uploaded_file.name}`")
            
            # AI 분석 수행 (화면 출력 전에 먼저 계산)
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            with st.spinner(f"'{uploaded_file.name}' 분석 중..."):
                results = model.predict(img_array, conf=0.25, verbose=False)
                res_plotted = results[0].plot()
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                boxes = results[0].boxes

            # [핵심] 레이아웃을 3개로 쪼개어 가운데(col2)에만 이미지를 배치 (좌우 여백을 주어 크기 축소 효과)
            # 비율을 1.5 : 2.0 : 1.5 로 설정하여 가운데 이미지가 약 30~40% 크기로 나오게 조절합니다.
            col1, col2, col3 = st.columns([1.5, 2.0, 1.5])
            
            with col2:
                # 1. 상단: 원본 이미지 (가로 폭에 맞추되 고정폭 240~300px 수준으로 콤팩트하게 출력)
                st.write("**📍 [상단] 원본 너트 이미지**")
                st.image(image, use_container_width=True)
                
                st.markdown("<br>", unsafe_allow_html=True) # 약간의 상하 공백
                
                # 2. 하단: 검출 이미지
                st.write("**🎯 [하단] AI 결함 탐지 결과**")
                st.image(res_rgb, use_container_width=True)
                
                # 3. 최하단: 결과 피드백 문구
                if len(boxes) > 0:
                    st.error(f"🚨 이 너트에서 총 {len(boxes)}개의 결함이 발견되었습니다.")
                else:
                    st.success("✅ 결함이 발견되지 않은 정상 너트 제품입니다.")
                        
            st.markdown("---") # 이미지 묶음 간 구분을 위한 하단 구분선
            
    else:
        st.info("💡 검사할 너트 이미지들을 업로드하면 여기에 상/하 정렬된 분석 결과가 표시됩니다.")