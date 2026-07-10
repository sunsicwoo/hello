import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# 웹 브라우저 탭 설정
st.set_page_config(page_title="AI 결함 탐지 서비스", layout="wide")

st.title("🛠️ YOLOv8 기반 부품 결함 탐지 시스템")
st.markdown("GitHub에 업로드된 인공지능 모델을 활용해 실시간으로 불량 영역을 탐지합니다.")
st.markdown("---")

# [핵심] 스트림릿 서버 환경에서는 파일이 같은 최상위 폴더에 있으므로 이름만 적어줍니다.
MODEL_PATH = "best.pt" 

# @st.cache_resource를 사용해야 사용자가 접속할 때마다 모델을 새로 로드하지 않아 서버가 안 멈춥니다.
@st.cache_resource
def load_yolo_model():
    try:
        return YOLO(MODEL_PATH)
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None

model = load_yolo_model()

if model is not None:
    # 화면을 좌우 2개 칸으로 나눔
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. 이미지 업로드")
        uploaded_file = st.file_uploader("검사할 이미지를 업로드하세요 (.png, .jpg)", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="업로드된 원본 이미지", use_container_width=True)

    with col2:
        st.subheader("2. AI 결함 분석 결과")
        if uploaded_file is not None:
            with st.spinner("서버에서 인공지능 모델 분석 중..."):
                # 이미지 변환 및 예측
                img_array = np.array(image)
                results = model.predict(img_array, conf=0.25)
                
                # 결과 시각화 및 RGB 채널 변환
                res_plotted = results[0].plot()
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                st.image(res_rgb, caption="결함 탐지 완료!", use_container_width=True)
                
                # 결과 피드백
                boxes = results[0].boxes
                if len(boxes) > 0:
                    st.error(f"🚨 총 {len(boxes)}개의 결함이 발견되었습니다. 조치가 필요합니다.")
                else:
                    st.success("✅ 결함이 발견되지 않은 정상 제품입니다.")
        else:
            st.info("왼쪽에 이미지를 업로드하면 AI 분석 결과가 여기에 표시됩니다.")