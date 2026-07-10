import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# 웹 브라우저 탭 설정
st.set_page_config(page_title="AI 너트 결함 탐지 서비스", layout="wide")

st.title("🛠️ 너트 검사 시스템 (초소형 다중 검사)")
st.markdown("여러 개의 너트 이미지를 한눈에 볼 수 있도록 컴팩트하게 검사합니다.")
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
            # 파일명과 결과를 한 줄에 콤팩트하게 표시하기 위해 가로 레이아웃(2칼럼) 구성
            # 왼쪽(col_text)에는 텍스트 정보, 오른쪽(col_img)에는 초소형 이미지들을 배치합니다.
            col_text, col_img = st.columns([1, 1])
            
            # AI 분석 수행
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            with st.spinner(f"'{uploaded_file.name}' 분석 중..."):
                results = model.predict(img_array, conf=0.25, verbose=False)
                res_plotted = results[0].plot()
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                boxes = results[0].boxes

            # 1. 왼쪽 칸: 파일명과 AI 판정 결과 텍스트 안내
            with col_text:
                st.markdown(f"#### 🔍 {idx+1}. `{uploaded_file.name}`")
                if len(boxes) > 0:
                    st.error(f"🚨 결함 {len(boxes)}개 발견! 조치 필요.")
                else:
                    st.success("✅ 정상 너트 제품입니다.")
            
            # 2. 오른쪽 칸: 상단 원본 / 하단 결과를 초소형(width=160)으로 배치
            with col_img:
                # 내부에서 다시 상하로 배치 (width를 160으로 지정하여 기존의 20% 수준으로 축소)
                st.caption("📷 원본")
                st.image(image, width=160)
                
                st.caption("🎯 AI 결과")
                st.image(res_rgb, width=160)
                        
            st.markdown("---") # 이미지 묶음 간 구분을 위한 하단 구분선
            
    else:
        st.info("💡 검사할 너트 이미지들을 업로드하면 여기에 컴팩트한 분석 결과가 표시됩니다.")