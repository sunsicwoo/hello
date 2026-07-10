import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# 웹 브라우저 탭 설정
st.set_page_config(page_title="AI 너트 결함 탐지 서비스", layout="wide")

st.title("🛠️ 너트 검사 시스템 (다중 이미지 검사)")
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
    # 1. 여러 장의 파일 업로드 허용 ([수정] accept_multiple_files=True 추가)
    uploaded_files = st.file_uploader(
        "검사할 너트 이미지들을 모두 선택하거나 드래그하세요 (.png, .jpg)", 
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )
    
    # 파일이 하나라도 업로드 되었을 때
    if uploaded_files:
        st.subheader(f"📊 총 {len(uploaded_files)}장의 너트 이미지 분석 결과")
        
        # 각 이미지별로 처리를 반복하며 화면에 출력
        for idx, uploaded_file in enumerate(uploaded_files):
            # 시각적인 구분을 위한 구분선과 서브타이틀
            st.markdown(f"### 🔍 {idx+1}. 파일명: `{uploaded_file.name}`")
            
            # 좌우 분할 레이아웃 생성
            col1, col2 = st.columns(2)
            
            with col1:
                # 원본 이미지 표시
                image = Image.open(uploaded_file)
                st.image(image, caption="원본 너트 이미지", use_container_width=True)
                
            with col2:
                # AI 분석 및 결과 표시
                with st.spinner(f"'{uploaded_file.name}' 분석 중..."):
                    img_array = np.array(image)
                    results = model.predict(img_array, conf=0.25, verbose=False) # verbose=False로 터미널 로그 단축
                    
                    # 결과 시각화 및 RGB 변환
                    res_plotted = results[0].plot()
                    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                    
                    st.image(res_rgb, caption="결함 탐지 완료!", use_container_width=True)
                    
                    # 결과 피드백 문구
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        st.error(f"🚨 이 너트에서 총 {len(boxes)}개의 결함이 발견되었습니다.")
                    else:
                        st.success("✅ 결함이 발견되지 않은 정상 너트 제품입니다.")
                        
            st.markdown("---") # 이미지 간 구분을 위한 하단 구분선
            
    else:
        st.info("💡 검사할 너트 이미지들을 업로드하면 여기에 다중 분석 결과가 순서대로 표시됩니다.")