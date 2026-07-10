import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# 웹 브라우저 탭 설정 (넓은 화면 모드)
st.set_page_config(page_title="AI 너트 결함 탐지 서비스", layout="wide")

st.title("🛠️ 너트 검사 시스템 (2열 모니터링)")
st.markdown("공간 효율을 극대화하여 2열 바둑판 형태로 너트 검사 결과를 한눈에 확인합니다.")
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
        
        # [핵심] 화면을 크게 왼쪽(col_left)과 오른쪽(col_right) 2열로 나눕니다.
        # 비율은 1:1로 정확히 반반씩 씁니다.
        grid_cols = st.columns(2)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            # idx가 짝수(0, 2, 4...)면 1열(왼쪽), 홀수(1, 3, 5...)면 2열(오른쪽)에 배치합니다.
            target_col = grid_cols[idx % 2]
            
            # AI 분석 수행 (공통 작업)
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            with st.spinner(f"'{uploaded_file.name}' 분석 중..."):
                results = model.predict(img_array, conf=0.25, verbose=False)
                res_plotted = results[0].plot()
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                boxes = results[0].boxes

            # 타겟 열(왼쪽 또는 오른쪽) 내부를 오밀조밀하게 채웁니다.
            with target_col:
                # 겉싸개 상자(st.container)를 씌워서 1장 단위로 깔끔하게 구역을 묶어줍니다.
                with st.container(border=True):
                    st.markdown(f"##### 🔍 {idx+1}. `{uploaded_file.name}`")
                    
                    # 카드 내부에서 이미지 2개를 나란히 가로로 배치 (상단 원본, 하단 결과 대신 좌 원본, 우 결과)
                    # 크기가 작기 때문에 가로 배치가 세로 스크롤을 줄이는 데 훨씬 유리합니다.
                    img_cols = st.columns(2)
                    with img_cols[0]:
                        st.caption("📷 원본 (20%)")
                        st.image(image, width=150)
                    with img_cols[1]:
                        st.caption("🎯 AI 결과 (20%)")
                        st.image(res_rgb, width=150)
                    
                    # 이미지 바로 아래에 결과 안내 텍스트 출력
                    if len(boxes) > 0:
                        st.error(f"🚨 결함 {len(boxes)}개 발견!")
                    else:
                        st.success("✅ 정상 제품")
                        
    else:
        st.info("💡 검사할 너트 이미지들을 업로드하면 2열 격자 형태로 결과가 표시됩니다.")