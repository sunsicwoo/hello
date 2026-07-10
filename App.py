import streamlit as st
from ultralytics import YOLO
from google import genai
from PIL import Image
import os
import io

# 1. 스트림릿 페이지 설정
st.set_page_config(page_title="나사 결함 탐지 AI (YOLO + Gemini)", layout="centered")
st.title("🔩 나사 결함 탐지 시스템 (YOLO + Gemini)")
st.write("나사 사진을 업로드하면 YOLO 모델이 결함을 탐지하고, Gemini AI가 상세한 분석을 제공합니다.")

# 2. 사이드바에 API 키 입력창 생성 (보안 유지)
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

# 3. YOLO 모델 로드 (GitHub에 업로드된 best.pt 파일 사용)
# 파일 경로를 적절히 수정해 주세요. 예: 'models/best.pt'
model_path = 'best.pt' 
yolo_model = None
if os.path.exists(model_path):
    try:
        yolo_model = YOLO(model_path)
    except Exception as e:
        st.error(f"YOLO 모델을 로드하는 중 오류가 발생했습니다: {e}")
else:
    st.error(f"'{model_path}' 파일을 찾을 수 없습니다. GitHub에 업로드했는지 확인해주세요.")


# 4. 파일 업로드 컴포넌트
uploaded_file = st.file_uploader("나사 이미지를 업로드하세요...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 업로드된 원본 이미지를 PIL 이미지로 로드
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="업로드된 나사 이미지", use_container_width=True)
    
    st.write("---")
    
    # 5. API 키와 YOLO 모델이 모두 준비되었는지 확인
    if not user_api_key:
        st.error("보안 및 상세 분석을 위해 왼쪽 사이드바에 Gemini API 키를 먼저 입력해야 합니다.")
    elif not yolo_model:
        st.error("YOLO 모델이 준비되지 않아 분석을 시작할 수 없습니다.")
    else:
        # 분석 실행 버튼
        if st.button("나사 결함 정밀 분석 시작"):
            with st.spinner("YOLO 모델이 결함을 탐지하고, Gemini가 상세 분석을 작성하고 있습니다..."):
                try:
                    # 5-1. **YOLOv8 모델로 객체 탐지 실행**
                    results = yolo_model(original_image)
                    
                    # 5-2. 탐지 결과가 반영된 이미지 가져오기 (박스가 쳐진 이미지)
                    # YOLOv8의 plot() 메서드는 numpy array를 반환하므로, 이를 PIL 이미지로 변환해야 합니다.
                    detected_img_array = results[0].plot()
                    detected_image = Image.fromarray(detected_img_array[..., ::-1]) # RGB로 변환

                    # 5-3. 탐지 결과 정보를 텍스트로 정리 (Gemini에게 전달할 힌트)
                    detected_info = ""
                    if len(results[0].boxes) > 0:
                        detected_info = "YOLO 모델이 다음 결함들을 탐지했습니다:\n"
                        for box in results[0].boxes:
                            class_id = int(box.cls[0])
                            class_name = yolo_model.names[class_id]
                            confidence = float(box.conf[0])
                            detected_info += f"- 결함 유형: {class_name}, 신뢰도: {confidence:.2f}\n"
                    else:
                        detected_info = "YOLO 모델이 결함을 탐지하지 못했습니다. 나사가 정상일 가능성이 높습니다."

                    # **화면에 YOLO 탐지 결과 이미지 표시**
                    st.subheader("🔍 1단계: YOLO 결함 탐지 결과 (Bounding Box)")
                    st.image(detected_image, caption="결함 부위가 박스로 표시된 이미지", use_container_width=True)
                    st.write(detected_info)

                    st.write("---")

                    # 5-4. **Gemini API 호출하여 상세 분석 요청**
                    
                    # Gemini 클라이언트 초기화
                    client = genai.Client(api_key=user_api_key)
                    
                    # 프롬프트 작성 (YOLO 탐지 정보를 힌트로 제공)
                    prompt = f"""
                    당신은 정밀 제조 부품 검사 전문가입니다. 
                    제공된 나사(Screw/Bolt) 이미지와 YOLOv8 모델의 탐지 결과를 바탕으로 정밀한 분석을 제공해주세요.
                    
                    **YOLOv8 탐지 정보:**
                    {detected_info}
                    
                    **분석 요청사항:**
                    위 정보를 참고하여, 제공된 나사 이미지의 상태를 다음 양식에 맞춰 한국어로 상세히 기술해주세요:
                    
                    1. 결함 정밀 진단: (YOLO가 탐지한 결함이 실제 이미지에서 어떻게 보이는지 설명하고, 오탐지 가능성이나 추가적인 결함이 있다면 언급해주세요.)
                    2. 발생 원인 추론: (마모, 부식, 나사산 손상 등이 왜 발생했을지 제조 공정이나 사용 환경 관점에서 추론해주세요.)
                    3. 품질 영향도 평가: (이 결함이 나사의 결합력이나 전체 시스템의 안전성에 어떤 영향을 미칠지 평가해주세요.)
                    4. 최종 조치 권고 사항: (폐기, 재사용 가능, 정밀 재검사, 공정 개선 등 구체적인 조치를 제안해주세요.)
                    """
                    
                    # Gemini 멀티모달 모델 호출 (최신 gemini-1.5-flash 모델 사용 - 3.5 모델 지원 확인 필요)
                    response = client.models.generate_content(
                        model='gemini-3.5-flash',
                        contents=[original_image, detected_image, prompt]
                    )
                    
                    # **결과 처리: Gemini 상세 분석 텍스트 표시**
                    st.subheader("📝 2단계: Gemini AI 상세 분석 리포트")
                    st.markdown(response.text)
                    st.success("YOLO + Gemini 정밀 분석 완료!")
                    
                except Exception as e:
                    st.error(f"분석 중 오류가 발생했습니다. API 키가 올바른지, 혹은 모델 경로가 정확한지 확인해주세요. (오류 내용: {e})")

else:
    st.info("나사 이미지를 업로드하면 분석 준비가 완료됩니다.")