import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace

st.set_page_config(
    page_title="Emotion Detector App",
    page_icon="🎭",
    layout="centered"
)

st.title("🎭 Facial Emotion Recognition")
st.markdown("""
Welcome to the Emotion Detector! Upload an image or snap a photo using your webcam. 
The AI will detect your face and predict your dominant emotion (Happy, Sad, Angry, Neutral, Surprise, Fear, or Disgust).
""")
st.divider()

def process_and_analyze_image(image_bytes):
    np_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    try:
        results = DeepFace.analyze(
            img_bgr,
            actions=['emotion'],
            enforce_detection=True
        )

        if not isinstance(results, list):
            results = [results]

        for face in results:
            region = face['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']

            dominant_emotion = face['dominant_emotion']
            confidence = face['emotion'][dominant_emotion]

            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 3)

            label = f"{dominant_emotion.capitalize()} ({confidence:.1f}%)"
            cv2.rectangle(img_bgr, (x, y - 40), (x + w, y), (0, 255, 0), -1)
            cv2.putText(
                img_bgr,
                label,
                (x + 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2
            )

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb, results

    except ValueError:
        return None, None

input_method = st.radio(
    "Choose Input Method:",
    ("Upload Image", "Use Webcam"),
    horizontal=True
)

image_data = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload a clear photo of a face",
        type=['jpg', 'jpeg', 'png']
    )
    if uploaded_file is not None:
        image_data = uploaded_file.read()

elif input_method == "Use Webcam":
    camera_file = st.camera_input("Take a picture")
    if camera_file is not None:
        image_data = camera_file.read()

if image_data:
    with st.spinner("Analyzing face..."):
        annotated_img, analysis_results = process_and_analyze_image(image_data)

        if annotated_img is None:
            st.error(
                "⚠️ No face detected. Please try another image with a clearer view of the face."
            )
        else:
            st.image(
                annotated_img,
                caption="Processed Image",
                use_column_width=True
            )

            st.subheader("Results")

            for idx, face in enumerate(analysis_results):
                dominant_emotion = face['dominant_emotion']
                confidence = face['emotion'][dominant_emotion]

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label=f"Face {idx + 1} Emotion",
                        value=dominant_emotion.capitalize()
                    )
                with col2:
                    st.metric(
                        label="Confidence Score",
                        value=f"{confidence:.2f}%"
                    )
                st.divider()