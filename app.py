import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import json
import tempfile
from PIL import Image, ImageDraw, ImageFont


try:
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
except ImportError:
    st.error("Failed to import MediaPipe.")


ROWS_PER_FRAME = 543  # number of landmarks per frame

def mediapipe_detection(image, model, draw_landmarks, colors):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # color conversion
    image.flags.writeable = False  # img no longer writeable
    pred = model.process(image)  # make landmark prediction
    image.flags.writeable = True  # img now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # color reconversion

    if draw_landmarks:
        if pred.face_landmarks:
            mp_drawing.draw_landmarks(image, pred.face_landmarks, mp_holistic.FACEMESH_TESSELATION, landmark_drawing_spec=mp_drawing.DrawingSpec(color=colors['face'], thickness=1, circle_radius=1), connection_drawing_spec=mp_drawing.DrawingSpec(color=colors['face'], thickness=1))
        if pred.pose_landmarks:
            mp_drawing.draw_landmarks(image, pred.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=colors['pose'], thickness=2, circle_radius=2), connection_drawing_spec=mp_drawing.DrawingSpec(color=colors['pose'], thickness=2))
        if pred.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, pred.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=colors['left_hand'], thickness=1, circle_radius=1), connection_drawing_spec=mp_drawing.DrawingSpec(color=colors['left_hand'], thickness=1))
        if pred.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, pred.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=colors['right_hand'], thickness=1, circle_radius=1), connection_drawing_spec=mp_drawing.DrawingSpec(color=colors['right_hand'], thickness=1))

    return image, pred

def extract_coordinates(results):
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros((468, 3))
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    return np.concatenate([face, lh, pose, rh])

def load_json_file(json_path):
    with open(json_path, 'r') as f:
        sign_map = json.load(f)
    return sign_map

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

train = pd.read_csv('train.csv')

# Add ordinally Encoded Sign (assign number to each sign name)
train['sign_ord'] = train['sign'].astype('category').cat.codes

# Dictionaries to translate sign <-> ordinal encoded sign
SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

def draw_text_with_font(image, text, position, font_path, font_size, color):
    """Draw text on an image using a specific font."""
    # Convert the image to PIL format
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def process_video(video_path, interpreter, prediction_fn, detection_confidence, tracking_confidence, draw_landmarks, show_predictions_in_video, colors):
    sequence_data = []
    cap = cv2.VideoCapture(video_path)
    video_placeholder = st.empty()

    font_path = "TH Krub.ttf"  # Update the path font

    with mp_holistic.Holistic(min_detection_confidence=detection_confidence, min_tracking_confidence=tracking_confidence) as holistic:
        message = ""  # Initialize message variable
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic, draw_landmarks, colors)
            landmarks = extract_coordinates(results)
            sequence_data.append(landmarks)
            
            if show_predictions_in_video and sequence_data:
                prediction = prediction_fn(inputs=np.array([sequence_data[-1]], dtype=np.float32))
                sign = np.argmax(prediction["outputs"])
                message = ORD2SIGN[sign]
                image = draw_text_with_font(image, message, (10, 30), font_path, 32, colors['font'])

            # Display the frame with landmarks using Streamlit
            video_placeholder.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")
        
        cap.release()
    
    if sequence_data and not show_predictions_in_video:
        sequence_data = np.array(sequence_data, dtype=np.float32)
        prediction = prediction_fn(inputs=sequence_data)
        sign = np.argmax(prediction["outputs"])
        st.info(f"Predicted Sign: {ORD2SIGN[sign]}")

def intro():
    st.write(
        """
        Welcome to the Thai Sign Language (TSL) Detection Application :wave:
        
        สามารถดูคำที่ Predict ได้ที่ https://github.com/meprxxw/Thai-Sign-Language/blob/main/actions
        และ ตัวอย่างภาษามือได้ที่ https://www.th-sl.com/?openExternalBrowser=1.
        
        How to Use This Application :eyes:
        
        Detector Mode 📝
        - เลือก **"Detector"** จากแถบด้านข้าง
        - สามารถเลือก configure detection และ tracking confidence, landmark drawing, และตั้งค่าสีโดยการเปิด **"Show Options"**
        - อัปโหลดวิดีโอไฟล์ เช่น MP4, MOV, AVI, or MKV
        - ดูภาษามือที่แสดงออกมาพร้อมวิดีโอ

        Features
        - **Detection Confidence:** ปรับระดับความมั่นใจก่อนการทำนายภาษามือ
        - **Tracking Confidence:** ปรับระดับความมั่นใจในการตรวจจับจุดที่เชื่อมกับส่วนของร่างกาย
        - **Landmark Drawing:** เปิดการแสดงผลการตรวจจับจุดที่เชื่อมกับส่วนของร่างกาย
        - **Color Settings:** ปรับสีที่แสดงผลของใบหน้า, ท่าทาง, มือซ้าย, มือขวา และตัวอักษรที่แสดงออกมา
        
        """
    )

def hex_to_rgb(hex_color):
    """Converts hexadecimal color code to (B, G, R) format."""
    hex_color = hex_color.lstrip('#')  # Remove '#' from the beginning
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return rgb

def tsl():
    st.write(
        """
        This app allows you to upload a video file for Thai Sign Language detection.
        """
    )
    interpreter = tf.lite.Interpreter(model_path="model-withflip.tflite")
    interpreter.allocate_tensors()
    prediction_fn = interpreter.get_signature_runner("serving_default")

    detection_confidence = st.slider("Detection Confidence", 0.0, 1.0, 0.5)
    tracking_confidence = st.slider("Tracking Confidence", 0.0, 1.0, 0.5)

    # Toggle button for showing/hiding options
    show_options = st.checkbox("Show Options", value=False)

    if show_options:
        draw_landmarks = st.checkbox("Draw Landmarks", value=True)
        show_predictions_in_video = st.checkbox("Show Predictions in Video", value=True)

        # Color pickers for each element
        face_color = st.color_picker("Face Color", value="#FFFFFF")
        pose_color = st.color_picker("Pose Color", value="#FFFFFF")
        left_hand_color = st.color_picker("Left Hand Color", value="#FFFFFF")
        right_hand_color = st.color_picker("Right Hand Color", value="#FFFFFF")
        font_color = st.color_picker("Font Color", value="#FFFFFF")

        # Convert hexadecimal colors to (B, G, R) format
        face_color_rgb = hex_to_rgb(face_color)
        pose_color_rgb = hex_to_rgb(pose_color)
        left_hand_color_rgb = hex_to_rgb(left_hand_color)
        right_hand_color_rgb = hex_to_rgb(right_hand_color)
        font_color_rgb = hex_to_rgb(font_color)

        colors = {
            'face': face_color_rgb,
            'pose': pose_color_rgb,
            'left_hand': left_hand_color_rgb,
            'right_hand': right_hand_color_rgb,
            'font': font_color_rgb
        }
    else:
        # Default values if options are not shown
        draw_landmarks = True
        show_predictions_in_video = True
        colors = {
            'face': (255, 255, 255),
            'pose': (255, 255, 255),
            'left_hand': (255, 255, 255),
            'right_hand': (255, 255, 255),
            'font': (255, 255, 255)
        }

    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        process_video(tfile.name, interpreter, prediction_fn, detection_confidence, tracking_confidence, draw_landmarks, show_predictions_in_video, colors)

def main():
    st.header("Thai Sign Language Detection")

    page_names_to_funcs = {
        "—": intro,
        "Detector 🕵️‍♀️": tsl,
    }   
    app_mode = st.sidebar.selectbox("Choose the app mode", page_names_to_funcs.keys())

    st.subheader(app_mode)

    if app_mode == "Detector 🕵️‍♀️":
        tsl()
    else:
        page_func = page_names_to_funcs[app_mode]
        page_func()

if __name__ == "__main__":
    main()
