import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import os
from twilio.rest import Client

account_sid = os.getenv('ACbd6cc0dd4dcaaceb2ad8080a4a26d1e0')
auth_token = os.getenv('59d5391bcd3ffaad2f7f4284600f1d5d')
client = Client(account_sid, auth_token)

token = client.tokens.create()

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

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

class SignLanguageTransformer(VideoTransformerBase):
    def __init__(self):
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.interpreter = tf.lite.Interpreter(model_path="model-withflip.tflite")
        self.interpreter.allocate_tensors()
        self.prediction_fn = self.interpreter.get_signature_runner("serving_default")
        self.sequence_data = []
        self.message = ""
        self.messages = []  # List to store messages
        self.font_path = "TH Krub.ttf"  # Update the path font

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image, results = mediapipe_detection(image, self.holistic, True, {
            'face': (255, 255, 255),
            'pose': (255, 255, 255),
            'left_hand': (255, 255, 255),
            'right_hand': (255, 255, 255),
            'font': (255, 255, 255)
        })
        landmarks = extract_coordinates(results)
        self.sequence_data.append(landmarks)

        if len(self.sequence_data) % 30 == 0:
            prediction = self.prediction_fn(inputs=np.array(self.sequence_data, dtype=np.float32))
            sign = np.argmax(prediction["outputs"])
            
            self.message = ORD2SIGN[sign]
            self.messages.append(self.message)
            if len(self.messages) > 5:
                self.messages.pop(0)  # Keep only the last 5 messages
            self.sequence_data = []

        if self.message:
            image = draw_text_with_font(image, self.message, (10, 30), self.font_path, 32, (255, 255, 255))

        return image

def intro():
    st.write(
        """
        Welcome to the Thai Sign Language (TSL) Detection Application :wave:

        How to Use This Application :eyes:
        
        1. Detector Mode 📝
        - เลือก **"Detector"** จากแถบด้านข้าง
        - สามารถเลือก configure detection และ tracking confidence, landmark drawing, และตั้งค่าสีโดยการเปิด **"Show Options"**
        - อัปโหลดวิดีโอไฟล์ เช่น MP4, MOV, AVI, or MKV
        - ดูภาษามือที่แสดงออกมาพร้อมวิดีโอ
        2. Live Detector :star:  (Local Host only)
        - เลือก **"Live Detector"** จากแถบด้านข้าง
        - สามารถเลือก landmark drawing, และตั้งค่าสีโดยการเปิด **"Show Options"**
        - กด **"Start TSL Detection"** เพื่อเริ่ม real-time detection โดยใช้กล้อง
        - ดูภาษามือที่แสดงออกมาในรูปแบบเรียลไทม์
        - กด **"Stop TSL Detection"** เพื่อปิด

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
    st.write('---')

    with st.sidebar.form(key="Detector"):
        st.title('Detector Mode 📝')
        st.write('---')

        detector_mode = st.selectbox(
            "Detector Mode", ["Upload Detector"]
        )

        with st.expander("Show Options"):
            detection_confidence = st.slider(
                "Detection Confidence", min_value=0.0, max_value=1.0, value=0.5
            )
            tracking_confidence = st.slider(
                "Tracking Confidence", min_value=0.0, max_value=1.0, value=0.5
            )
            draw_landmarks = st.checkbox("Draw Landmarks", value=True)
            st.write('---')

            face_color = st.color_picker('Face Landmark Color', '#ffffff')
            pose_color = st.color_picker('Pose Landmark Color', '#ffffff')
            left_hand_color = st.color_picker('Left Hand Landmark Color', '#ffffff')
            right_hand_color = st.color_picker('Right Hand Landmark Color', '#ffffff')
            font_color = st.color_picker('Font Color', '#ffffff')
            st.write('---')

        colors = {
            'face': hex_to_rgb(face_color),
            'pose': hex_to_rgb(pose_color),
            'left_hand': hex_to_rgb(left_hand_color),
            'right_hand': hex_to_rgb(right_hand_color),
            'font': hex_to_rgb(font_color),
        }

        submitted = st.form_submit_button("Apply Changes")

    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])

    if video_file is not None:
        st.video(video_file)
        with mp_holistic.Holistic(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        ) as holistic:
            cap = cv2.VideoCapture(video_file.name)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image, results = mediapipe_detection(frame, holistic, draw_landmarks, colors)
                st.image(image)

            cap.release()
            cv2.destroyAllWindows()

def live_tsl():
    st.write('## Real-time Thai Sign Language Detection')
    st.write('---')
    webrtc_streamer(key="example", video_transformer_factory=SignLanguageTransformer , rtc_configuration={  # Add this line
        "iceServers": token.ice_servers})

PAGES = {
    "Introduction": intro,
    "TSL Detection": tsl,
    "Live TSL Detection": live_tsl,
}

st.sidebar.title('Thai Sign Language (TSL) Detection Application')
selection = st.sidebar.selectbox("Choose a mode", list(PAGES.keys()))
page = PAGES[selection]
page()
