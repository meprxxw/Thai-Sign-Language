import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import tensorflow as tf
import pandas as pd
import json
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

ROWS_PER_FRAME = 543  # number of landmarks per frame

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # color conversion
    image.flags.writeable = False  # img no longer writeable
    pred = model.process(image)  # make landmark prediction
    image.flags.writeable = True  # img now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # color reconversion
    return image, pred

def draw(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=3),
                              mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=1, circle_radius=0))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0, 150, 0), thickness=3, circle_radius=3),
                              mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(200, 56, 12), thickness=3, circle_radius=3),
                              mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(250, 56, 12), thickness=3, circle_radius=3),
                              mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2))

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

def draw_text_on_image(image, text, position, font, color=(0, 255, 0)):
    # Convert the OpenCV image (BGR) to a PIL image (RGB)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    draw.text(position, text, font=font, fill=color)
    # Convert the PIL image back to an OpenCV image
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    return image_cv

class CFG:
    data_dir = ""
    sequence_length = 12
    rows_per_frame = 543

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

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        model_path = "model-traintestflip.tflite"
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found at {model_path}")
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Process the image using MediaPipe
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            image, results = mediapipe_detection(img, holistic)
            landmarks = extract_coordinates(results)
        
        if len(landmarks) % 30 == 0:
            sequence_data = np.array([landmarks], dtype=np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], sequence_data)
            self.interpreter.invoke()
            prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
            sign = np.argmax(prediction)
            message = ORD2SIGN[sign]
            img = draw_text_on_image(img, message, (3, 30), ImageFont.truetype("./angsana.ttc", 32), (0, 0, 0))

        return img

def main():
    st.header("Thai Sign Language Detection")

    page_names_to_funcs = {
        "—": intro,
        "Detector": tsl,
        "Live Detector": live_detector,
    }   
    app_mode = st.sidebar.selectbox("Choose the app mode", page_names_to_funcs.keys())

    st.subheader(app_mode)

    if "run" not in st.session_state:
        st.session_state.run = False

    if app_mode == "Live Detector":
        if not st.session_state.run:
            if st.button("Start TSL Detection", key="start_detection_button"):
                st.session_state.run = True
                st.experimental_rerun()
        else:
            if st.button("Stop TSL Detection", key="stop_detection_button"):
                st.session_state.run = False
                st.experimental_rerun()

        if st.session_state.run:
            live_detector()
    else:
        page_func = page_names_to_funcs[app_mode]
        page_func()

def intro():
    st.write(
        """
       
    """
    )

def tsl():
    st.write(
        """
        This app allows you to upload a video
