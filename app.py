import streamlit as st
# import mediapipe as mp
# import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import tensorflow as tf
import pandas as pd
import json
import tempfile
import os
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode

# Mediapipe and drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Constants
ROWS_PER_FRAME = 543

# Load sign language mapping
train = pd.read_csv('train.csv')
train['sign_ord'] = train['sign'].astype('category').cat.codes
SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.sequence_data = []
        self.last_prediction = None
        self.last_prediction_time = 0
        self.prediction_display_duration = 3  # seconds
        self.font_path = "./angsana.ttc"
        self.font_size = 32
        self.font = ImageFont.truetype(self.font_path, self.font_size)
        self.interpreter = tf.lite.Interpreter(model_path="model-traintestflip.tflite")
        self.interpreter.allocate_tensors()
        self.prediction_fn = self.interpreter.get_signature_runner("serving_default")
        self.messages = []

    def mediapipe_detection(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def draw_text_on_image(self, image, text, position, font, color=(0, 255, 0)):
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)
        draw.text(position, text, font=font, fill=color)
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        return image_cv

    def extract_coordinates(self, results):
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros((468, 3))
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
        return np.concatenate([face, lh, pose, rh])

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image, results = self.mediapipe_detection(image)
        landmarks = self.extract_coordinates(results)
        self.sequence_data.append(landmarks)

        if len(self.sequence_data) % 30 == 0:
            prediction = self.prediction_fn(inputs=np.array(self.sequence_data, dtype=np.float32))
            sign = np.argmax(prediction["outputs"])
            message = ORD2SIGN[sign]
            self.messages.append(message)
            if len(self.messages) > 5:
                self.messages.pop(0)
            self.last_prediction = message
            self.last_prediction_time = time.time()
            self.sequence_data = []

        current_time = time.time()
        if self.last_prediction and (current_time - self.last_prediction_time < self.prediction_display_duration):
            image = self.draw_text_on_image(image, self.last_prediction, (3, 30), self.font, (0, 0, 0))

        return image

def main():
    st.header("Thai Sign Language Detection")

    page_names_to_funcs = {
        "—": intro,
        "Detector": tsl,
        "Live Detector": live_detector,
    }
    app_mode = st.sidebar.selectbox("Choose the app mode", page_names_to_funcs.keys())
    st.subheader(app_mode)

    if app_mode == "Live Detector":
        live_detector()
    else:
        page_func = page_names_to_funcs[app_mode]
        page_func()

def intro():
    st.write("Welcome to the Thai Sign Language Detector!")

def tsl():
    st.write("This app allows you to upload a video file for Thai Sign Language detection.")
    interpreter = tf.lite.Interpreter(model_path="model-withflip.tflite")
    interpreter.allocate_tensors()
    prediction_fn = interpreter.get_signature_runner("serving_default")

    option = st.radio("Choose input method:", ("Upload a video file", "Record from webcam"))

    if option == "Upload a video file":
        video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            process_video(tfile.name, interpreter, prediction_fn)
    else:
        if st.button("Start Recording", key="start_recording_button"):
            video_path = record_video()
            process_video(video_path, interpreter, prediction_fn)
            os.remove(video_path)

def live_detector():
    webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, video_transformer_factory=VideoTransformer)

if __name__ == "__main__":
    main()
