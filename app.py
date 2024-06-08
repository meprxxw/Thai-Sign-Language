import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import json
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av

# Initialize MediaPipe holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

ROWS_PER_FRAME = 543  # Number of landmarks per frame

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Color conversion
    image.flags.writeable = False  # Image no longer writeable
    results = model.process(image)  # Make landmark prediction
    image.flags.writeable = True  # Image now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Color reconversion
    return image, results

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

# Add ordinally encoded sign (assign number to each sign name)
train['sign_ord'] = train['sign'].astype('category').cat.codes

# Dictionaries to translate sign <-> ordinal encoded sign
SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

class VideoRecorder(VideoProcessorBase):
    def __init__(self):
        self.recording = False
        self.frames = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.recording:
            self.frames.append(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def start_recording(self):
        self.recording = True
        self.frames = []

    def stop_recording(self):
        self.recording = False
        video_path = 'recorded_video.mp4'
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (640, 480))
        for frame in self.frames:
            out.write(frame)
        out.release()
        return video_path

def process_video(video_path, interpreter, prediction_fn):
    sequence_data = []
    cap = cv2.VideoCapture(video_path)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            landmarks = extract_coordinates(results)
            sequence_data.append(landmarks)

        cap.release()
    
    if sequence_data:
        sequence_data = np.array(sequence_data, dtype=np.float32)
        prediction = prediction_fn(inputs=sequence_data)
        sign = np.argmax(prediction["outputs"])
        st.write(f"Predicted Sign: {ORD2SIGN[sign]}")

def intro():
    st.write(
        """
        Welcome to the Thai Sign Language Detection App.
    """
    )

def tsl():
    st.write(
        """
        This app allows you to upload a video file or record from webcam for Thai Sign Language detection.
        """
    )
    interpreter = tf.lite.Interpreter(model_path="model-withflip.tflite")
    interpreter.allocate_tensors()
    prediction_fn = interpreter.get_signature_runner("serving_default")
    option = st.radio("Choose input method:", ("Upload a video file", "Record from webcam"))

    if option == "Upload a video file":
        video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            st.write("Processing video...")
            process_video(tfile.name, interpreter, prediction_fn)
    else:
        global video_recorder
        if not video_recorder:
            video_recorder = VideoRecorder()

        ctx = webrtc_streamer(key="example", 
                              mode=WebRtcMode.SENDRECV,
                              video_processor_factory=lambda: video_recorder,
                              rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}))
        
        if ctx.state.playing:
            if not video_recorder.recording:
                video_recorder.start_recording()
                st.write("Started recording...")
            else:
                st.write("Already recording...")
        else:
            st.write("Stopping recording...")
            video_path = video_recorder.stop_recording()
            st.write("Processing video...")
            # Display processed video
            with open(video_path, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)
            process_video(video_path, interpreter, prediction_fn)
            os.remove(video_path)
             
    
# Initialize VideoRecorder
video_recorder = None

def main():
    st.header("Thai Sign Language Detection")

    page_names_to_funcs = {
        "—": intro,
        "Detector": tsl,
    }
    app_mode = st.sidebar.selectbox("Choose the app mode", page_names_to_funcs.keys())

    st.subheader(app_mode)

    page_func = page_names_to_funcs[app_mode]
    page_func()

if __name__ == "__main__":
    main()
