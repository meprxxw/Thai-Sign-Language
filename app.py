import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import json
import tempfile
import os

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

def process_video(video_path, interpreter, prediction_fn, detection_confidence, tracking_confidence):
    sequence_data = []
    cap = cv2.VideoCapture(video_path)
    with mp_holistic.Holistic(min_detection_confidence=detection_confidence, min_tracking_confidence=tracking_confidence) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            landmarks = extract_coordinates(results)
            sequence_data.append(landmarks)          
    cap.release()
    
    st.video(video_path, format="video/mp4", start_time=0)

    if sequence_data:
        sequence_data = np.array(sequence_data, dtype=np.float32)
        prediction = prediction_fn(inputs=sequence_data)
        sign = np.argmax(prediction["outputs"])
        st.info(f"Predicted Sign: {ORD2SIGN[sign]}")

def intro():
    st.write(
        """
       
    """
    )

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
    
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        process_video(tfile.name, interpreter, prediction_fn, detection_confidence, tracking_confidence)
   

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
