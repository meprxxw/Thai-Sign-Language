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

def real_time_tsl():
    interpreter = tf.lite.Interpreter(model_path="model-traintestflip.tflite")
    interpreter.allocate_tensors()
    prediction_fn = interpreter.get_signature_runner("serving_default")

    sequence_data = []
    cap = cv2.VideoCapture(0)

    last_prediction = None
    last_prediction_time = 0
    prediction_display_duration = 3  # seconds

    video_placeholder = st.empty()
    message_placeholder = st.empty()

    font_path = "./angsana.ttc"
    font_size = 32
    font = ImageFont.truetype(font_path, font_size)

    messages = []  # List to store messages

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            # draw(image, results)
           
            # Create a placeholder for the video feed
            landmarks = extract_coordinates(results)
            sequence_data.append(landmarks)

            if len(sequence_data) % 30 == 0:
                prediction = prediction_fn(inputs=np.array(sequence_data, dtype=np.float32))
                sign = np.argmax(prediction["outputs"])
                
                message = ORD2SIGN[sign]
                messages.append(message)
                if len(messages) > 5:
                    messages.pop(0)  # Keep only the last 5 messages
                last_prediction = message
                last_prediction_time = time.time()
                sequence_data = []

            current_time = time.time()
            if last_prediction and (current_time - last_prediction_time < prediction_display_duration):
                image = draw_text_on_image(image, last_prediction, (3, 30), font, (0, 0, 0))

            # Display the frame using Streamlit
            video_placeholder.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")
            # Display messages
            message_text = " ".join(messages)  # Concatenate messages with newline
            message_placeholder.info(message_text)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

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

def record_video():
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(temp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    recording = True
    while recording:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        cv2.imshow('Recording', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if st.button("Stop Recording"):
            recording = False

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return temp_video_file.name

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

def main():
    st.header("Thai Sign Language Detection")

    page_names_to_funcs = {
        "—": intro,
        "Detector": tsl,
        "Live Detector": real_time_tsl,
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
            real_time_tsl()
    else:
        page_func = page_names_to_funcs[app_mode]
        page_func()

if __name__ == "__main__":
    main()
