import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import json
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, VideoProcessorBase
from PIL import Image, ImageDraw, ImageFont
from  sample_utils.turn import get_ice_servers

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
            mp_drawing.draw_landmarks(image, pred.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=colors['face'], thickness=1, circle_radius=1),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=colors['face'], thickness=1))
        if pred.pose_landmarks:
            mp_drawing.draw_landmarks(image, pred.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=colors['pose'], thickness=2, circle_radius=2),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=colors['pose'], thickness=2))
        if pred.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, pred.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=colors['left_hand'], thickness=1, circle_radius=1),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=colors['left_hand'], thickness=1))
        if pred.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, pred.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=colors['right_hand'], thickness=1, circle_radius=1),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=colors['right_hand'], thickness=1))

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

class SignLanguageProcessor(VideoTransformerBase):
    def __init__(self):
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.sequence_data = []
        self.sign = ""
        self.colors = {
            'face': (255, 255, 255),
            'pose': (255, 255, 255),
            'left_hand': (255, 255, 255),
            'right_hand': (255, 255, 255),
            'font': (255, 255, 255)
        }
        self.draw_landmarks = True
        self.show_predictions_in_video = True

    def update_params(self, draw_landmarks, show_predictions_in_video, colors):
        self.draw_landmarks = draw_landmarks
        self.show_predictions_in_video = show_predictions_in_video
        self.colors = colors

    def predict(self, frame):
        interpreter = tf.lite.Interpreter(model_path="model-withflip.tflite")
        interpreter.allocate_tensors()
        prediction_fn = interpreter.get_signature_runner("serving_default")

        landmarks = np.array(self.sequence_data, dtype=np.float32)
        prediction = prediction_fn(inputs=landmarks)
        sign = np.argmax(prediction["outputs"])
        return ORD2SIGN[sign]

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image, results = mediapipe_detection(image, self.holistic, self.draw_landmarks, self.colors)
        landmarks = extract_coordinates(results)
        self.sequence_data.append(landmarks)
        
        if len(self.sequence_data) % 30 == 0:
            self.sign = self.predict(image)
            self.sequence_data = []

        if self.show_predictions_in_video and self.sign:
            image = draw_text_with_font(image, self.sign, (10, 30), "PK Maehongson Medium.ttf", 32, self.colors['font'])
        
        return image

def real_time_tsl():
    st.write(
        """
        This app allows you to perform real-time Thai Sign Language detection using your webcam.
        """
    )

    webrtc_ctx = webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, 
                                 video_processor_factory=SignLanguageProcessor, 
                                 rtc_configuration={"iceServers": get_ice_servers()},     
                                 video_frame_callback=callback,
                                    media_stream_constraints={"video": True, "audio": False})
,


    if webrtc_ctx.video_processor:
        show_options = st.checkbox("Show Options", value=False)

        if show_options:
            draw_landmarks = st.checkbox("Draw Landmarks", value=True)
            show_predictions_in_video = st.checkbox("Show Predictions in Video", value=True)

            face_color = st.color_picker("Face Color", value="#FFFFFF")
            pose_color = st.color_picker("Pose Color", value="#FFFFFF")
            left_hand_color = st.color_picker("Left Hand Color", value="#FFFFFF")
            right_hand_color = st.color_picker("Right Hand Color", value="#FFFFFF")
            font_color = st.color_picker("Font Color", value="#FFFFFF")

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
            draw_landmarks = True
            show_predictions_in_video = True
            colors = {
                'face': (255, 255, 255),
                'pose': (255, 255, 255),
                'left_hand': (255, 255, 255),
                'right_hand': (255, 255, 255),
                'font': (255, 255, 255)
            }

        webrtc_ctx.video_processor.update_params(draw_landmarks, show_predictions_in_video, colors)

def process_video(video_path, interpreter, prediction_fn, detection_confidence, tracking_confidence, draw_landmarks, show_predictions_in_video, colors):
    sequence_data = []
    cap = cv2.VideoCapture(video_path)
    video_placeholder = st.empty()

    font_path = "PK Maehongson Medium.ttf"
    font_size = 32

    with mp_holistic.Holistic(min_detection_confidence=detection_confidence, min_tracking_confidence=tracking_confidence) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame, results = mediapipe_detection(frame, holistic, draw_landmarks, colors)
            sequence = extract_coordinates(results)
            sequence_data.append(sequence)

            if len(sequence_data) == 30:
                landmarks = np.array(sequence_data, dtype=np.float32)
                prediction = prediction_fn(inputs=landmarks)
                sign = np.argmax(prediction["outputs"])
                sequence_data = []

            if show_predictions_in_video and len(sequence_data) % 30 == 0:
                sign_name = ORD2SIGN[sign]
                frame = draw_text_with_font(frame, sign_name, (10, 30), font_path, font_size, colors['font'])

            video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def tsl_video():
    st.title('Sign Language Video Detection')

    video_file = st.file_uploader('Upload a video file', type=['mp4', 'mov', 'avi', 'mkv'])
    if not video_file:
        st.warning('Please upload a video file.')
        return

    temp_video_path = 'temp_video.mp4'
    with open(temp_video_path, 'wb') as f:
        f.write(video_file.getvalue())

    interpreter = tf.lite.Interpreter(model_path="model-withflip.tflite")
    interpreter.allocate_tensors()
    prediction_fn = interpreter.get_signature_runner("serving_default")

    show_options = st.checkbox("Show Options", value=False)

    if show_options:
        detection_confidence = st.slider('Detection Confidence', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        tracking_confidence = st.slider('Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        draw_landmarks = st.checkbox("Draw Landmarks", value=True)
        show_predictions_in_video = st.checkbox("Show Predictions in Video", value=True)

        face_color = st.color_picker("Face Color", value="#FFFFFF")
        pose_color = st.color_picker("Pose Color", value="#FFFFFF")
        left_hand_color = st.color_picker("Left Hand Color", value="#FFFFFF")
        right_hand_color = st.color_picker("Right Hand Color", value="#FFFFFF")
        font_color = st.color_picker("Font Color", value="#FFFFFF")

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
        detection_confidence = 0.5
        tracking_confidence = 0.5
        draw_landmarks = True
        show_predictions_in_video = True
        colors = {
            'face': (255, 255, 255),
            'pose': (255, 255, 255),
            'left_hand': (255, 255, 255),
            'right_hand': (255, 255, 255),
            'font': (255, 255, 255)
        }

    process_video(temp_video_path, interpreter, prediction_fn, detection_confidence, tracking_confidence, draw_landmarks, show_predictions_in_video, colors)

def display_train_data(train):
    st.subheader('Training Data Preview')
    st.write(train.head())

st.set_page_config(page_title='Sign Language Detection', layout='wide')

def main():
    st.title('Sign Language Detection with MediaPipe and TensorFlow Lite')
    st.sidebar.title('Navigation')
    options = ['Home', 'Live Detector', 'Upload Video', 'Train Data']
    choice = st.sidebar.radio('Select Option', options)

    if choice == 'Home':
        st.write('This application uses MediaPipe Holistic and TensorFlow Lite to perform real-time sign language detection.')
        st.write('Select an option from the sidebar to get started.')

    elif choice == 'Live Detector':
        real_time_tsl()

    elif choice == 'Upload Video':
        tsl_video()

    elif choice == 'Train Data':
        display_train_data(train)

if __name__ == '__main__':
    main()
