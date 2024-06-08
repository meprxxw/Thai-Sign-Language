import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import time
import tensorflow as tf
import pandas as pd
from queue import Queue
from multiprocessing import Process
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode
import av

# Suppress warnings
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Mediapipe and drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Load sign language mapping
train = pd.read_csv('train.csv')
train['sign_ord'] = train['sign'].astype('category').cat.codes
SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

# Define a sentinel object for stopping the subprocess
_SENTINEL_ = object()

class FakeLandmarkObject:
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class FakeLandmarksObject:
    def __init__(self, landmark):
        self.landmark = landmark

class FakeResultObject:
    def __init__(self, pose_landmarks):
        self.pose_landmarks = pose_landmarks

def pose_process(
    in_queue: Queue,
    out_queue: Queue,
    static_image_mode,
    model_complexity,
    min_detection_confidence,
    min_tracking_confidence,
):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    while True:
        input_item = in_queue.get(timeout=10)
        if isinstance(input_item, type(_SENTINEL_)) and input_item == _SENTINEL_:
            break

        results = pose.process(input_item)
        if results.pose_landmarks:
            picklable_results = FakeResultObject(pose_landmarks=FakeLandmarksObject(landmark=[
                FakeLandmarkObject(
                    x=pose_landmark.x,
                    y=pose_landmark.y,
                    z=pose_landmark.z,
                    visibility=pose_landmark.visibility,
                ) for pose_landmark in results.pose_landmarks.landmark
            ]))
        else:
            picklable_results = FakeResultObject(pose_landmarks=None)

        out_queue.put_nowait(picklable_results)

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self._in_queue = Queue()
        self._out_queue = Queue()
        self._pose_process = Process(target=pose_process, kwargs={
            "in_queue": self._in_queue,
            "out_queue": self._out_queue,
            "static_image_mode": False,
            "model_complexity": 1,
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5,
        })

        self._pose_process.start()
        self.sequence_data = []
        self.last_prediction = None
        self.last_prediction_time = 0
        self.prediction_display_duration = 3  # seconds
        self.font_path = "./angsana.ttc"
        self.font_size = 32
        self.font = ImageFont.truetype(self.font_path, self.font_size)

        model_path = "model-withflip.tflite"
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found at {model_path}")

        file_size = os.path.getsize(model_path)
        if file_size < 7:
            raise ValueError(f"Model file {model_path} is too small. Size: {file_size} bytes")

        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.prediction_fn = self.interpreter.get_signature_runner("serving_default")
        except Exception as e:
            raise ValueError(f"Failed to load TFLite model. Error: {e}")

    def __del__(self):
        self._in_queue.put_nowait(_SENTINEL_)
        self._pose_process.join(timeout=10)

    def draw_text_on_image(self, image, text, position, font, color=(0, 255, 0)):
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)
        draw.text(position, text, font=font, fill=color)
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        return image_cv

    def extract_coordinates(self, results):
        if results.pose_landmarks:
            pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark])
        else:
            pose = np.zeros((33, 3))
        return pose

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        self._in_queue.put_nowait(image)

        try:
            results = self._out_queue.get(timeout=10)
        except Queue.Empty:
            return av.VideoFrame.from_ndarray(image, format="bgr24")

        landmarks = self.extract_coordinates(results)
        self.sequence_data.append(landmarks)

        if len(self.sequence_data) == 30:  # Process every 30 frames
            try:
                input_data = np.array([self.sequence_data], dtype=np.float32)  # Wrap in a list for batch processing
                prediction = self.prediction_fn(inputs=input_data)
                sign = np.argmax(prediction["outputs"])
                message = ORD2SIGN[sign]
                self.last_prediction = message
                self.last_prediction_time = time.time()
                self.sequence_data = []  # Reset sequence data after prediction
            except Exception as e:
                st.error(f"Prediction error: {e}")

        current_time = time.time()
        if self.last_prediction and (current_time - self.last_prediction_time < self.prediction_display_duration):
            image = self.draw_text_on_image(image, self.last_prediction, (3, 30), self.font, (0, 0, 0))

        return av.VideoFrame.from_ndarray(image, format="bgr24")

def live_detector():
    webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

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
    model_path = "model-withflip.tflite"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return
    
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        prediction_fn = interpreter.get_signature_runner("serving_default")
    except Exception as e:
        st.error(f"Failed to load TFLite model. Error: {e}")
        return

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

def process_video(video_path, interpreter, prediction_fn):
    # Process the video with Mediapipe and the model
    cap = cv2.VideoCapture(video_path)
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    sequence_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image, results = mediapipe_detection(frame, holistic)
        landmarks = extract_coordinates(results)
        sequence_data.append(landmarks)

        if len(sequence_data) == 30:  # Process every 30 frames
            input_data = np.array([sequence_data], dtype=np.float32)  # Wrap in a list
            prediction = prediction_fn(inputs=input_data)
            sign = np.argmax(prediction["outputs"])
            message = ORD2SIGN[sign]
            st.write(f"Detected Sign: {message}")
            sequence_data = []  # Reset sequence data after prediction

    cap.release()

def mediapipe_detection(image, holistic):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_coordinates(results):
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros((468, 3))
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    return np.concatenate([face, lh, pose, rh])

def record_video():
    # Logic to record video using webcam
    # Implement as needed, possibly using OpenCV or other libraries
    pass

if __name__ == "__main__":
    main()
