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

if __name__ == "__main__":
    main()
