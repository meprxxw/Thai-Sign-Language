import streamlit as st
import mediapipe as mp
import numpy as np
import cv2

st.title("Mediapipe Example")

# Your mediapipe code here
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Example: Process an empty numpy array
empty_image = np.zeros((640, 480, 3), dtype=np.uint8)
results = hands.process(empty_image)

st.write("Processing complete!")
