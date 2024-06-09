'''

        Real-Time Version

'''

from streamlit_webrtc import VideoProcessorBase, RTCConfiguration,WebRtcMode,webrtc_streamer
from utils import *
import cv2
import streamlit as st
import mediapipe as mp
import av
from PIL import ImageFont, Image, ImageDraw
import numpy as np

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

actions = np.array(['สวัสดี', 'ตก', 'รอ', 'กลับ', 'ขอบคุณ', 'ตัด', 'ลง', 'ขึ้น', 'เฝ้า','คุย', 'ช่วย', 'เชื่อ', 'ฟัง', 'มอง', 'พา', 'ชวน หลีกภัย', 'ทักษิณ ชินวัตร', 'ขนม', 'พิธา ลิ้มเจริญรัตน์', 'ศิริกัญญา ตันสกุล','แบก', 'อนุทิน ชาญวีรกูล', 'รังสิมันต์ โรม', 'พีระพันธุ์ สาลีรัฐวิภาค', 'คุณหญิงสุดารัตน์ เกยุราพันธุ์', 'พลเอกอนุพงษ์ เผ่าจินดา', 'สุวัจน์ ลิปตพัลลภ', 'กรณ์ จาติกวณิช', 'วราวุธ ศิลปะอาชา', 'พล.ต.อ.เสรีพิศุทธ์ เตมียเวสวราวุธ ศิลปะอาชา', 'ศักดิ์สยาม ชิดชอบ', 'ชาดา ไทยเศรษฐ์', 'สุชัชวีร์ สุวรรณสวัสดิ์', 'จุรินทร์ ลักษณวิศิษฏ์', 'ไตรรงค์ สุวรรณคีรี', 'พลเอกประยุทธ์ จันทร์โอชา', 'นฤมล ภิญโญสินวัฒน์', 'ธรรมนัส พรหมเผ่า', 'ชัยวุฒิ ธนาคมานุสรณ์', 'ไพบูลย์ นิติตะวัน', 'พลเอกประวิตร วงษ์สุวรรณ', 'พริษฐ์ วัชรสินธุ', 'ยิ่งลักษณ์ ชินวัตร', 'ณัฐวุฒิ ใสยเกื้อ', 'นพ.ชลน่าน ศรีแก้ว', 'เศรษฐา ทวีสิน', 'แพทองธาร ชินวัตร', 'อะไร', 'สมัคร', 'กระโดด', 'ยก', 'ชน', 'ผ่าน', 'แนะนำ', 'จำได้', 'ปลูก', 'รวม', 'ทำหาย', 'เจอ', 'หาย', 'พนักงานขาย', 'นักเขียนโปรแกรม', 'พ่อครัว', 'เจ้าหน้าที่ตำรวจ', 'เนื้อลูกแกะ', 'เนื้อหมู', 'คุณสบายดีไหม', 'ดูเหมือน', 'วาง', 'อยู่', 'ก', 'ข', 'ค', 'ฆ', 'ต', 'ถ', 'ง', 'จ', 'ฉ', 'ช', 'ซ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ล', 'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ'])

model = load_model('action2.h5',actions)

def main():
    st.header("Live stream processing")

    sign_language_det = "Sign Language Live Detector"
    app_mode = st.sidebar.selectbox( "Choose the app mode",
        [
            sign_language_det
        ],
    )

    st.subheader(app_mode)

    if app_mode == sign_language_det:
        sign_language_detector()


def sign_language_detector():

    class OpenCVVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.sequence = []
            self.sentence = []
            self.predictions = []
            self.threshold = 0.8

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            flip_img = cv2.flip(img, 1)

            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

                image, results = mediapipe_detection(flip_img, holistic)


                draw_styled_landmarks(image, results)


                keypoints = extract_keypoints(results)
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-30:]

                if len(self.sequence) == 30:
                    res = model.predict(np.expand_dims(self.sequence, axis=0))[0]
                    self.predictions.append(np.argmax(res))

                    if np.unique(self.predictions[-10:])[0]==np.argmax(res):
                        if res[np.argmax(res)] > self.threshold: 
                            if len(self.sentence) > 0: 
                                if actions[np.argmax(res)] != self.sentence[-1]:
                                    self.sentence.append(actions[np.argmax(res)])
                            else:
                                self.sentence.append(actions[np.argmax(res)])

                    if len(self.sentence) > 5: 
                        self.sentence = self.sentence[-5:]

                cv2.rectangle(image, (0,0), (640, 40), (234, 234, 77), 1)
                fontpath = "./angsana.ttc"
                font = ImageFont.truetype(fontpath, 30)
                img_pil = Image.fromarray(image)
                draw = ImageDraw.Draw(img_pil)
                draw.text((30, 0),  ' '.join(self.sentence), font = font, fill = (255, 255, 255, 255))
                image = np.array(img_pil)

            return av.VideoFrame.from_ndarray(image, format="bgr24")


    webrtc_ctx = webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=OpenCVVideoProcessor,
        async_processing=True,
    )


if __name__ == "__main__":
    main()
