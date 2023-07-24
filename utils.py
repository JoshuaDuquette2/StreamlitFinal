#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     utils.py
   @Author:        Luyao.zhang
   @Date:          2023/5/16
   @Description:
-------------------------------------------------
"""
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import threading
import queue
import time

lock = threading.Lock()
image_container = {"img":None}

def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def infer_uploaded_image(conf, model):
    """
    Execute inference for uploaded image
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                res = model.predict(uploaded_image,
                                    conf=conf)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

                with col2:
                    st.image(res_plotted,
                             caption="Detected Image",
                             use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.xywh)
                    except Exception as ex:
                        st.write("No image is uploaded yet!")
                        st.write(ex)


def infer_uploaded_video(conf, model):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(
                        tfile.name)
                    st_frame = st.empty()
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(conf,
                                                     model,
                                                     st_frame,
                                                     image
                                                     )
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"Error loading video: {e}")

@st.cache_data  # type: ignore
def get_ice_servers():
    return [{"urls": ["stun:stun.l.google.com:19302"]}]


async def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with lock:
        image_container["img"] = img
    return frame

def infer_uploaded_webcam(conf, model):
    """
    Execute inference for webcam.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    try:
        flag = st.button(
            label="Stop running"
        )
        webrtc_ctx = webrtc_streamer(
            key="detect",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            video_frame_callback=video_frame_callback,
            rtc_configuration={"iceServers":get_ice_servers()}
        )

        if not webrtc_ctx.state.playing:
            return

        st_frame = st.empty()
        while not flag:
            if webrtc_ctx.video_receiver:
                with lock:
                    img = image_container["img"]
                if img is None:
                    continue
                
                _display_detected_frames(
                    conf,
                    model,
                    st_frame,
                    img
                )
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")


class FrameReader(threading.Thread):
    def __init__(self, src, queue_size=128):
        super(FrameReader, self).__init__()
        self.stream = cv2.VideoCapture(src)
        self.queue = queue.Queue(maxsize=queue_size)

    def run(self):
        while True:
            if not self.queue.full():
                (grabbed, frame) = self.stream.read()
                if not grabbed:
                    break
                self.queue.put(frame)
            else:
                time.sleep(0.1)  # Rest for 100ms, we have a full queue

    def read(self):
        return self.queue.get()

# in your infer_rtsp_stream function

def infer_rtsp_stream(conf, model, rtsp_url):
    """
    Execute inference for RTSP stream.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param rtsp_url: The URL of the RTSP stream.
    :return: None
    """
    try:
        frame_reader = FrameReader(rtsp_url)
        frame_reader.start()

        flag = st.button(label="Stop running")
        st_frame = st.empty()
        while not flag:
            image = frame_reader.read()
            if image is not None:
                _display_detected_frames(
                    conf,
                    model,
                    st_frame,
                    image
                )
            else:
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")
def infer_rtsp_stream2(conf, model, rtsp_url):
    """
    Execute inference for RTSP stream.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param rtsp_url: The URL of the RTSP stream.
    :return: None
    """
    try:
        flag = st.button(label="Stop running")
        vid_cap = cv2.VideoCapture(rtsp_url)  # RTSP stream
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    conf,
                    model,
                    st_frame,
                    image
                )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")

def infer_uploaded_webcam2(conf, model):
    """
    Execute inference for webcam.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    try:
        flag = st.button(
            label="Start / Stop webcam"
        )
        vid_cap = cv2.VideoCapture(0)  # local camera
        st_frame = st.empty()
        while flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    conf,
                    model,
                    st_frame,
                    image
                )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")