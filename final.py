#crowd management 
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import time
import winsound
from deep_sort_realtime.deepsort_tracker import DeepSort

#sidebar
st.set_page_config(layout="wide")
st.sidebar.title(" Settings")
st.title("Crowd Management")

source_type = st.sidebar.radio("Select Video Source", ["Webcam", "Upload File", "Path_url"])


threshold_drift_delta = 0.5
risk_threshold_low = 50
risk_threshold_medium = 80

# different ways to get video
cap = None

if 'cap' in st.session_state:
    st.session_state.cap.release()
    del st.session_state['cap']

if source_type == "Webcam":
    cap = cv2.VideoCapture(0)
    st.session_state.cap = cap

elif source_type == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        cap = cv2.VideoCapture("temp_video.mp4")
        st.session_state.cap = cap
    else:
        st.warning("Please upload a video.")
        st.stop()

elif source_type == "Path_url":
    url = st.sidebar.text_input(
        "Enter RTSP / HTTP URL",
        value="rtsp://username:password@192.168.1.100:554/stream1"
    )
    cap = cv2.VideoCapture(url)
    st.session_state.cap = cap
if cap is None or not cap.isOpened():
    st.error("Unable to open video source.")
    st.stop()
# vedio properties
fps = cap.get(cv2.CAP_PROP_FPS) #it can process 30 frames per second
#print(fps)
#cosidering 0.5 sec frames 
half_sec_frames = int(fps * 0.5) #6 frame in asecond

#model
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)
#Background Subtraction
#It is also a Gaussian Mixture-based Background/Foreground Segmentation Algorithm.
#  It provides better adaptability to varying scenes due illumination changes etc.
fgbg = cv2.createBackgroundSubtractorMOG2()
frame_buffer = []
# collection store multiple item in a single variable
drift_scores = deque(maxlen=200)
time_stamps = deque(maxlen=200)
alert_regions = []
frame_count = 0
last_alert_time = 0

# 
def compute_drift_magnitude(f1, f2):
    # Converts each frame to grayscale - we previously  
    # only converted the first frame to grayscale 
    prev_gray = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
       
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray,
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Computes the magnitude and angle of the 2D vectors 
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(magnitude)

model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

def calculate_risk(density, speed, anomalies):
    risk_score = (density * 0.4) + (speed * 0.3) + (anomalies * 0.3)
    return min(100, risk_score * 2)

def generate_drift_plot(timestamps, scores, alert_regions):
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(timestamps, scores, label="Drift Magnitude", color='blue')
    ax.set_ylim(0, max(10, max(scores, default=1)))
    ax.set_xlabel("Time")
    ax.set_ylabel("Drift")

    for start, end in alert_regions:
        ax.axvspan(start, end, color='red', alpha=0.3)

    ax.legend()
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return img

# STREAMLIT PLACEHOLDERS 
video_placeholder = st.empty()
graph_placeholder = st.empty()

# ----------------- MAIN LOOP ------------------
while cap.isOpened():
    # ret = a boolean return value from getting 
    # the frame, frame = the current frame being 
    # projected in the video 
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (750, 380))
    fgmask = fgbg.apply(frame)
    frame_buffer.append(frame.copy())
    frame_count += 1

    if len(frame_buffer) == half_sec_frames:
        start_frame = frame_buffer[0]
        mid_frame = frame_buffer[half_sec_frames // 2]
        end_frame = frame_buffer[-1]

        drift1 = compute_drift_magnitude(start_frame, mid_frame)
        drift2 = compute_drift_magnitude(mid_frame, end_frame)
        drift_score = (drift1 + drift2) / 2

        drift_scores.append(drift_score)
        time_stamps.append(len(time_stamps))

        # people_count, results = count_people(frame)/
        results = model(frame)[0]
        detections = []
        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            if int(cls) == 0:  # Person 
                x1, y1, x2, y2 = box[:4]
                w, h = int(x2 - x1), int(y2 - y1)
                detections.append(([int(x1), int(y1), w, h], float(conf), 'person'))

        tracks = tracker.update_tracks(detections, frame=frame)
        people_count = len(tracks)
        density = people_count / (frame.shape[0] * frame.shape[1])
        speed = drift_score
        anomalies = 0

        is_alert = False
        if len(drift_scores) >= 3:
            a, b, c = list(drift_scores)[-3:]
            if a < b < c and (c - a) > threshold_drift_delta:
                is_alert = True
                anomalies = 1
                alert_regions.append((len(time_stamps) - 3, len(time_stamps) - 1))
                if time.time() - last_alert_time > 1:
                    winsound.Beep(1000, 500)
                    last_alert_time = time.time()

        final_risk_score = calculate_risk(density, speed, anomalies)
        
        if final_risk_score < risk_threshold_low:
            risk_color = (0, 255, 0)
        elif final_risk_score <= risk_threshold_medium:
            risk_color = (0, 165, 255)
        elif final_risk_score>risk_threshold_medium:
            risk_color = (0, 0, 255)

        flashing = is_alert and (int(time.time() * 2) % 2 == 0)

        for cls, box in zip(results.boxes.cls, results.boxes.xyxy):
            if int(cls) == 0:
                x1, y1, x2, y2 = map(int, box[:4])
                color = (0, 0, 255) if flashing else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        cv2.putText(frame, f"People Count: {people_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Display Risk Score
        cv2.putText(frame, f"Risk Score: {final_risk_score:.2f}", (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, risk_color, 2)

        # Add Risk Level Text in corresponding color
        if final_risk_score < risk_threshold_low:
            risk_text = "LOW RISK"
            risk_text_color = (0, 255, 0)  # Green
        elif final_risk_score < risk_threshold_medium:
            risk_text = "MEDIUM RISK"
            risk_text_color = (0, 165, 255)  # Orange
        else:
            risk_text = "HIGH RISK"
            risk_text_color = (0, 0, 255)  # Red

        cv2.putText(frame, risk_text, (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, risk_text_color, 3)

        if is_alert:
            cv2.putText(frame, "HIGH ALERT", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 5)

        drift_graph = generate_drift_plot(time_stamps, drift_scores, alert_regions)
        drift_graph = cv2.cvtColor(drift_graph, cv2.COLOR_RGBA2RGB)
        graph_image = Image.fromarray(drift_graph)

        fgmask_color = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((frame, fgmask_color))
        video_placeholder.image(combined, channels="BGR", use_container_width=True)
        graph_placeholder.image(graph_image)

        frame_buffer = []



