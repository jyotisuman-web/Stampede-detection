import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from collections import deque
from PIL import Image

def compute_drift_magnitude(f1, f2):
    prev_gray = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray,
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(magnitude)

# --- Streamlit UI ---
st.title("Real-Time Drift Detection for Stampede Monitoring")
video_path = st.text_input("Enter video path", "your_video.mp4")
start_button = st.button("Start Analysis")

if start_button:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    half_sec_frames = int(fps * 0.5)

    drift_scores = deque(maxlen=100)
    time_stamps = deque(maxlen=100)
    rising_regions = []
    frame_buffer = []
    frame_count = 0

    frame_placeholder = st.empty()
    graph_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_buffer.append(frame)
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

            # Detect exponential rise
            if len(drift_scores) >= 3:
                last_values = list(drift_scores)[-3:]
                is_rising = all((last_values[i+1] - last_values[i]) > 0.2 * last_values[i]
                                for i in range(2))
                if is_rising:
                    rising_regions.append((time_stamps[-3], time_stamps[-1]))

            # Plot graph
            fig, ax = plt.subplots()
            ax.plot(time_stamps, drift_scores, label='Drift Score', color='blue')
            ax.set_ylim(0, 10)
            ax.set_xlabel('Time')
            ax.set_ylabel('Drift Magnitude')

            for start, end in rising_regions:
                ax.axvspan(start, end, color='red', alpha=0.3)

            ax.legend()
            graph_placeholder.pyplot(fig)

            # Display video frame
            cv2.putText(frame, f"Drift: {drift_score:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_image = Image.fromarray(frame_rgb)
            frame_placeholder.image(frame_image, channels="RGB")

            frame_buffer = []

    cap.release()
