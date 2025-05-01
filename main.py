import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def compute_drift_magnitude(f1, f2):
    prev_gray = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray,
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(magnitude)

# Video input
cap = cv2.VideoCapture("vedio1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
half_sec_frames = int(fps * 0.5)

frame_buffer = []
drift_scores = deque(maxlen=100)
time_stamps = deque(maxlen=100)
frame_count = 0

# Matplotlib setup
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], label='Drift Score')
ax.set_ylim(0, 10)
ax.set_xlabel('Time Window')
ax.set_ylabel('Drift Magnitude')
ax.legend()

# Detection parameters
min_rise_len = 3
min_increase_pct = 0.2  # 20%
rising_regions = []

while True:
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

        # Detect exponential rising regions
        if len(drift_scores) >= min_rise_len:
            last_values = list(drift_scores)[-min_rise_len:]
            is_rising = all((last_values[i+1] - last_values[i]) > min_increase_pct * last_values[i]
                            for i in range(min_rise_len - 1))
            if is_rising:
                rising_regions.append((time_stamps[-min_rise_len], time_stamps[-1]))

        # Clear plot and redraw
        ax.clear()
        ax.set_ylim(0, 10)
        ax.set_xlabel('Time Window')
        ax.set_ylabel('Drift Magnitude')
        ax.plot(time_stamps, drift_scores, label="Drift Score", color='blue')

        # Shade rising areas
        for start, end in rising_regions:
            ax.axvspan(start, end, color='red', alpha=0.3)

        ax.legend()
        plt.draw()
        plt.pause(0.01)

        # Show video with drift overlay
        cv2.putText(frame, f'Drift: {drift_score:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Video', frame)

        frame_buffer = []

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
