import cv2
import numpy as np
import os

# =============================== PARAMETERS ===============================
STEP = 15
VELOCITY_THRESHOLD = 1
NEIGHBORHOOD_SIZE = 3
RIGHT_WALL_THRESHOLD = 60
VELOCITY_TRACKING_ENABLED = True
VELOCITY_OUTPUT_ENABLED = True
FRAME_COUNT = 0
VELOCITY_BUFFER = []

# =============================== VIDEO SETUP ===============================
avi_file = "/Users/ziya03/data/280 bar 1M 410µs 2.AVI"
mp4_file = avi_file.replace(".AVI", ".mp4")

if not os.path.exists(mp4_file):
    os.system(f"ffmpeg -i '{avi_file}' -c:v libx264 -crf 23 -preset fast '{mp4_file}'")

cap = cv2.VideoCapture(mp4_file)

# =============================== FRAME PROCESSING ===============================
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    return np.uint8(255 * sobel / np.max(sobel))

ret, prev_frame = cap.read()
prev_sobel = preprocess_frame(prev_frame)

# =============================== OPTICAL FLOW ===============================
while True:
    ret, next_frame = cap.read()
    if not ret:
        break

    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if frame_number < 29:
        prev_sobel = preprocess_frame(next_frame)
        continue

    next_sobel = preprocess_frame(next_frame)
    flow = cv2.calcOpticalFlowFarneback(prev_sobel, next_sobel, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    u, v = flow[..., 0], flow[..., 1]
    speed = np.sqrt(u**2 + v**2)
    u[speed < VELOCITY_THRESHOLD] = 0
    v[speed < VELOCITY_THRESHOLD] = 0

    sobel_bgr = cv2.cvtColor(next_sobel, cv2.COLOR_GRAY2BGR)
    h, w = next_sobel.shape
    mid_y = (h // 2) - 26
    cv2.line(next_frame, (0, mid_y), (w, mid_y), (255, 255, 255), 2)

    for y in range(0, h, STEP):
        for x in range(0, w, STEP):
            if speed[y, x] >= VELOCITY_THRESHOLD:
                cv2.arrowedLine(next_frame, (x, y), (int(x + u[y, x] * 5), int(y + v[y, x] * 5)), (0, 255, 0), 1, tipLength=0.3)

    max_x = max((x for x in range(0, w, STEP) if speed[mid_y, x] >= VELOCITY_THRESHOLD), default=None)
    if max_x is not None:
        distance_to_wall = w - max_x
        if distance_to_wall < RIGHT_WALL_THRESHOLD:
            if VELOCITY_TRACKING_ENABLED:
                print("Stopping velocity tracking.")
                VELOCITY_TRACKING_ENABLED = VELOCITY_OUTPUT_ENABLED = False
        elif VELOCITY_OUTPUT_ENABLED:
            x_min, x_max = max(0, max_x - NEIGHBORHOOD_SIZE), min(w - 1, max_x + NEIGHBORHOOD_SIZE)
            VELOCITY_BUFFER.append(np.mean(speed[mid_y, x_min:x_max + 1]))
            FRAME_COUNT += 1
            if FRAME_COUNT == 3:
                print(f"Frame {frame_number}: Mean velocity = {np.mean(VELOCITY_BUFFER):.2f} px/frame, Distance to wall = {distance_to_wall} px")
                FRAME_COUNT = 0
                VELOCITY_BUFFER = []
        if VELOCITY_TRACKING_ENABLED:
            cv2.circle(next_frame, (max_x, mid_y), 5, (0, 0, 255), -1)

    cv2.imshow("Velocity field", next_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_sobel = next_sobel

cap.release()
cv2.destroyAllWindows()