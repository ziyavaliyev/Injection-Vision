import cv2
import numpy as np
import os
import skimage.measure

# =============================== PARAMETERS ===============================
STEP = 15
VELOCITY_THRESHOLD = 1
NEIGHBORHOOD_SIZE = 0
RIGHT_WALL_THRESHOLD = 60
VELOCITY_TRACKING_ENABLED = True
VELOCITY_OUTPUT_ENABLED = True
FRAME_COUNT = 0
VELOCITY_BUFFER = []
NOZZLE_OUTLET_DIAMETER = 300  # micrometer

# =============================== VIDEO SETUP ===============================
avi_file = "/Users/ziya03/data/280 bar 1M 410µs 2.AVI"
mp4_file = avi_file.replace(".AVI", ".mp4")
output_video = avi_file.replace(".AVI", "_processed.mp4")

if not os.path.exists(mp4_file):
    os.system(f"ffmpeg -i '{avi_file}' -c:v libx264 -crf 23 -preset fast '{mp4_file}'")

cap = cv2.VideoCapture(mp4_file)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# =============================== FRAME PROCESSING ===============================
def convert(pixels):
    return pixels * 4  # 1 pixel = 4 micrometers

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    return np.uint8(255 * sobel / np.max(sobel))

ret, prev_frame = cap.read()
prev_sobel = preprocess_frame(prev_frame)

# =============================== OPTICAL FLOW ===============================
i=0
while True:
    ret, next_frame = cap.read()
    if not ret:
        break
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    KERNEL_SIZE = 9
    next_sobel = preprocess_frame(next_frame)
    blurred_sobel = cv2.GaussianBlur(next_sobel, (KERNEL_SIZE, KERNEL_SIZE), 0)
    next_sobel = cv2.subtract(next_sobel, blurred_sobel)
    
    flow = cv2.calcOpticalFlowFarneback(prev_sobel, next_sobel, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    u, v = flow[..., 0], flow[..., 1]
    speed = np.sqrt(u**2 + v**2) * 4
    
    u[speed < VELOCITY_THRESHOLD] = 0
    v[speed < VELOCITY_THRESHOLD] = 0
    u_reduced = skimage.measure.block_reduce(u, (32, 28), np.mean)
    v_reduced = skimage.measure.block_reduce(v, (32, 28), np.mean)
    speed_reduced = skimage.measure.block_reduce(speed, (32, 28), np.mean)

    # =============================== HEATMAP ===============================
    speed_norm = (255 * (speed / np.max(speed))).astype(np.uint8)
    heatmap = cv2.applyColorMap(speed_norm, cv2.COLORMAP_JET)
    sobel_bgr = cv2.cvtColor(next_sobel, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(sobel_bgr, 0.5, heatmap, 0.5, 0)

    # =============================== VELOCITY FIELD ===============================
    """Pool the video into smaller rasters"""
    h, w = next_sobel.shape
    rows, cols = speed_reduced.shape

    for i in range(rows):
        for j in range(cols):
            if speed_reduced[i, j] >= VELOCITY_THRESHOLD:
                x = j * 28 + 14
                y = i * 32 + 16

                u_val = u_reduced[i, j]
                v_val = v_reduced[i, j]

                cv2.arrowedLine(
                    blended,
                    (x, y),
                    (int(x + u_val * 5), int(y + v_val * 5)),
                    (0, 255, 0), 1, tipLength=0.3
                )

    # =============================== LEGEND ===============================
    legend_width = 160
    legend_height = 180
    legend_x, legend_y = w - legend_width - 10, 10  # Fixed position
    legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
    bins = [0, 10, 15, 20, 25, 30, 40, np.inf]
    num_bins = len(bins) - 1

    for i in range(num_bins):
        lower, upper = bins[i], bins[i + 1]
        color_value = int((i / (num_bins - 1)) * 255)
        color = cv2.applyColorMap(np.uint8([[color_value]]), cv2.COLORMAP_JET)[0, 0]
        y1, y2 = i * 20, (i + 1) * 20
        label = f"{lower}-{upper if upper != np.inf else 'inf'} m/s"
        cv2.rectangle(legend, (10, y1), (30, y2), color.tolist(), -1)        
        cv2.putText(legend, label, (40, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    blended[legend_y:legend_y + legend_height, legend_x:legend_x + legend_width] = legend
    out.write(blended)

    cv2.imshow("Velocity Field with Heatmap", blended)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_sobel = next_sobel

cap.release()
out.release()
cv2.destroyAllWindows()