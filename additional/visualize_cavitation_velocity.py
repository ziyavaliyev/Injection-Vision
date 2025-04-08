import cv2
import numpy as np
import matplotlib.pyplot as plt

avi_file = "/Users/ziya03/data/280 bar 1M 410µs 2.AVI"
mp4_file = avi_file.replace(".AVI", ".mp4")
cap = cv2.VideoCapture(mp4_file)
ret, prev_frame = cap.read()

step = 15
frame_count = 0
roi = None
mean_u = []
mean_v = []
mean_speed = []
frame_numbers = []

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
flow = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2), dtype=np.float32)
flags_farneback = cv2.OPTFLOW_FARNEBACK_GAUSSIAN | cv2.OPTFLOW_USE_INITIAL_FLOW

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    return np.uint8(255 * sobel / np.max(sobel))

while True:
    ret, next_frame = cap.read()
    if not ret:
        break

    KERNEL_SIZE = 9
    next_sobel = preprocess_frame(next_frame)
    blurred_sobel = cv2.GaussianBlur(next_sobel, (KERNEL_SIZE, KERNEL_SIZE), 0)
    next_sobel = cv2.subtract(next_sobel, blurred_sobel)
    
    frame_count += 1
    if frame_count == 20:
        roi = cv2.selectROI("Select ROI", next_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
    
    if roi is None:
        continue

    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prev_gray = next_gray
    
    u, v = flow[..., 0], flow[..., 1]
    
    #Extract ROI
    x_start, y_start, w, h = roi
    x_end, y_end = x_start + w, y_start + h

    frame_with_arrows = next_frame.copy()
    temp_u = 4*10*np.max(abs(u[y_start:y_end, x_start:x_end]))
    temp_v = 4*10*np.max(abs(v[y_start:y_end, x_start:x_end]))

    mean_u.append(temp_u)
    mean_v.append(temp_v)
    mean_speed.append(np.sqrt(temp_u**2 + temp_v**2))
    frame_numbers.append(frame_count)
    
    for y in range(y_start, y_end, step):
        for x in range(x_start, x_end, step):
            if np.sqrt(u[y, x]**2 + v[y, x]**2) > 0.5:
                cv2.arrowedLine(frame_with_arrows, (x, y), (int(x + u[y, x] * 5), int(y + v[y, x] * 5)), (0, 255, 0), 1, tipLength=0.3)
                
    cv2.imshow("Video + Velocity Arrows", frame_with_arrows)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
fig.suptitle(mp4_file)
axs[0].plot(frame_numbers[5:], (mean_u)[5:], label='U (axial-direction)', color='b')
axs[0].set_ylabel("Velocity (meter/second)")
axs[0].legend()
axs[0].grid()
axs[1].plot(frame_numbers[5:], (mean_v)[5:], label="V (radial-direction)', color='r'")
axs[1].set_ylabel("Velocity (meter/second)")
axs[1].legend()
axs[1].grid()
axs[2].plot(frame_numbers[5:], (mean_speed)[5:], label='Speed (Magnitude)', color='g')
axs[2].set_xlabel("Frame Number")
axs[2].set_ylabel("Velocity (meter/second)")
axs[2].legend()
axs[2].grid()
plt.show()