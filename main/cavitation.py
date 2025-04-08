import cv2
import numpy as np
import time

avi_file = "/Users/ziya03/data/280 bar 1M 410µs 2.AVI"
mp4_file = avi_file.replace(".AVI", ".mp4")

cap = cv2.VideoCapture(mp4_file)
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

frame_count = 0
roi = None

while True:
    ret, next_frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count == 20:
        roi = cv2.selectROI("Select ROI", next_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
    
    if roi is None:
        continue
    
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prev_gray = next_gray
    
    u, v = abs(flow[..., 0]), flow[..., 1]
    speed = np.sqrt(u**2 + v**2)
    
    x_start, y_start, w, h = roi
    x_end, y_end = x_start + w, y_start + h
    
    frame_with_arrows = next_frame.copy()
    
    for y in range(y_start, y_end, 15):
        for x in range(x_start, x_end, 15):
            if speed[y, x] > 0.2:
                cv2.arrowedLine(frame_with_arrows, (x, y), (int(x + u[y, x] * 5), int(y + v[y, x] * 5)), (0, 255, 0), 1, tipLength=0.3)
    
    #cv2.rectangle(frame_with_arrows, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    cv2.imshow("Video + Velocity Arrows", frame_with_arrows)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()