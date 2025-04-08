import cv2
import numpy as np
import time

avi_file = "/Users/ziya03/data/280 bar 1M 410µs 2.AVI"
mp4_file = avi_file.replace(".AVI", ".mp4")

cap = cv2.VideoCapture(mp4_file)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = int(1000 / fps)

while True:
    start_time = time.time()
    
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    sobel_x = cv2.Sobel(threshold, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(threshold, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    sobel = np.uint8(255 * sobel / np.max(sobel))

    cv2.imshow("Sobel", sobel)

    elapsed_time = int((time.time() - start_time) * 1000)
    delay = max(frame_time - elapsed_time, 1)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()