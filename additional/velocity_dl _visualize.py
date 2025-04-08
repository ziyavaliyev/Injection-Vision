import cv2
import numpy as np

video_path = "/Users/ziya03/data/280 bar 1M 410µs 2.mp4"
cap = cv2.VideoCapture(video_path)
output_video = video_path.replace(".mp4", "_raft.mp4")
velocity_input = video_path.replace(".mp4", "_velocity.npy")

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.resize(frame, (960, 520)))
cap.release()

# Draw the velocity field
velocities = np.load(velocity_input)
print(f"Loaded velocity data: {velocities.shape}")
step = 20
for i in range(len(velocities)):
    frame = frames[i].copy()
    u, v = velocities[i][..., 0], velocities[i][..., 1]
    
    for y in range(0, frame.shape[0], step):
        for x in range(0, frame.shape[1], step):
            dx, dy = u[y, x] * 5, v[y, x] * 5
            #dont show small velocities
            if np.sqrt(dx**2 + dy**2) > 1:
                cv2.arrowedLine(frame, (x, y), (int(x + dx), int(y + dy)), (0, 255, 0), 1, tipLength=0.3)
            
    
    cv2.imshow("Velocity Field", frame)
    if cv2.waitKey(60) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()