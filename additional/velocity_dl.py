import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

"""
This script extracts the optical flow from nozzle injection videos of diesel engines. 
Since RAFT-based processing is computationally intensive, we use this script to compute and save the velocities first. 
The resulting data can then be visualized using the velocity_dl_visualize.py script.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained RAFT model
weights = Raft_Large_Weights.DEFAULT
model = raft_large(weights=weights).to(device).eval()

video_path = "/Users/ziya03/data/converted_video.mp4"
cap = cv2.VideoCapture(video_path)
output_video = video_path.replace(".mp4", "_raft.mp4")
velocity_output = video_path.replace(".mp4", "_velocity.npy")

def preprocess_frame(frame):
    return cv2.resize(frame, (960, 520))

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(preprocess_frame(frame))
cap.release()

print(f"Total frames: {len(frames)}")

transform = T.ToTensor()
velocities = []  # store the velocities here

for i in range(len(frames) - 1):
    prev_frame = frames[i]
    next_frame = frames[i + 1]

    prev_tensor = transform(prev_frame).unsqueeze(0).to(device)
    next_tensor = transform(next_frame).unsqueeze(0).to(device)

    with torch.no_grad():
        flow = model(prev_tensor, next_tensor)[-1].squeeze().permute(1, 2, 0).cpu().numpy()

    velocities.append(flow)

    print(f"Processed frame {i+1}/{len(frames)-1}")

# Save the velocities
np.save(velocity_output, np.array(velocities))
print(f"Velocities saved as {velocity_output}")