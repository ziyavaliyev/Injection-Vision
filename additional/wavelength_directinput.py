import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

image_path = "wave2.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# extract wave contours
y_coords, x_coords = np.where(image > 100)
if len(y_coords) == 0 or len(x_coords) == 0:
    raise ValueError("No valid wave detected in the image.")

sorted_indices = np.argsort(x_coords)
x = x_coords[sorted_indices]
y = y_coords[sorted_indices]
peaks, _ = find_peaks(-y, distance=len(x) // 5)

peak_distances = np.diff(x[peaks])
estimated_wavelength = np.mean(peak_distances)
estimated_amplitude = np.mean(y[peaks]) - np.min(y)

print(f"Estimated Wavelength: {estimated_wavelength:.2f} pixels")
print(f"Estimated Amplitude: {estimated_amplitude:.2f} pixels")

image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for px, py in zip(x[peaks], y[peaks]):
    cv2.circle(image_bgr, (px, py), radius=5, color=(0, 0, 255), thickness=-1)  # Red dots
plt.figure(figsize=(8, 6))
plt.imshow(image_bgr)
plt.title("Detected Peaks on Wave Image")
plt.axis("off")
plt.show()