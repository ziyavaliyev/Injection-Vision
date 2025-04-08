import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from collections import deque
import os

# ==================================== PARAMETERS ====================================

"""Depending on the task, following parameters could be fine-tuned to achieve a better result"""

GRADIENT_THRESHOLD = 100              # delete non-relevant gradients (in y-direction) : default=100
EPSILON = 20                          # maximum distance between points in a single cluster : default=20
MIN_PTS = 10                          # minimum number of points to create a cluster : default=10
BOUNDARY_MARGIN = 20                  # ignore clusters near the boundary : default=20
POINT_THICKNESS = 5                   # thickness of cluster points : default=5
PRINT_INTERVAL = 10                   # print every PRINT_INTERVAL frames : default=10
ANGLE_HISTORY_SIZE = PRINT_INTERVAL   # store the last ANGLE_HISTORY_SIZE angle predictions : default=PRINT_INTERVAL
LARGEST_CLUSTERS = 4                  # keep only the largest clusters to detect edges : default=4


# grayscale -> blur -> vertical gradients.
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, dx=0, dy=1, ksize=3)
    sobel_y = cv2.convertScaleAbs(sobel_y)  # Convert back to uint8
    _, sobel_thresh = cv2.threshold(sobel_y, GRADIENT_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    return sobel_thresh

# apply DBSCAN clustering to detect the edges
def get_clusters(edge_points, width, height):
    if len(edge_points) == 0:
        return []

    dbscan = DBSCAN(eps=EPSILON, min_samples=MIN_PTS).fit(edge_points)
    labels = dbscan.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)

    # filter out the clusters to extract only the edges and ignore the detections near the boundary of the frame
    cluster_sizes = dict(zip(unique_labels, counts))
    cluster_sizes.pop(-1, None)
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    valid_clusters = {label for label, _ in sorted_clusters[:LARGEST_CLUSTERS]}

    cluster_data = []
    for label in valid_clusters:
        cluster_coords = edge_points[labels == label]
        if cluster_coords.size == 0:
            continue

        x_min, y_min = np.min(cluster_coords, axis=0)
        x_max, y_max = np.max(cluster_coords, axis=0)

        if (x_min < BOUNDARY_MARGIN or x_max > width - BOUNDARY_MARGIN or
                y_min < BOUNDARY_MARGIN or y_max > height - BOUNDARY_MARGIN):
            continue

        cluster_data.append((label, np.mean(cluster_coords[:, 0]), cluster_coords))

    return cluster_data

# compute average angle from x-axis, given the cluster coordinates
def compute_average_angle(cluster_coords):
    if len(cluster_coords) < 2:
        return None

    #fit a line using least squares and compute the angle
    [vx, vy, _, _] = cv2.fitLine(cluster_coords.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
    angle = np.degrees(np.arctan2(abs(vy), abs(vx)))

    return angle

# preprocess the frame, detect clusters, compute angles and visualize
def process_frame(frame):
    height, width, _ = frame.shape
    sobel_thresh = preprocess_frame(frame)
    edge_points = np.column_stack(np.where(sobel_thresh > 0))
    cluster_data = get_clusters(edge_points, width, height)

    if len(cluster_data) < 2:
        return frame, None

    #select the cluster with the max and min average y-values
    top_cluster = min(cluster_data, key=lambda x: x[1])
    bottom_cluster = max(cluster_data, key=lambda x: x[1])

    angle_top = compute_average_angle(top_cluster[2])
    angle_bottom = compute_average_angle(bottom_cluster[2])

    if angle_top is not None and angle_bottom is not None:
        avg_angle = 90 - np.mean([angle_top, angle_bottom])
    else:
        avg_angle = None

    # visualize
    cluster_image = frame.copy()
    cluster_colors = {top_cluster[0]: (0, 255, 0), bottom_cluster[0]: (255, 0, 0)}

    for cluster in [top_cluster, bottom_cluster]:
        label, _, cluster_coords = cluster
        for (y, x) in cluster_coords:
            color = cluster_colors[label]
            cv2.circle(cluster_image, (x, y), POINT_THICKNESS, color, -1)
    output_frame = cv2.addWeighted(frame, 0.6, cluster_image, 0.4, 0)

    return output_frame, avg_angle


def main(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_delay = int(1000 / fps)  # match real-time playback speed

    angle_history = deque(maxlen=ANGLE_HISTORY_SIZE)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output_frame, avg_angle = process_frame(frame)

        if avg_angle is not None:
            angle_history.append(avg_angle)

        if frame_count % PRINT_INTERVAL == 0 and len(angle_history) == ANGLE_HISTORY_SIZE:
            avg_of_last_20 = np.mean(angle_history)
            print(f"Cone angle: {avg_of_last_20:.2f}°")

        frame_count += 1

        cv2.imshow("DBSCAN clustered edges", output_frame)

        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    avi_file = "/Users/ziya03/data/converted_video.mp4"
    video_path = avi_file.replace(".AVI", ".mp4")
    if not os.path.exists(video_path):
        os.system(f"ffmpeg -i '{avi_file}' -c:v libx264 -crf 23 -preset fast '{video_path}'")
    main(video_path)