import cv2
import numpy as np
import sys

def main():
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "/Users/ziya03/data/280 bar 1M 410µs 2.mp4"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error -> Could not open the video")
        return
    
    roi = None
    points = []
    drawing = False
    paused = False
    last_frame = None
    roi_selected = False
    start_point = None
    current_mouse_pos = (0, 0)

    def video_mouse_callback(event, x, y, flags, param):
        nonlocal roi, drawing, start_point, roi_selected, current_mouse_pos
        
        current_mouse_pos = (x, y)
        
        if paused:
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                start_point = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and drawing:
                drawing = False
                end_point = (x, y)
                x1 = min(start_point[0], end_point[0])
                y1 = min(start_point[1], end_point[1])
                x2 = max(start_point[0], end_point[0])
                y2 = max(start_point[1], end_point[1])
                
                w = x2 - x1
                h = y2 - y1
                
                if w > 10 and h > 10:
                    roi = (x1, y1, w, h)
                    roi_selected = True
                    points.clear()
    def roi_mouse_callback(event, x, y, flags, param):
        nonlocal points, roi_display_scale
        
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            original_x = int(x / roi_display_scale)
            original_y = int(y / roi_display_scale)
            x1, y1, w, h = roi
            if 0 <= original_x < w and 0 <= original_y < h:
                points.append((original_x, original_y))
                print(f"Added point at ({original_x}, {original_y})")

    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", video_mouse_callback)
    
    # Select ROI
    while not roi_selected:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            last_frame = frame.copy()
        else:
            frame = last_frame.copy()
        
        if drawing and start_point:
            cv2.rectangle(frame, start_point, current_mouse_pos, (0, 255, 0), 2)
        if roi:
            x, y, w, h = roi
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imshow("Video", frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            return
        elif key == ord('p'):
            paused = not paused
    cv2.destroyWindow("Video")
    
    # extract roi
    if roi and last_frame is not None:
        x, y, w, h = roi
        roi_img = last_frame[y:y+h, x:x+w].copy()
        roi_display_scale = 4.0
        display_width = int(w * roi_display_scale)
        display_height = int(h * roi_display_scale)
        
        cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("ROI", roi_mouse_callback)
        
        while True:
            display_img = roi_img.copy()
            for point in points:
                cv2.circle(display_img, point, 5, (0, 0, 255), -1)
            
            if len(points) == 2:
                cv2.line(display_img, points[0], points[1], (0, 0, 255), 2)
                distance = np.sqrt(((points[1][0] - points[0][0]) ** 2) + 
                                  ((points[1][1] - points[0][1]) ** 2))
                print(f"Wavelength: {distance:.2f} px")
            
            #show
            display_img = cv2.resize(display_img, (display_width, display_height))
            cv2.imshow("ROI", display_img)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                points = []

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        cv2.destroyAllWindows()