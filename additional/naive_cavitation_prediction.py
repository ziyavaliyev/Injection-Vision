import cv2
import numpy as np
import scipy.sparse

avi_file = "/Users/ziya03/data/280 bar 1M 410µs 2.AVI"
mp4_file = avi_file.replace(".AVI", ".mp4")

cap = cv2.VideoCapture(mp4_file)

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
frame_count = 0
roi = None
roi_selected = False

cavitation_bubbles = []
arrow_list = []
while True:
    ret, next_frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count == 20 and not roi_selected:
        roi = cv2.selectROI("Select ROI", next_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
        roi_selected = True
    
    if not roi_selected:
        continue

    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prev_gray = next_gray

    u, v = abs(flow[..., 0]), flow[..., 1]
    speed = np.sqrt(u**2 + v**2)
    if frame_count < 50:
        cv2.imshow("Video + Velocity Paths", next_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    x_start, y_start, w, h = roi
    x_end, y_end = x_start + w, y_start + h

    frame_with_arrows = next_frame.copy()


    to_remove = []  # Bubbles to be removed
    to_add = []     #new bubbles to be added
    for bubble in cavitation_bubbles:
        if bubble["frame"] != frame_count:
            new_x = int(bubble["x"] + np.ceil(bubble["u"]))
            new_y = int(bubble["y"] + np.ceil(bubble["v"]))
            if new_x <= x_end and (new_y < y_start or new_y > y_end):
                new_y = bubble["y"]
            """new_u = u[new_y, new_x]
            new_v = v[new_y, new_x]"""
            try:
                new_u_neighborhood_max = np.max(u[new_y-3:new_y+4, new_x-3:new_x+4])
                new_u = u[new_y, new_x]
                new_v = v[new_y, new_x]
            except:
                new_u_neighborhood_max = new_u
                
            #1.75 is the coefficient that needs to be added in order to track the spray
            new_bubble = {
                        "frame": frame_count,
                        "y": new_y + int(np.round(bubble["v"])*1.75),
                        "x": new_x + int(np.round(bubble["u"])*1.75),
                        "u": bubble["u"],#new_u
                        "v": new_v,
                        "new_u_neighborhood_max": new_u_neighborhood_max
                        }
            


            if True: # this can be adjusted if needed
                arrow_list.append((
                    (bubble["x"], bubble["y"]),
                    (new_x + int(np.round(bubble["u"])*1.75), new_y + int(np.round(bubble["v"])*1.75)),
                    (0, 0, 255), 1
                ))
                to_add.append(new_bubble) 
            to_remove.append(bubble)

    #Remove bubbles
    for bubble in to_remove:
        cavitation_bubbles.remove(bubble)

    # Add new bubbles
    cavitation_bubbles.extend(to_add)
    u, v, speed = u[y_start:y_end, x_start:x_end], v[y_start:y_end, x_start:x_end], speed[y_start:y_end, x_start:x_end]

    u_sparse = scipy.sparse.csr_matrix(u)
    v_sparse = scipy.sparse.csr_matrix(v)

    rows, cols, values = scipy.sparse.find(u_sparse)
    sorted_indices = np.argsort(values)[::-1]

    # select top 5 velocities with at least 10 pixels far away from each other
    selected_points = []
    for idx in sorted_indices:
        candidate_point = (rows[idx] + y_start, cols[idx] + x_start)
        
        #minimal distance of 20 pixels
        if all(np.linalg.norm(np.array(candidate_point) - np.array(p)) >= 20 for p in selected_points):
            selected_points.append(candidate_point)
        
        if u[selected_points[-1][0] - y_start, selected_points[-1][1] - x_start] < 25:
            break  #stop when velocity of the found cavitation bubble is too small

    print(f"Frame: {frame_count}, Velocity mean: {40*np.mean(speed):.2f} m/s")

    for (y, x) in selected_points:
        if u[y - y_start, x - x_start] > 0.5:
            cv2.arrowedLine(frame_with_arrows, (x, y), (int(x + u[y - y_start, x - x_start]*40), int(y + v[y - y_start, x - x_start]*40)), (0, 255, 0), 1, tipLength=0.3)
            try:
                new_u_neighborhood_max = np.max(u[new_y-3:new_y+4, new_x-3:new_x+4])
            except:
                new_u_neighborhood_max = u[y - y_start, x - x_start]
            cavitation_bubbles.append({
                "frame": frame_count,
                "y": y,
                "x": x,
                "u": u[y - y_start, x - x_start],
                "v": v[y - y_start, x - x_start],
                "new_u_neighborhood_max": new_u_neighborhood_max
            })
    for arrow in arrow_list:
        start, end, color, thickness = arrow
        cv2.arrowedLine(frame_with_arrows, start, end, color, thickness)
    cv2.imshow("Video + Velocity Paths", frame_with_arrows)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()