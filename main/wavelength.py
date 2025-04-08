import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def fourier_series(x, *params):
    a0 = params[0]
    terms = (len(params) - 1) // 2
    result = a0
    for n in range(terms):
        an = params[1 + 2 * n]
        bn = params[2 + 2 * n]
        result += an * np.cos((n + 1) * x) + bn * np.sin((n + 1) * x)
    return result

# Load video
avi_file = "/Users/ziya03/data/280 bar 1M 410µs 2.AVI"
mp4_file = avi_file.replace(".AVI", ".mp4")
video_path = mp4_file

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error -> Cannot open video.")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
paused = False
selected_frame = None

def extract_contour(image):
    edges = preprocess(image)

    y_coords, x_coords = np.where(edges > 50)
    sorted_indices = np.argsort(x_coords)
    x_data = x_coords[sorted_indices]
    y_data = y_coords[sorted_indices]
    
    return x_data, y_data

def fit_fourier(x_data, y_data, N_terms=3):
    y_mean = np.mean(y_data)
    y_std = np.std(y_data)
    
    # Initialize parameters with small variations
    initial_params = [y_mean] + list(np.random.uniform(-y_std, y_std, 2 * N_terms))
    params, _ = curve_fit(fourier_series, x_data, y_data, p0=initial_params)

    return params

def compute_wavelength_amplitude(params, peaks):
    """ Estimate wavelength and amplitude from Fourier coefficients """

    if len(params) < 3:
        return None, None  #Not enough terms to estimate

    #wavelength as mean of the differences between peaks
    wavelength = np.mean(np.diff(peaks[0]))

    # Amplitude as the largest Fourier coefficient
    amplitudes = np.sqrt(np.array(params[1::2])**2 + np.array(params[2::2])**2)
    max_amplitude = np.max(amplitudes) if len(amplitudes) > 0 else None

    return wavelength, max_amplitude

def find_wave_segment(x_data, fitted_curve, wavelength):
    """Finds one wave period in the middle of the curve."""
    mid_index = len(x_data) // 2
    start_x = x_data[mid_index]
    end_x = start_x + wavelength

    # find closest x-values in data
    start_idx = np.abs(x_data - start_x).argmin()
    end_idx = np.abs(x_data - end_x).argmin()

    return x_data[start_idx:end_idx], fitted_curve[start_idx:end_idx]

def process_frame(frame):
    global selected_frame
    roi = cv2.selectROI("Select ROI", preprocess(frame), False, False)
    cv2.destroyAllWindows()
    if roi[2] > 0 and roi[3] > 0:
        selected_frame = frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        x_data, y_data = extract_contour(selected_frame)
        params = fit_fourier(x_data, y_data)

        if params is None:
            print("Could not fit Fourier series.")
            return

        fitted_curve = fourier_series(x_data, *params)
        peaks = find_peaks(fitted_curve)

        wavelength, amplitude = compute_wavelength_amplitude(params, peaks)
        wave_x, wave_y = find_wave_segment(x_data, fitted_curve, wavelength)
        print("λ (Wavelength): {wavelength:.2f} px\nA (Amplitude): {amplitude:.2f} px")

        plt.figure(figsize=(10, 5))
        plt.imshow(preprocess(selected_frame), cmap='gray', extent=[0, x_data[-1], np.max(y_data), np.min(y_data)])
        plt.plot(x_data, fitted_curve-1, color="red", linewidth=2) #, label="Fitted Fourier"
        plt.plot(wave_x, wave_y-1, color="green", linewidth=3)#, label="One Wavelength" # Highlight one wave period
        plt.axvline(x=wave_x[0], color="green", linestyle="dashed", linewidth=2)#, label="Wavelength Start"
        plt.axvline(x=wave_x[-1], color="green", linestyle="dashed", linewidth=2)#, label="Wavelength End"
        plt.scatter(x_data[peaks[0]], fitted_curve[peaks[0]]-1, s=5, label="Peaks", color="red")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

def main():
    global paused
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Video Playback", frame)
        
        key = cv2.waitKey(15) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('p'):  # Pause/Play
            paused = not paused
        elif key == ord('s') and paused:  # Select ROI
            cv2.destroyAllWindows()
            process_frame(frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()