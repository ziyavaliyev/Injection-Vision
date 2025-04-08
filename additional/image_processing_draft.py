import cv2

path = "/Users/ziya03/Desktop/methodology/raw.png"
img = cv2.imread(path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
sobel_x = cv2.Sobel(gaussian, cv2.CV_64F, 1, 0, ksize=3)
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.Sobel(gaussian, cv2.CV_64F, dx=0, dy=1, ksize=3)
sobel_y = cv2.convertScaleAbs(sobel_y)
sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
canny = cv2.Canny(gaussian, 100, 200)
cv2.imshow("Visualize: ", canny)
cv2.waitKey(0)
cv2.destroyAllWindows()