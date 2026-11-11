import cv2
import numpy as np

# Aap ki test image ka path
img_path = r'C:\Users\Abbott Computer\Documents\Measure-It-Segmentation\dataset\test\images\IMG20261104220455.jpg'
img = cv2.imread(img_path)

if img is None:
    print("Image hi nahi mili!")
    exit()

# Image choti karein (processing ke liye)
img = cv2.resize(img, (800, int(800 * img.shape[0] / img.shape[1])))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 30, 150)

# Saare contours dhoondein
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

print(f"Total contours found: {len(contours)}")

for i, c in enumerate(contours[:10]): # Top 10 bade contours check karein
    area = cv2.contourArea(c)
    if area > 5000:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        print(f"Contour {i}: Area={area}, Corners={len(approx)}, W={w}, H={h}")
        # Har bade contour ko draw karein
        cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)

cv2.imshow("Detected Edges & Contours", img)
cv2.waitKey(0)
cv2.destroyAllWindows()