import cv2
import numpy as np
import glob
import os

# 1. Board ki dimensions (Inner corners)
# Aap ke print ke mutabiq yeh 7x9 hai
CHECKERBOARD = (7, 9) 

# 2. Square ka size (mm mein)
# Print par likha hai: Squares are 20x20 mm
SQUARE_SIZE = 20.0 

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = [] 
imgpoints = [] 

# 3D points (0,0,0), (20,0,0), (40,0,0) ....,(120,160,0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

# Images ka path
images_path = glob.glob('calibration/images/*.jpg')
# Agar aap ke phone se .jpeg mein save hui hain toh yeh line use karein:
if len(images_path) == 0:
    images_path = glob.glob('calibration/images/*.jpeg')

print(f"Total images found: {len(images_path)}")

successful_images = 0

for fname in images_path:
    img = cv2.imread(fname)
    if img is None:
        continue
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    if ret == True:
        objpoints.append(objp)
        # Sub-pixel accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        successful_images += 1
        
        # VISUAL CHECK: Yeh image par corners draw karke dikhayega
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Checkerboard Detected', img)
        cv2.waitKey(500) # 0.5 second ke liye show hoga

print(f"Successful images: {successful_images} / {len(images_path)}")
cv2.destroyAllWindows()

if successful_images > 0:
    print("\nCalibrating camera... Please wait.")
    # Calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    print("\n--- Calibration Results ---")
    print(f"Reprojection Error: {ret:.4f} (Kam hona chahiye, ideally < 1.0)")
    print("Camera Matrix:\n", mtx)
    print("Distortion Coefficients:\n", dist)

    # Save to file
    os.makedirs('calibration', exist_ok=True)
    np.savez('calibration/params.npz', mtx=mtx, dist=dist)
    print("\nSuccessfully saved to calibration/params.npz")
else:
    print("\nError: Koi bhi image process nahi hui. Check karein ke images sahi folder mein hain.")