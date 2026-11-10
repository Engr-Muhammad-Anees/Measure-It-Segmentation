import cv2
import numpy as np
import glob

# Params load karein
params = np.load('calibration/params.npz')
mtx = params['mtx']
dist = params['dist']

# Koi bhi ek image uthayein (apna path check kar lein)
img_path = glob.glob('calibration/images/*.jpg')[0]
img = cv2.imread(img_path)

# Undistort karein
undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

# Dono ko side by side dikhayein
cv2.imshow('Original (Distorted)', cv2.resize(img, (600, 400)))
cv2.imshow('Undistorted (Fixed)', cv2.resize(undistorted_img, (600, 400)))
cv2.waitKey(0)
cv2.destroyAllWindows()