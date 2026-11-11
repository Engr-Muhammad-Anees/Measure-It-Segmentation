import cv2
import numpy as np
import os

# --- 1. SETTINGS ---
CALIB_PATH = 'calibration/params.npz'
CARD_WIDTH_MM = 85.6 

# --- 2. LOAD CALIBRATION ---
calib = np.load(CALIB_PATH)
mtx = calib['mtx']
dist = calib['dist']

def find_blue_card(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Blue range for ID Card
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, w, h

def measure_object(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Image nahi mili: {image_path}")
        return
        
    # 1. Undistort image
    img_undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    # 2. Find Card
    card_bbox = find_blue_card(img_undist)
    if not card_bbox:
        print("❌ Card nahi mila.")
        return
    cx, cy, cw, ch = card_bbox
    # Calculate scale
    mm_per_pixel = CARD_WIDTH_MM / max(cw, ch)
    print(f"✅ Card Found! Scale: {mm_per_pixel:.4f} mm/pixel")
    
    # 3. Find Cup (Using Color and Shape - OpenCV)
    hsv = cv2.cvtColor(img_undist, cv2.COLOR_BGR2HSV)
    
    # Cup ke liye light/white color ka mask banayein
    lower_white = np.array([0, 0, 120]) # Light intensity threshold
    upper_white = np.array([180, 60, 255])
    cup_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Card ko mask se hata dein (taake woh cup na samjha jaye)
    cv2.rectangle(cup_mask, (cx, cy), (cx+cw, cy+ch), 0, -1)
    
    # Mask ko saaf karein (noise remove)
    kernel = np.ones((15,15), np.uint8)
    cup_mask = cv2.morphologyEx(cup_mask, cv2.MORPH_CLOSE, kernel)
    cup_mask = cv2.morphologyEx(cup_mask, cv2.MORPH_OPEN, kernel)
    
    # 4. Find Cup Contours
    contours, _ = cv2.findContours(cup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("❌ Cup nahi mila (Color threshold failed).")
        return
        
    # Sab se bara safe area dhoondein
    largest_cup = max(contours, key=cv2.contourArea)
    cup_x, cup_y, cup_w, cup_h = cv2.boundingRect(largest_cup)
    
    # 5. Convert to MM
    cup_width_mm = cup_w * mm_per_pixel
    cup_height_mm = cup_h * mm_per_pixel
    
    # 6. Draw Output
    img_result = img_undist.copy()
    # Card Box (Green)
    cv2.rectangle(img_result, (cx, cy), (cx+cw, cy+ch), (0, 255, 0), 3) 
    # Cup Box (Red)
    cv2.rectangle(img_result, (cup_x, cup_y), (cup_x+cup_w, cup_y+cup_h), (0, 0, 255), 3) 
    
    # Draw Polygon for cup (Red)
    cv2.drawContours(img_result, [largest_cup], -1, (0, 0, 255), 3)
    
    text = f"Cup: {cup_width_mm:.1f} x {cup_height_mm:.1f} mm"
    cv2.putText(img_result, text, (cup_x, cup_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    
    cv2.imwrite(output_path, img_result)
    print(f"\n🎯 Measurement Complete!")
    print(f"Cup Width: {cup_width_mm:.1f} mm | Cup Height: {cup_height_mm:.1f} mm")
    print(f"Image saved at: {output_path}")

# --- RUN IT ---
if __name__ == "__main__":
    # Aap ki test image ka path
    test_image = r'C:\Users\Abbott Computer\Documents\Measure-It-Segmentation\dataset\test\images\IIMG20261104222942.jpg' 
    out_image = r'C:\Users\Abbott Computer\Documents\Measure-It-Segmentation\dataset\test\result_measured.jpg'
    
    if os.path.exists(test_image):
        measure_object(test_image, out_image)
