# Project Plan: Measure It - Segmentation & Metric Measurement

## 1. Approach & Timeline (8 Hours Work)
* **Hour 1 (Setup & Data):** Camera calibration using 20+ checkerboard images. Dataset collection: 70+ images of a cup (train) and 10+ images of cup + ID Card (test).
* **Hour 2-3 (Annotation):** Labeling cup and ID card segmentation masks using CVAT.
* **Hour 4 (Training):** Training a U-Net (ResNet34) segmentation model on Google Colab.
* **Hour 5-6 (Inference & Measurement):** Writing the end-to-end script to undistort images, find reference object, calculate mm/pixel, and output cup dimensions.
* **Hour 7 (Testing & Accuracy):** Comparing raw vs undistorted images, creating accuracy tables against calliper measurements.
* **Hour 8 (Documentation & Video):** Writing README, Definition of Done, Objectives, and recording the Loom video.

## 2. What I am skipping
* Advanced camera tilt correction (focusing on standard top-down view).
* Training a heavier transformer model (choosing lightweight U-Net due to hardware constraints).