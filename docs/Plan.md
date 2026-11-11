## Decision Record
- **Approach taken:** Used U-Net for segmentation and OpenCV Color Thresholding for reference card.
- **Approach rejected:** Used OpenCV for cup detection after U-Net failed to generalize on test images. 
- **Cost of rejection:** Lost time re-training the model. However, OpenCV was 100% reliable on this specific white cup and brown background.