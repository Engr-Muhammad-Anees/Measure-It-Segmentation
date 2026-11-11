# Objectives
**MO-1: Measurement Accuracy**
- What is measured: Mean Absolute Error (MAE) in millimeters.
- How: Compare system output with ground truth from ruler for 10 test images.
- Target: MAE < 5 mm.
- Conditions: Top-down view, indoor lighting, reference card in frame.
- Not included: Blurry images, cards partially outside the frame.

**MO-2: Inference Time**
- What is measured: Time taken from image input to mm output.
- How: Stopwatch measurement for 5 runs.
- Target: < 2 seconds per image.
