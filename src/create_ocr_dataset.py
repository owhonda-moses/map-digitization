import cv2
import numpy as np
import os

print("Creating OCR dataset")

# paths
IMAGE_PATH = os.path.join("data", "input", "stockton_1.png")
MASK_PATH = os.path.join("data", "input", "text_mask.png")
OUTPUT_DIR = os.path.join("data", "ocr_data")
IMG_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)

# load data
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")
text_mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
if text_mask is None:
    raise FileNotFoundError(f"Text mask not found at {MASK_PATH}")

# dilate mask
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
dilated_mask = cv2.dilate(text_mask, kernel, iterations=1)

# find contours
contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if not contours:
    print("No contours found.")
else:
    print(f"Found {len(contours)} text regions. Saving to '{IMG_OUTPUT_DIR}'...")

# save text regions
for i, cnt in enumerate(contours):
    if cv2.contourArea(cnt) > 20:
        x, y, w, h = cv2.boundingRect(cnt)
        padding = 5
        x_pad, y_pad = max(0, x - padding), max(0, y - padding)
        w_pad, h_pad = w + (padding * 2), h + (padding * 2)
        text_roi = image[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
        file_name = f"image_{i}.png"
        file_path = os.path.join(IMG_OUTPUT_DIR, file_name)
        cv2.imwrite(file_path, text_roi)

print("Dataset images created")