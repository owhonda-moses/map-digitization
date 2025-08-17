import cv2
import numpy as np
from skimage import color
import os

print("Creating draft mask")

# paths
IMAGE_PATH = os.path.join("data", "input", "stockton_1.png")
OUTPUT_PATH = os.path.join("data", "input", "draft_mask.png")

# load image
image_bgr = cv2.imread(IMAGE_PATH)
if image_bgr is None:
    raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# hsv mask
hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
m1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
m2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
red_hsv = cv2.bitwise_or(m1, m2) > 0

# l*a*b* mask
lab = color.rgb2lab(image_rgb / 255.0)
a_channel = lab[..., 1]
a_thr = np.percentile(a_channel, 99.2)
red_lab = (a_channel > a_thr)

# combine and clean
red_combined_bool = (red_hsv | red_lab)
red_combined_int = red_combined_bool.astype(np.uint8) * 255
kernel = np.ones((5, 5), np.uint8)
final_mask = cv2.morphologyEx(red_combined_int, cv2.MORPH_CLOSE, kernel)

# save draft mask
cv2.imwrite(OUTPUT_PATH, final_mask)
print(f"Draft mask saved to '{OUTPUT_PATH}'")