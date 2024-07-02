import cv2
import numpy as np

# Load the image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Use GaussianBlur to reduce noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Apply threshold to get a binary image
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Detect horizontal lines using morphological operations
kernel = np.ones((5, 200), np.uint8)
morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Find line contours
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by top to bottom
contours = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[1])

# Loop through and save the lines as a separate images
for i, cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)
    line_image = image[y:y+h, x:x+w]
    cv2.imwrite(f'{i}.jpg', line_image)

print("Image cut into lines and saved as individual jpg files.")
