import os

import cv2
import numpy as np


# Create the output folder if it does not exist
os.makedirs("outputs", exist_ok=True)

# Create a blank image
# Shape: height=400, width=600, color_channels=3
image = np.zeros((400, 600, 3), dtype=np.uint8)

# Add a title
cv2.putText(
    image,
    "Smart Safety Vision",
    (95, 70),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2,
)

# Draw a green rectangle
cv2.rectangle(
    image,
    (150, 120),
    (450, 300),
    (0, 255, 0),
    3,
)

# Add a label
cv2.putText(
    image,
    "Detection Area",
    (190, 220),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2,
)

# Save the image
output_path = "outputs/first_cv_image.png"
cv2.imwrite(output_path, image)

print(f"Image saved to: {output_path}")