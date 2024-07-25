import numpy as np
import cv2
from PyShearLab import ShearletTransform3D, InverseShearletTransform3D

def embed_watermark(image, watermark, strength=0.05):
    # Resize watermark to match image dimensions
    watermark = cv2.resize(watermark, (image.shape[1], image.shape[0]))
    watermark = watermark.astype(np.float32) / 255

    # Convert image to float32
    image = image.astype(np.float32)

    # Shearlet transform
    st = ShearletTransform3D(image.shape)
    image_shearlet = st.forward(image)

    # Add watermark
    for i in range(len(image_shearlet)):
        image_shearlet[i] += strength * watermark

    # Inverse Shearlet transform
    watermarked_image = st.inverse(image_shearlet)

    # Clip values to valid range and convert to uint8
    watermarked_image = np.clip(watermarked_image, 0, 255)
    watermarked_image = watermarked_image.astype(np.uint8)

    return watermarked_image

def extract_watermark(image, watermarked_image, strength=0.05):
    # Shearlet transform
    st = ShearletTransform3D(image.shape)
    image_shearlet = st.forward(image)
    watermarked_shearlet = st.forward(watermarked_image)

    # Extract watermark
    watermark = (watermarked_shearlet - image_shearlet) / strength

    # Average the extracted watermark
    watermark_avg = np.mean(np.array(watermark), axis=0)

    # Clip values and convert to uint8
    watermark_avg = np.clip(watermark_avg, 0, 1)
    watermark_avg = (watermark_avg * 255).astype(np.uint8)

    return watermark_avg

# Example usage:
# Load images
image = cv2.imread('path_to_image.png', cv2.IMREAD_GRAYSCALE)
watermark = cv2.imread('path_to_watermark.png', cv2.IMREAD_GRAYSCALE)

# Embed watermark
watermarked_image = embed_watermark(image, watermark)

# Save the watermarked image
cv2.imwrite('watermarked_image.png', watermarked_image)

# Extract watermark
extracted_watermark = extract_watermark(image, watermarked_image)

# Save the extracted watermark
cv2.imwrite('extracted_watermark.png', extracted_watermark)
