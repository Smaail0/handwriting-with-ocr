import cv2
import numpy as np

def process_image(image_input):
    """
    Minimal preprocessing: Convert to grayscale, slight denoising, and light contrast enhancement.
    :param image_input: File path (str) or image array (numpy array)
    :return: Processed image as a NumPy array
    """

    # Load image based on input type
    if isinstance(image_input, str):  
        img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"❌ ERROR: Could not read image at {image_input}. Check the file path and format.")
    elif isinstance(image_input, np.ndarray):
        img = image_input  # Already an image array
    else:
        raise TypeError("❌ ERROR: Invalid image input. Must be a file path (str) or a NumPy array.")

    # Ensure the image is valid
    if img is None or img.size == 0:
        raise ValueError("❌ ERROR: Image loading failed. Ensure the file exists and is a valid image.")

    # Convert to grayscale (if not already)
    if len(img.shape) == 3:  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # VERY LIGHT Blur (Optional: Comment this out if not needed)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Contrast enhancement (CLAHE - Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    # **Save preprocessed image for debugging**
    cv2.imwrite("debug_preprocessed.png", img)

    return img
