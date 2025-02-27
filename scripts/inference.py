import os
import cv2
import torch
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from torchvision import transforms
from medication_correction import correct_medication, load_medication_database
from preprocessing import process_image  # Use the fixed preprocessing function

# Load the Fine-Tuned Model
MODEL_PATH = "../models/trocr_finetuned"
processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)

# Move Model to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load Medication Database
medication_df = load_medication_database()

# Image Preprocessing for OCR
transform = transforms.Compose([
    transforms.Resize((384, 384)),  
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,))  
])

# Function to Predict Text from an Image
def predict_text(image_path):
    abs_path = os.path.abspath(image_path)
    print(f"🔍 Processing Image: {abs_path}")

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"❌ ERROR: Image not found at {abs_path}")

    # Load the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"❌ ERROR: Could not load image at {image_path}. Ensure the file exists and is a valid image format.")

    # Preprocess the image
    preprocessed_image = process_image(image)  # Now correctly passing a NumPy array

    # Convert to PIL format
    image_pil = Image.fromarray(preprocessed_image).convert("RGB")

    # Apply OCR model transformations
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    # Generate OCR prediction
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(pixel_values=image_tensor)

    # Decode output tokens into text
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Medication Correction
    corrected_name, dosage = correct_medication(predicted_text, medication_df)

    return corrected_name, dosage

# Test the Model on a New Image
if __name__ == "__main__":
    test_image = "data/inference_images/img041.png"
    corrected_name, dosage = predict_text(test_image)
    print(f"✅ Predicted Medication: {corrected_name}, Dosage: {dosage}")