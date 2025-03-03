import os
import cv2
import torch
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from torchvision import transforms
from medication_correction import correct_medication, load_medication_database
from preprocessing import process_image  # Use the fixed preprocessing function
from image_segmentation import detect_text_regions

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
def predict_text(image_array):
    
    """Processes a segmented text region (NumPy array) and predicts text using OCR."""
    
    # Convert OpenCV image (NumPy array) to PIL format
    image_pil = Image.fromarray(image_array).convert("RGB")

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

# Load Image and Perform OCR
image_path = "data/inference_images/img000.png"
segmented_texts = detect_text_regions(image_path)

for idx, segment in enumerate(segmented_texts):
    corrected_name, dosage = predict_text(segment)
    print(f"✅ Segmented Text {idx}: {corrected_name}, Dosage: {dosage}")