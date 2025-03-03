import os
import cv2
import torch
import numpy as np
from pdf2image import convert_from_path
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

# Function to Convert PDF to Images
def convert_pdf_to_images(pdf_path, output_folder="temp_images"):
    """Converts a PDF file into separate image files for each page."""
    os.makedirs(output_folder, exist_ok=True)
    
    poppler_path = r"C:\poppler-24.08.0\Library\bin"  # Change to your Poppler path
    
    images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)

    image_paths = []
    for i, image in enumerate(images):
        img_path = os.path.join(output_folder, f"page_{i}.png")
        image.save(img_path, "PNG")
        image_paths.append(img_path)

    return image_paths

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

# Process Either an Image or a PDF
def process_document(file_path):
    """Handles both images and PDFs, running segmentation + OCR."""
    
    # Check if the file is a PDF
    if file_path.lower().endswith(".pdf"):
        print("📄 Processing PDF file...")
        image_paths = convert_pdf_to_images(file_path)
    else:
        print("🖼 Processing image file...")
        image_paths = [file_path]

    # Loop through each extracted image (for PDFs, multiple pages)
    for img_path in image_paths:
        print(f"🔍 Processing Page: {img_path}")
        
        # Run text segmentation
        segmented_texts = detect_text_regions(img_path)

        # Apply OCR to each segmented region
        for idx, segment in enumerate(segmented_texts):
            corrected_name, dosage = predict_text(segment)
            print(f"✅ Segmented Text {idx}: {corrected_name}, Dosage: {dosage}")


if __name__ == "__main__":
    test_file = "data/inference_images/file001.pdf"  # Change to your PDF file
    process_document(test_file)