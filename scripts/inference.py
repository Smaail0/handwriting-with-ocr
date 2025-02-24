import os
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from torchvision import transforms
from medication_correction import correct_medication, load_medication_database

#load medication database
medication_df = load_medication_database()

#Step 1: Load the Fine-Tuned Model
MODEL_PATH = "../models/trocr_finetuned"
processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)

#Move Model to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#Step 2: Define Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((384, 384)),  # Resize image to match TrOCR input size
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values
])

#Step 3: Function to Predict Text from an Image
def predict_text(image_path):
    abs_path = os.path.abspath(image_path)  # Get absolute path
    print(f"Checking Image at: {abs_path}")  # Print the actual path

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Image not found: {abs_path}")

    print(f"Processing: {abs_path}")
    
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Generate prediction
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(pixel_values=image)

    # Decode output tokens into text
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Correct the predicted text using the medication database
    corrected_text, dosage = correct_medication(predicted_text, medication_df)
    
    return predicted_text, dosage

#Step 4: Test the Model on a New Image
if __name__ == "__main__":
    test_image = "data/inference_images/img019.jpg"
    result = predict_text(test_image)
    print(f"Predicted Text: {result}")
