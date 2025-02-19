import os
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.utils.data import DataLoader
from dataset import HandwritingDataset  # Import your dataset
from torchvision import transforms
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

# ✅ Step 1: Load Pretrained TrOCR Model & Processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id # Ensure it matches the tokenizer

# ✅ Step 2: Move Model to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ Step 3: Define Transformations (Ensure images match model input size)
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ✅ Step 4: Load Dataset
IMAGE_FOLDER = "data/images"
LABELS_FILE = "data/labels.txt"
dataset = HandwritingDataset(IMAGE_FOLDER, LABELS_FILE, transform)

# ✅ Step 5: Create DataLoader
batch_size = 4
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ✅ Step 6: Define Optimizer & Loss Function
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = CrossEntropyLoss()

# ✅ Step 7: Training Loop
num_epochs = 3  # Adjust as needed
model.train()

for epoch in range(num_epochs):
    total_loss = 0

    for batch in dataloader:
        images, labels = batch
        images = images.to(device)

        # Convert labels into tokenized format
        inputs = processor(text=labels, return_tensors="pt", padding=True, truncation=True).input_ids
        inputs = inputs.to(device)

        # Forward pass
        outputs = model(pixel_values=images, labels=inputs)
        loss = outputs.loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# ✅ Step 8: Save Fine-Tuned Model
model.save_pretrained("../models/trocr_finetuned")
processor.save_pretrained("../models/trocr_finetuned")

print("Training complete! Model saved to '../models/trocr_finetuned'")
