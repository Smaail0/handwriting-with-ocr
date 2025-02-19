import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class HandwritingDataset(Dataset):
    def __init__(self, image_folder, label_file, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.samples = []

        # Read labels file
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                img_path, label = line.strip().split("\t", 1)  # Split only on first tab
                self.samples.append((img_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        img_full_path = os.path.normpath(os.path.join(self.image_folder, img_path))

        # Debugging print
        print(f"Trying to open: {img_full_path}")

        # Check if file exists
        if not os.path.exists(img_full_path):
            raise FileNotFoundError(f"Image not found: {img_full_path}")

        # Load image
        img = Image.open(img_full_path).convert("RGB")

        # Apply transformations (this converts to tensor)
        if self.transform:
            img = self.transform(img)
        else:
            raise ValueError("Transformations must be applied to convert PIL images to tensors.")

        return img, label  # ✅ Returns (Tensor, Label)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((384, 384)),  # Resize to match TrOCR input size
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values
])

# Paths to data
IMAGE_FOLDER = "data/images"
LABELS_FILE = "data/labels.txt"

# Create dataset instance
dataset = HandwritingDataset(IMAGE_FOLDER, LABELS_FILE, transform)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Test loading
if __name__ == "__main__":
    for images, labels in dataloader:
        print("Batch Labels:", labels)  # Print first batch of labels
        break