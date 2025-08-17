import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import os
import random
import segmentation_models_pytorch as smp

# custom dataset
class MultiClassMapDataset(Dataset):
    def __init__(self, image, boundary_mask, text_mask, patch_size, num_samples, transform=None, mode='train', split_ratio=0.8):
        self.image, self.boundary_mask, self.text_mask = image, boundary_mask, text_mask
        self.patch_size, self.num_samples, self.transform, self.mode = patch_size, num_samples, transform, mode
        self.height, self.width, _ = self.image.shape
        self.split_point = int(self.width * split_ratio)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.mode == 'train':
            x = random.randint(0, self.split_point - self.patch_size)
        else:
            x = random.randint(self.split_point, self.width - self.patch_size)
        y = random.randint(0, self.height - self.patch_size)
        image_patch = self.image[y:y + self.patch_size, x:x + self.patch_size]
        boundary_patch = self.boundary_mask[y:y + self.patch_size, x:x + self.patch_size]
        text_patch = self.text_mask[y:y + self.patch_size, x:x + self.patch_size]
        combined_mask = np.zeros((self.patch_size, self.patch_size), dtype=np.int64)
        combined_mask[boundary_patch == 255] = 1
        combined_mask[text_patch == 255] = 2
        if self.transform:
            augmented = self.transform(image=image_patch, mask=combined_mask)
            image_patch, combined_mask = augmented['image'], augmented['mask']
        return image_patch, combined_mask

if __name__ == "__main__":
    # hyperparameters
    LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, PATCH_SIZE, NUM_CLASSES = 1e-4, 4, 50, 256, 3
    NUM_TRAIN_SAMPLES, NUM_VAL_SAMPLES = 250, 50
    PATIENCE = 10
    best_val_loss = float('inf')
    patience_counter = 0

    # paths
    IMAGE_PATH = os.path.join("data", "input", "stockton_1.png")
    BOUNDARY_MASK_PATH = os.path.join("data", "input", "boundaries_mask.png")
    TEXT_MASK_PATH = os.path.join("data", "input", "text_mask.png")
    MODEL_SAVE_PATH = os.path.join("outputs", "models", "tuned_model.pth")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load data
    full_image = cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB)
    full_boundary_mask = cv2.imread(BOUNDARY_MASK_PATH, cv2.IMREAD_GRAYSCALE)
    full_text_mask = cv2.imread(TEXT_MASK_PATH, cv2.IMREAD_GRAYSCALE)

    # transforms
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomBrightnessContrast(p=0.7),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2(),
    ])
    
    # datasets and dataloaders
    train_dataset = MultiClassMapDataset(full_image, full_boundary_mask, full_text_mask, PATCH_SIZE, NUM_TRAIN_SAMPLES, train_transform, mode='train')
    val_dataset = MultiClassMapDataset(full_image, full_boundary_mask, full_text_mask, PATCH_SIZE, NUM_VAL_SAMPLES, val_transform, mode='val')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # model and training setup
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights="imagenet",
        in_channels=3, classes=NUM_CLASSES,
    ).to(device)
    
    class_weights = torch.tensor([0.5, 1.5, 2.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device, dtype=torch.long)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Val loss improved. Saving model to {MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            print(f"Val loss did not improve. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break
            
    print("Finished Training.")