import os
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

from transformers import (
    SegformerForSemanticSegmentation,
    AutoImageProcessor
)

warnings.filterwarnings("ignore")

# Class mapping (BigEarthNet → contiguous labels)
class_mapping = {
    111: 0, 112: 1, 121: 2, 122: 3, 123: 4, 124: 5,
    131: 6, 132: 7, 133: 8, 141: 9, 142: 10,
    211: 11, 212: 12, 213: 13, 221: 14, 222: 15, 223: 16,
    231: 17, 241: 18, 242: 19, 243: 20, 244: 21,
    311: 22, 312: 23, 313: 24, 321: 25, 322: 26, 323: 27, 324: 28,
    331: 29, 332: 30, 333: 31, 334: 32,
    411: 33, 412: 34, 421: 35, 422: 36, 423: 37,
    511: 38, 512: 39, 521: 40, 522: 41, 523: 42,
    999: 43
}

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

num_classes = 44
batch_size = 32
num_epochs = 100
learning_rate = 1e-4

pretrained_model_name = (
    "/lustre/fsmisc/dataset/HuggingFace_Models/"
    "nvidia/segformer-b0-finetuned-ade-512-512"
)

print("SegFormer lr=1e-4, epochs=100")

# Paths
path_img = "/lustre/fsn1/projects/rech/tnl/uwh84qh/BiasesProject/DataSet/BigEarthNetRGB/"
path_mask = "/lustre/fsn1/projects/rech/tnl/uwh84qh/BiasesProject/DataSet/Reference_Maps/"

parquet_train = pd.read_parquet(
    "/lustre/fsn1/projects/rech/tnl/uwh84qh/BiasesProject/DataSet/parquet/qafinal/imagesname_train.parquet"
)
parquet_val = pd.read_parquet(
    "/lustre/fsn1/projects/rech/tnl/uwh84qh/BiasesProject/DataSet/parquet/qafinal/imagesname_val.parquet"
)

print("train:", len(parquet_train))
print("val:", len(parquet_val))

# Dataset
class CustomSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, parquet_file, processor, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.transform = transform

        self.image_names = parquet_file["img_name"].tolist()
        self.segm_folders = parquet_file["path_to_your_file"].tolist()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        segm_folder = self.segm_folders[idx]

        # Load image
        img_path = os.path.join(self.image_dir, img_name + ".tif")
        image = Image.open(img_path)

        if self.transform is not None:
            image = self.transform(image)

        # SegFormer preprocessing
        encoded = self.processor(image, return_tensors="pt")
        pixel_values = encoded["pixel_values"].squeeze(0)

        # Load mask
        mask_path = os.path.join(
            self.mask_dir,
            segm_folder,
            img_name + "_reference_map.tif"
        )
        mask = np.array(Image.open(mask_path))

        # Map original labels to contiguous ids
        mask_mapped = np.zeros_like(mask, dtype=np.int64)
        for orig, new in class_mapping.items():
            mask_mapped[mask == orig] = new

        mask_tensor = torch.from_numpy(mask_mapped).long()

        return pixel_values, mask_tensor

# Image processor and model
image_processor = AutoImageProcessor.from_pretrained(pretrained_model_name)

model = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name,
    num_labels=num_classes,
    ignore_mismatched_sizes=True
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Optional image transforms (not used by processor)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.1396, 0.1457, 0.1005],
        std=[0.07, 0.0547, 0.0449]
    )
])

# DataLoaders
train_dataset = CustomSegmentationDataset(
    path_img, path_mask, parquet_train, image_processor, transform=None
)
val_dataset = CustomSegmentationDataset(
    path_img, path_mask, parquet_val, image_processor, transform=None
)

train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

print("Dataloaders ready")

# Training loop
output_dir = (
    "/lustre/fsn1/projects/rech/tnl/uwh84qh/"
    "BiasesProject/DataSet/Models/Segmentation/Segform/1/"
)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for images, masks in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images).logits
        outputs = F.interpolate(
            outputs,
            size=(120, 120),
            mode="bilinear",
            align_corners=False
        )

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        del images, masks, outputs, loss
        torch.cuda.empty_cache()

    avg_train_loss = train_loss / len(train_loader)
    print("train loss:", avg_train_loss)

    # Validation 
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images).logits
            outputs = F.interpolate(
                outputs,
                size=(120, 120),
                mode="bilinear",
                align_corners=False
            )

            loss = criterion(outputs, masks)
            val_loss += loss.item()

            del images, masks, outputs, loss
            torch.cuda.empty_cache()

    avg_val_loss = val_loss / len(val_loader)
    print("val loss:", avg_val_loss)

    #Save checkpoint 
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "image_processor": image_processor,
    }

    torch.save(
        checkpoint,
        os.path.join(output_dir, f"_-4_v2{epoch + 1}.pth")
    )

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}\n"
    )
