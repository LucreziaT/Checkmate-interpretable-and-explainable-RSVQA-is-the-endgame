import os
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

import torchvision.transforms as transforms
from PIL import Image

import numpy as np
import pandas as pd
from tqdm import tqdm

from torchgeo.models import dofa_base_patch16_224
warnings.filterwarnings("ignore")

# Global configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
NUM_EPOCHS = 100
NUM_CLASSES = 44
LEARNING_RATE = 1e-6
OUTPUT_SIZE = (120, 120)

# RGB wavelengths (DOFA needs them explicitly)
WAVELIST = [0.665, 0.56, 0.49] ### For sar [5.405, 5.405] 

# Paths
PATH_IMG = ".../BiasesProject/DataSet/BigEarthNetRGB/"
PATH_MASK = ".../BiasesProject/DataSet/Reference_Maps/"

PARQUET_TRAIN = ".../BiasesProject/DataSet/parquet/qafinal/imagesname_train.parquet"
PARQUET_VAL   = ".../BiasesProject/DataSet/parquet/qafinal/imagesname_val.parquet"

OUTPUT_DIR = ".../BiasesProject/DataSet/Models/Segmentation/DOFA/1/"

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

# Load parquet files
parquet_train = pd.read_parquet(PARQUET_TRAIN)
parquet_val   = pd.read_parquet(PARQUET_VAL)

print("Train samples:", len(parquet_train))
print("Val samples:", len(parquet_val))

# Dataset
class SegmentationDataset(data.Dataset):
    def __init__(self, image_dir, mask_dir, parquet_file, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_names = parquet_file["img_name"].tolist()
        self.mask_folders = parquet_file["path_to_your_file"].tolist()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        mask_folder = self.mask_folders[idx]

        img_path = os.path.join(self.image_dir, img_name + ".tif")
        mask_path = os.path.join(
            self.mask_dir,
            mask_folder,
            img_name + "_reference_map.tif"
        )

        # Image
        image = Image.open(img_path)
        if self.transform is not None:
            image = self.transform(image)

        # Mask (remap original labels)
        mask = np.array(Image.open(mask_path))
        mapped_mask = np.zeros_like(mask, dtype=np.int64)

        for orig, new in class_mapping.items():
            mapped_mask[mask == orig] = new

        mask_tensor = torch.from_numpy(mapped_mask).long()

        return image, mask_tensor

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.1396, 0.1457, 0.1005],
        std=[0.07,   0.0547, 0.0449]
    ),
])

# DataLoaders
train_dataset = SegmentationDataset(PATH_IMG, PATH_MASK, parquet_train, transform)
val_dataset   = SegmentationDataset(PATH_IMG, PATH_MASK, parquet_val, transform)

train_loader = data.DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
val_loader   = data.DataLoader(val_dataset, BATCH_SIZE, shuffle=False)

# DOFA backbone setup
backbone = dofa_base_patch16_224(weights=None).to(DEVICE)

weights_path = ".../BiasesProject/DataSet/Models/Segmentation/DOFA/dofa_base_patch16_224-a0275954.pth"
state_dict = torch.load(weights_path, map_location=DEVICE)
backbone.load_state_dict(state_dict, strict=False)

# Remove classification head
backbone.head = nn.Identity()

# Override forward to extract intermediate feature maps
def forward_features(self, x, wave_list):
    x, _ = self.patch_embed(x, wave_list)

    features = []
    for i, blk in enumerate(self.blocks):
        x = blk(x)
        if i in [2, 5, 8, 11]:
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            fmap = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            features.append(fmap)

    return features

backbone.forward = forward_features.__get__(backbone)

# Segmentation model
class DOFASegmentationModel(nn.Module):
    def __init__(self, backbone, num_classes, output_size):
        super().__init__()
        self.backbone = backbone
        self.output_size = output_size

        self.decoder = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x, wavelengths):
        features = self.backbone(x, wavelengths)
        logits = self.decoder(features[-1])
        logits = F.interpolate(
            logits,
            size=self.output_size,
            mode="bilinear",
            align_corners=False
        )
        return logits

model = DOFASegmentationModel(backbone, NUM_CLASSES, OUTPUT_SIZE).to(DEVICE)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1} - Train"):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        wave_tensor = torch.tensor(WAVELIST, device=DEVICE)
        outputs = model(images, wave_tensor)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        del images, masks, outputs, loss
        torch.cuda.empty_cache()

    train_loss /= len(train_loader)
    print("Train loss:", train_loss)

    # Validation
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1} - Val"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            wave_tensor = torch.tensor(WAVELIST, device=DEVICE)
            outputs = model(images, wave_tensor)

            loss = criterion(outputs, masks)
            val_loss += loss.item()

            del images, masks, outputs, loss
            torch.cuda.empty_cache()

    val_loss /= len(val_loader)
    print("Val loss:", val_loss)

    # Save checkpoint
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }

    torch.save(
        checkpoint,
        os.path.join(OUTPUT_DIR, f"_-6all_{epoch+1}.pth")
    )

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] done\n")
