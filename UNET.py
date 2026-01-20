import os
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.models as models

from segmentation_models_pytorch import Unet
from PIL import Image

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Class mapping (BigEarthNet labels → contiguous ids)
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

print("UNet + ResNet50 encoder | lr=1e-3")

# Paths and data
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

# Model configuration
num_classes = 44
batch_size = 32
num_epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom ResNet50 encoder (feature extractor only)
class CustomResNet50Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(pretrained=False)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])

    def forward(self, x):
        return self.encoder(x)

encoder = CustomResNet50Encoder()

# UNet with custom encoder
model = Unet(
    encoder=encoder,
    in_channels=3,
    classes=num_classes
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Dataset
class SegmentationDataset(data.Dataset):
    def __init__(self, image_dir, mask_dir, parquet_file, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_names = parquet_file["img_name"].tolist()
        self.segm_folders = parquet_file["path_to_your_file"].tolist()

        # preload everything into memory
        self.images = []
        self.masks = []

        for img_name, folder in zip(self.image_names, self.segm_folders):
            img_path = os.path.join(image_dir, img_name + ".tif")
            mask_path = os.path.join(
                mask_dir, folder, img_name + "_reference_map.tif"
            )

            image = Image.open(img_path)
            if self.transform is not None:
                image = self.transform(image)
            self.images.append(image)

            mask = np.array(Image.open(mask_path))
            mapped_mask = np.zeros_like(mask, dtype=np.int64)
            for orig, new in class_mapping.items():
                mapped_mask[mask == orig] = new

            self.masks.append(torch.from_numpy(mapped_mask).long())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]

# Transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# DataLoaders
train_dataset = SegmentationDataset(
    path_img, path_mask, parquet_train, transform
)
val_dataset = SegmentationDataset(
    path_img, path_mask, parquet_val, transform
)

train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size, shuffle=False)

# Training loop
output_dir = (
    "/lustre/fsn1/projects/rech/tnl/uwh84qh/"
    "BiasesProject/DataSet/Models/Segmentation/UNET/1/"
)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for images, masks in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)
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

            outputs = model(images)
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

    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
    }

    torch.save(
        checkpoint,
        os.path.join(output_dir, f"_-3{epoch + 1}.pth")
    )

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}\n"
    )
