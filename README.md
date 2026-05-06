# 📊 Checkmate and Chessboard

This repository implements a full pipeline for **Remote Sensing Visual Question Answering (RSVQA)** based on:

- Semantic segmentation (UNet, SegFormer, DOFA)
- Grid-based spatial context extraction
- Textual context generation
- Transformer-based VQA training

---

## 📁 Project Structure

├── build_grid_context.py

├── grid_to_text_context.py

├── text_to_vqa_training_context.py

│

├── segmentation/

│ ├── UNET.py

│ ├── SegFormer.py

│ └── DOFA.py

│

├── DataSet/

│ ├── BigEarthNetRGB/

│ ├── Reference_Maps/

│ ├── Predicted_Masks_*/

│ ├── *_context/

│ ├── *_text_context/

│ └── parquet/qafinal/

---

## 🚀 Pipeline Overview

### 1. Semantic Segmentation (`segmentation/`)

Three models are implemented to generate land cover maps.

#### 🔹 UNet (`UNET.py`)
- Encoder: custom ResNet50
- Library: `segmentation_models_pytorch`
- Loss: CrossEntropy
- Preloads dataset in memory (fast but memory intensive)

#### 🔹 SegFormer (`SegFormer.py`)
- Transformer-based segmentation
- Pretrained on ADE20K
- Uses HuggingFace `SegformerForSemanticSegmentation`

#### 🔹 DOFA (`DOFA.py`)
- Transformer backbone for remote sensing
- Uses spectral wavelength input
- Custom decoder for segmentation

#### Output
All models produce:
- Pixel-wise classification maps (`.tif`)
- 44 land cover classes (CORINE-based)

---

## 🧩 2. Grid Context Extraction

Script: `build_grid_context.py`

### Process
- Divide image into a **4×4 grid**
- Count pixels per class per cell
- Apply:
  - Thresholding (`< 30 → 0`)
  - Scaling (`×100`)

### Output
- `.parquet` file per image
- Rows → grid cells (`a1–d4`)
- Columns → land cover classes

---

## 📝 3. Text Context Generation

Script: `grid_to_text_context.py`

### Process
- Load question and corresponding grid table
- Convert non-zero values into tokens:
(class_name, grid_cell): value

- Combine into:
Question: ...; Table: ...

### Output
- `.txt` file per sample

---

## 🤖 4. VQA Model Training

Script: `text_to_vqa_training_context.py`

### Model
- Backbone: **DistilBERT**
- Input:
  - Question
  - Flattened grid context

### Outputs
- Patch prediction (16 grid cells)
- Answer prediction (335 classes)

---

### Key Components

#### Dataset
- Loads:
  - Questions
  - Context tables
  - Labels (patch + answer)

#### Embeddings
- Uses:
  - `patch.json`
  - `answ.json`
- Converted to multi-hot vectors

#### Loss
Loss = a0 * patch_loss + b0 * answer_loss

Default:

a0 = 0.0

b0 = 1.0

---

## ⚙️ Usage

### 1. Train Segmentation Model

python segmentation/UNET.py
python segmentation/SegFormer.py
python segmentation/DOFA.py

---

### 2. Generate Grid Context

python build_grid_context.py

---

### 3. Generate Text Prompts
python grid_to_text_context.py

---

### 4. Train VQA Model
python text_to_vqa_training_context.py

---

## 🗺️ Grid Definition

a1 b1 c1 d1

a2 b2 c2 d2

a3 b3 c3 d3

a4 b4 c4 d4

---

## 🌍 Land Cover Classes

Based on **CORINE Land Cover**, including:

- Urban areas
- Agricultural land
- Forests
- Water bodies
- Wetlands
- Coastal areas

---

## 📌 Notes

- Supports:
  - UNet
  - SegFormer
  - DOFA
  - Ground truth masks
- Designed for:
  - Spatial reasoning
  - Multimodal learning
  - LLM-based VQA

---

## 🔧 Dependencies

pip install torch torchvision transformers

pip install numpy pandas pillow tqdm pyarrow

pip install segmentation-models-pytorch

pip install torchgeo

---

## 📈 Use Case

This framework enables:

- Integration of segmentation and language
- Structured spatial reasoning
- Context-aware RSVQA
- Comparison across segmentation models

---

## 📄 License

Copyright (c) 2026 Tosato Lucrezia 
MIT License

---

## ✉️ Contact

Lucrezia Tosato: ltosato (at) sarmap.ch

---

## ✅ Citation

@article{tosato2025checkmate,
  title={Checkmate: interpretable and explainable RSVQA is the endgame},
  author={Tosato, Lucrezia and Chappuis, Christel Tartini and Montariol, Syrielle and Weissgerber, Flora and Lobry, Sylvain and Tuia, Devis},
  journal={arXiv preprint arXiv:2508.13086},
  year={2025}
}

