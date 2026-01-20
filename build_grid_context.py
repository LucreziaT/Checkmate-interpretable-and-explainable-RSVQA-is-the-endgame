impimport os
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# GRID DEFINITION (4x4)

GRID_MAPPING = {
    (0, 0): "a1", (30, 0): "b1", (60, 0): "c1", (90, 0): "d1",
    (0, 30): "a2", (30, 30): "b2", (60, 30): "c2", (90, 30): "d2",
    (0, 60): "a3", (30, 60): "b3", (60, 60): "c3", (90, 60): "d3",
    (0, 90): "a4", (30, 90): "b4", (60, 90): "c4", (90, 90): "d4"
}

BLOCK_SIZE = 30

# CORINE CLASS MAPPING

CLASS_MAPPING = {
    111: "continuous urban fabrics",
    112: "discontinuous urban fabrics",
    121: "industrial or commercial units",
    122: "road and rail networks and associated land",
    123: "port areas",
    124: "airports",
    131: "mineral extraction sites",
    132: "dump sites",
    133: "construction sites",
    141: "green urban areas",
    142: "sport and leisure facilities",
    211: "non-irrigated arable lands",
    212: "permanently irrigated lands",
    213: "rice fields",
    221: "vineyards",
    222: "fruit trees and berry plantations",
    223: "olive groves",
    231: "pastures",
    241: "annual crops associated with permanent crops",
    242: "complex cultivation patterns",
    243: "land principally occupied by agriculture, with significant areas of natural vegetation",
    244: "agro-forestry areas",
    311: "broad-leaved forests",
    312: "coniferous forests",
    313: "mixed forests",
    321: "natural grasslands",
    322: "moors and heathlands",
    323: "sclerophyllous vegetation",
    324: "transitional woodlands/shrub",
    331: "beaches, dunes, sands",
    332: "bare rocks",
    333: "sparsely vegetated areas",
    334: "burnt areas",
    411: "inland marshes",
    412: "peatbogs",
    421: "salt marshes",
    422: "salines",
    423: "intertidal flats",
    511: "water courses",
    512: "water bodies",
    521: "coastal lagoons",
    522: "estuaries",
    523: "sea and ocean"
}

ORDERED_CLASS_NAMES = list(CLASS_MAPPING.values())

# BLOCK EXTRACTION
def iter_image_blocks(image_array, block_size, grid_mapping):
    """
    Yield (grid_name, image_block) for each grid cell.
    """
    height, width = image_array.shape
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            grid_name = grid_mapping.get((x, y))
            if grid_name is not None:
                yield grid_name, image_array[y:y+block_size, x:x+block_size]

# PIXEL COUNT TABLE CREATION
def create_class_pixel_table(image_path):
    """
    Create a grid x class table counting pixels per CORINE class.
    """
    image = np.array(Image.open(image_path))
    unique_classes = np.unique(image)

    class_translation = {
        cls: CLASS_MAPPING.get(cls, None)
        for cls in unique_classes
    }

    grid_counts = {
        grid: {cls: 0 for cls in ORDERED_CLASS_NAMES}
        for grid in GRID_MAPPING.values()
    }

    for grid_name, block in iter_image_blocks(image, BLOCK_SIZE, GRID_MAPPING):
        for cls in unique_classes:
            class_name = class_translation.get(cls)
            if class_name in ORDERED_CLASS_NAMES:
                grid_counts[grid_name][class_name] += np.sum(block == cls)

    df = pd.DataFrame.from_dict(
        grid_counts,
        orient="index",
        columns=ORDERED_CLASS_NAMES
    )

    df.index.name = "Grid"

    # Thresholding and scaling
    df[df < 30] = 0
    df *= 100

    return df

# MAIN PROCESSING LOOP
INPUT_FOLDER = ".../BiasesProject/DataSet/Predicted_Masks_UNET/" # either UNET, Segformer, DOFA or perfect context segmentation folder results
OUTPUT_FOLDER = ".../BiasesProject/DataSet/UNET_context/" # either UNET, Segformer, DOFA or perfect context table results

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

tif_files = [
    os.path.join(root, file)
    for root, _, files in os.walk(INPUT_FOLDER)
    for file in files
    if file.endswith(".tif")
]

for tif_path in tqdm(tif_files, desc="Processing UNet masks"):
    table = create_class_pixel_table(tif_path)

    image_name = os.path.basename(tif_path)
    output_name = f"{image_name[:-18]}.parquet"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)

    table.to_parquet(output_path, engine="pyarrow")
