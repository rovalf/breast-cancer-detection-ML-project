import os
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut
import pandas as pd
import numpy as np

# === CONFIGURATION ===
metadata_path = 'data/mass_case_description_train_set.csv'   # Training CSV
dicom_root = 'data/ddsm_mass_train/manifest-1756284064138/CBIS-DDSM'  # Path to new DICOM folders
output_root = 'dataset/train'  # Existing dataset structure

os.makedirs(os.path.join(output_root, 'benign'), exist_ok=True)
os.makedirs(os.path.join(output_root, 'malignant'), exist_ok=True)

# Read metadata
df = pd.read_csv(metadata_path)
df['folder_name'] = df['image file path'].apply(lambda x: x.split('/')[0].strip())
folder_pathology = df.groupby('folder_name')['pathology'].first().to_dict()

converted = 0
skipped = 0

for folder_name, pathology in folder_pathology.items():
    folder_path = os.path.join(dicom_root, folder_name)
    if not os.path.isdir(folder_path):
        skipped += 1
        continue

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.dcm'):
                dicom_path = os.path.join(root, file)
                try:
                    dcm = pydicom.dcmread(dicom_path)
                    if 'PixelData' not in dcm:
                        skipped += 1
                        continue

                    # Apply VOI LUT if available
                    data = apply_voi_lut(dcm.pixel_array, dcm)

                    # Normalize 0–255
                    data = data.astype(np.float32)
                    data -= np.min(data)
                    data /= np.max(data)
                    data *= 255.0

                    # Convert to grayscale PNG
                    img = Image.fromarray(data.astype(np.uint8)).convert('L')

                    # Build filename and save path
                    rel_subpath = os.path.relpath(root, folder_path)
                    filename = f"{folder_name}_{rel_subpath.replace(os.sep, '_')}_{file}".replace(' ', '_').replace('.dcm', '.png')
                    label_folder = 'malignant' if pathology.upper().startswith("MALIGNANT") else 'benign'
                    save_path = os.path.join(output_root, label_folder, filename)

                    # Avoid overwriting (skip if file already exists)
                    if not os.path.exists(save_path):
                        img.save(save_path)
                        converted += 1
                    else:
                        skipped += 1
                except Exception:
                    skipped += 1
                    continue

print(f"✅ {converted} images saved into '{output_root}/benign' and '{output_root}/malignant'")
print(f"❌ {skipped} files skipped (missing pixel data, errors, or already converted)")
