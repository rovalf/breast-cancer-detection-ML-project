import os
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut
import pandas as pd
import numpy as np

# === CONFIGURATION ===
metadata_path = 'data/mass_case_description_test_set.csv'  # Or another CSV for test
dicom_root = 'data/ddsm_mass_train/manifest-1753774741655/CBIS-DDSM'  # Path to test DICOM files
output_root = 'test_images'  # Output folder for test PNGs

os.makedirs(output_root, exist_ok=True)

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

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.dcm'):
                dicom_path = os.path.join(root, file)
                try:
                    dcm = pydicom.dcmread(dicom_path)
                    if 'PixelData' not in dcm:
                        skipped += 1
                        continue
                    # Apply VOI LUT if available for correct contrast
                    data = apply_voi_lut(dcm.pixel_array, dcm)

                    # Normalize to 0–255
                    data = data.astype(np.float32)
                    data -= np.min(data)
                    data /= np.max(data)
                    data *= 255.0

                    # Convert to PIL Image
                    img = Image.fromarray(data.astype(np.uint8)).convert('L')
                    rel_subpath = os.path.relpath(root, folder_path)
                    filename = f"{folder_name}_{rel_subpath.replace(os.sep, '_')}_{file}".replace(' ', '_').replace('.dcm', '.png')
                    label_folder = 'malignant' if pathology.upper().startswith("MALIGNANT") else 'benign'
                    save_path = os.path.join(output_root, f"{label_folder}_{filename}")
                    img.save(save_path)
                    converted += 1
                except Exception:
                    skipped += 1
                    continue

print(f"✅ {converted} test images saved to '{output_root}'")
print(f"❌ {skipped} DICOM files skipped")
