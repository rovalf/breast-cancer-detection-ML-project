# find_and_fix_bad_pngs.py
import os
from PIL import Image, ImageFile

# Do not allow truncated images — be strict
ImageFile.LOAD_TRUNCATED_IMAGES = False

ROOT = "dataset/train"
bad_files = []

# =============================================================================
# Scan all images under dataset/train/{benign, malignant}
# =============================================================================
for sub in ("benign", "malignant"):
    dir_path = os.path.join(ROOT, sub)
    if not os.path.isdir(dir_path):
        continue
    for fn in os.listdir(dir_path):
        if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        fpath = os.path.join(dir_path, fn)
        try:
            # First pass: verify header integrity
            with Image.open(fpath) as im:
                im.verify()
            # Second pass: fully load pixel data
            with Image.open(fpath) as im:
                im.load()
        except Exception:
            bad_files.append(fpath)

# =============================================================================
# Remove corrupted images
# =============================================================================
print(f"Found {len(bad_files)} corrupted images.")
for fpath in bad_files:
    try:
        os.remove(fpath)
        print("Removed:", fpath)
    except Exception:
        print("Failed to remove:", fpath)
