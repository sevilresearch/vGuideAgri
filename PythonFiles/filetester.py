import os
import cv2

IMAGE_DIR = r"C:\Python\PyTorchSegmentation\ChiliData\video 3\Images\Images\temps"
MASK_DIR = r"C:\Python\PyTorchSegmentation\ChiliData\video 3\Images\Annotations\temp"

TRAIN_LIST = r"C:\Users\maste\PycharmProjects\ChiliAnalysis\train.txt"
VAL_LIST = r"C:\Users\maste\PycharmProjects\ChiliAnalysis\val.txt"

all_files = []

for txt in [TRAIN_LIST, VAL_LIST]:
    with open(txt, "r", encoding="utf-8-sig") as f:
        all_files.extend([x.strip() for x in f if x.strip()])

print(f"Checking {len(all_files)} files")

errors = 0

for name in all_files:

    img_path = os.path.join(IMAGE_DIR, name)
    mask_path = os.path.join(MASK_DIR, name)

    if not os.path.exists(img_path):
        print("MISSING IMAGE:", img_path)
        errors += 1
        continue

    if not os.path.exists(mask_path):
        print("MISSING MASK:", mask_path)
        errors += 1
        continue

    img = cv2.imread(img_path)

    if img is None:
        print("BAD IMAGE:", img_path)
        errors += 1

    mask = cv2.imread(mask_path, 0)

    if mask is None:
        print("BAD MASK:", mask_path)
        errors += 1

print()
print("Errors:", errors)