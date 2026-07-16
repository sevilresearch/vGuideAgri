import os
import random

# === CONFIG ===
folder_path = r"C:\Python\PyTorchSegmentation\ChiliData\video 3\Images\Annotations\temp"  # <-- CHANGE THIS
output_subset = "val.txt"
output_remaining = "train.txt"
k = 68  # number of files to select

# Optional: make results reproducible
# random.seed(42)

# === GET FILES ===
files = [
    f for f in os.listdir(folder_path)
    if f.startswith("frame") and f.endswith(".png")
]

if len(files) < k:
    raise ValueError(f"Not enough files: found {len(files)}, need {k}")

# === SORT NUMERICALLY ===
files.sort(key=lambda x: int(x[5:10]))

n = len(files)

# === STRATIFIED SAMPLING ===
selected = []

for i in range(k):
    start = int(i * n / k)
    end = int((i + 1) * n / k)

    # Safety check (shouldn't happen, but just in case)
    if start >= end:
        continue

    chosen = random.choice(files[start:end])
    selected.append(chosen)

selected_set = set(selected)

# === REMAINING FILES ===
remaining = [f for f in files if f not in selected_set]

# === WRITE OUTPUT FILES ===
with open(output_subset, "w") as f:
    for name in selected:
        f.write(name + "\n")

with open(output_remaining, "w") as f:
    for name in remaining:
        f.write(name + "\n")

# === SUMMARY ===
print(f"Total files found: {n}")
print(f"Selected: {len(selected)}")
print(f"Remaining: {len(remaining)}")
print("Done!")