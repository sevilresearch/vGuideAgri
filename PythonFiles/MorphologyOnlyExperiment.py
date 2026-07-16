import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_fill_holes
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.transforms import InterpolationMode

# Import your dataset class the same way your original script imports AirSimData.
# If you make another dataset class, import it here and select it in build_dataset().
from ChiliDataset import ChiliData


# ============================================================
# User Settings
# ============================================================
DATASET_NAME = "ChiliData"
MODELSET_NAME = "ChiliData"

DATASET_PATH = "C:/Python/PyTorchSegmentation/ChiliData/Video 1/"
MODEL_SAVES_PATH = "C:/Python/PyTorchSegmentation/ModelSaves/"
OUTPUT_PATH = r"C:\Python\PyTorchSegmentation\Segmentations\ChiliVid1"

MODEL_FILENAME = "DeeplabV3ChiliData-0-0.963252067565918.pth"

NUM_CLASSES = 4
IMAGE_RESIZE = (256, 512)
DEVICE = torch.device("cpu")

# Class colors used when saving segmentation masks as RGB images.
COLOR_TABLE = np.array([
    (51, 221, 255),  # Sky
    (33, 160, 75),  # Tree
    (102, 255, 102),  # Chili
    (144, 107, 40)  # dirt
], dtype=np.uint8)

# Morphology experiment options.
KERNEL_SIZES = [5, 20, 40, 60]
KERNEL_SHAPES = ["square", "circle"]          # options: "square", "circle"
MORPH_OPERATIONS = ["openclose", "open", "close"]    # options: "none", "open", "close", "openclose"
EROSION_OPTIONS = [False]
IMFILL_OPTIONS = [False]


# ============================================================
# Dataset Setup
# ============================================================
def build_transforms():
    normalize_transform = transforms.Compose([
        transforms.Resize(IMAGE_RESIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    resize_transform = transforms.Compose([
        transforms.Resize(
            IMAGE_RESIZE,
            interpolation=InterpolationMode.NEAREST
        ),
    ])

    return normalize_transform, resize_transform


def build_dataset(dataset_path, normalize_transform, resize_transform):
    """
    Add new datasets here in the same style as AirSimData.

    Example:
        from MyDataset import MyDataset
        if DATASET_NAME == "MyDataset":
            return MyDataset(dataset_path, split="test", ...)
    """
    if DATASET_NAME == "ChiliData":
        return ChiliData(
            dataset_path,
            split="val",
            transform=normalize_transform,
            target_transform=resize_transform
        )

    raise ValueError(f"Unknown dataset name: {DATASET_NAME}")


# ============================================================
# Model Setup
# ============================================================
def load_segmentation_model(model_path, num_classes, device):
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.classifier = DeepLabHead(2048, num_classes)

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model.eval()
    model.to(device)

    return model


# ============================================================
# Morphology Functions
# ============================================================
def get_kernel(kernel_shape, kernel_size):
    if kernel_shape == "square":
        return np.ones((kernel_size, kernel_size), np.uint8)

    if kernel_shape == "circle":
        return cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )

    raise ValueError(f"Invalid kernel shape: {kernel_shape}")


def apply_morphology(
    mask,
    num_classes,
    morph_operation,
    kernel_shape,
    kernel_size,
    apply_erosion,
    apply_imfill
):
    """
    Applies morphology class-by-class so one class does not overwrite or blur
    directly into another class before processing.
    """
    kernel = get_kernel(kernel_shape, kernel_size)
    refined_mask = np.zeros_like(mask)

    for c in range(num_classes):
        class_mask = (mask == c).astype(np.uint8)

        if morph_operation == "none":
            pass

        elif morph_operation == "open":
            class_mask = cv2.morphologyEx(
                class_mask,
                cv2.MORPH_OPEN,
                kernel
            )

        elif morph_operation == "close":
            class_mask = cv2.morphologyEx(
                class_mask,
                cv2.MORPH_CLOSE,
                kernel
            )

        elif morph_operation == "openclose":
            class_mask = cv2.morphologyEx(
                class_mask,
                cv2.MORPH_OPEN,
                kernel
            )
            class_mask = cv2.morphologyEx(
                class_mask,
                cv2.MORPH_CLOSE,
                kernel
            )

        else:
            raise ValueError(f"Invalid morphology operation: {morph_operation}")

        if apply_erosion:
            class_mask = cv2.erode(class_mask, kernel)

        if apply_imfill:
            class_mask = binary_fill_holes(class_mask).astype(np.uint8)

        refined_mask[class_mask == 1] = c

    return refined_mask


def update_iou_counts(pred_mask, target_mask, intersections, unions, num_classes):
    for c in range(num_classes):
        pred_class = pred_mask == c
        target_class = target_mask == c

        intersection = torch.logical_and(target_class, pred_class)
        union = torch.logical_or(target_class, pred_class)

        intersections[c] += torch.count_nonzero(intersection)
        unions[c] += torch.count_nonzero(union)

    return intersections, unions


def safe_iou(intersections, unions):
    return torch.where(
        unions > 0,
        intersections / unions,
        torch.zeros_like(unions)
    )


def save_colored_mask(mask, save_path, color_table):
    rgb_mask = color_table[mask]
    plt.imsave(save_path, rgb_mask)


# ============================================================
# Main Experiment
# ============================================================
def run_morphology_experiment(args):
    pre_folder = os.path.join(args.output_path, "pre")
    morph_folder = os.path.join(args.output_path, "morph")
    os.makedirs(pre_folder, exist_ok=True)
    os.makedirs(morph_folder, exist_ok=True)

    normalize_transform, resize_transform = build_transforms()

    test_dataset = build_dataset(
        args.dataset_path,
        normalize_transform,
        resize_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )

    model = load_segmentation_model(
        args.model_path,
        args.num_classes,
        args.device
    )

    precomputed_data = []

    print("\n====================================")
    print("PRECOMPUTING SEGMENTATIONS")
    print("====================================")

    for i, (test_batch, target_batch) in enumerate(test_loader):
        test_batch = test_batch.to(args.device)

        with torch.no_grad():
            output_batch = model(test_batch)["out"]

        prediction = output_batch.argmax(1)[0].cpu()
        target = target_batch[0].cpu().long()
        raw_mask = prediction.numpy().astype(np.uint8)

        filename = os.path.splitext(test_dataset.imagesList[i])[0]

        save_colored_mask(
            raw_mask,
            os.path.join(pre_folder, filename + "_seg_pre.png"),
            COLOR_TABLE
        )

        precomputed_data.append({
            "prediction": prediction,
            "target": target,
            "raw_mask": raw_mask,
            "filename": filename
        })

        print("Precomputed:", filename)

    results = []

    for kernel_size in KERNEL_SIZES:
        for kernel_shape in KERNEL_SHAPES:
            for morph_operation in MORPH_OPERATIONS:
                for apply_erosion in EROSION_OPTIONS:
                    for apply_imfill in IMFILL_OPTIONS:

                        experiment_name = (
                            f"{morph_operation}_"
                            f"{kernel_shape}_"
                            f"{kernel_size}_"
                            f"E{apply_erosion}_"
                            f"F{apply_imfill}"
                        )

                        experiment_morph_folder = os.path.join(
                            morph_folder,
                            experiment_name
                        )
                        os.makedirs(experiment_morph_folder, exist_ok=True)

                        print("\n====================================")
                        print("STARTING MORPHOLOGY EXPERIMENT")
                        print("====================================")
                        print("Experiment:", experiment_name)

                        raw_intersections = torch.zeros(args.num_classes)
                        raw_unions = torch.zeros(args.num_classes)
                        morph_intersections = torch.zeros(args.num_classes)
                        morph_unions = torch.zeros(args.num_classes)

                        images_processed = 0

                        for data in precomputed_data:
                            prediction = data["prediction"]
                            target = data["target"]
                            raw_mask = data["raw_mask"]
                            filename = data["filename"]

                            raw_intersections, raw_unions = update_iou_counts(
                                prediction,
                                target,
                                raw_intersections,
                                raw_unions,
                                args.num_classes
                            )

                            morph_mask = apply_morphology(
                                raw_mask,
                                args.num_classes,
                                morph_operation,
                                kernel_shape,
                                kernel_size,
                                apply_erosion,
                                apply_imfill
                            )

                            morph_mask_tensor = torch.from_numpy(morph_mask)

                            morph_intersections, morph_unions = update_iou_counts(
                                morph_mask_tensor,
                                target,
                                morph_intersections,
                                morph_unions,
                                args.num_classes
                            )

                            save_colored_mask(
                                morph_mask,
                                os.path.join(
                                    experiment_morph_folder,
                                    filename + "_seg_morph.png"
                                ),
                                COLOR_TABLE
                            )

                            images_processed += 1
                            print("Processed:", images_processed, filename)

                        raw_class_iou = safe_iou(raw_intersections, raw_unions)
                        morph_class_iou = safe_iou(morph_intersections, morph_unions)

                        raw_overall_iou = (
                            torch.sum(raw_intersections) /
                            torch.sum(raw_unions)
                        )

                        morph_overall_iou = (
                            torch.sum(morph_intersections) /
                            torch.sum(morph_unions)
                        )

                        results.append({
                            "Operation": morph_operation,
                            "Kernel Shape": kernel_shape,
                            "Kernel Size": kernel_size,
                            "Erosion": apply_erosion,
                            "Imfill": apply_imfill,
                            "Raw Overall IoU": raw_overall_iou.item(),
                            "Morph Overall IoU": morph_overall_iou.item(),
                            "IoU Improvement": (
                                morph_overall_iou - raw_overall_iou
                            ).item(),
                            "Raw Class IoU": raw_class_iou.tolist(),
                            "Morph Class IoU": morph_class_iou.tolist(),
                            "Images Processed": images_processed
                        })

                        results_table = pd.DataFrame(results)
                        results_table.to_csv(
                            os.path.join(
                                args.output_path,
                                "MorphologyOnlyResults.csv"
                            ),
                            index=False
                        )

                        print("\nEXPERIMENT COMPLETE")
                        print("Raw Overall IoU:", raw_overall_iou.item())
                        print("Morph Overall IoU:", morph_overall_iou.item())
                        print("Improvement:", (morph_overall_iou - raw_overall_iou).item())

    results_table = pd.DataFrame(results)
    print("\n====================================")
    print("FINAL RESULTS")
    print("====================================")
    print(results_table)


# ============================================================
# Command Line Arguments
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run segmentation morphology experiments only."
    )

    parser.add_argument(
        "--dataset-path",
        default=DATASET_PATH,
        help="Path to dataset images folder."
    )

    parser.add_argument(
        "--model-path",
        default=os.path.join(MODEL_SAVES_PATH, MODEL_FILENAME),
        help="Path to trained DeepLabV3 model .pth file."
    )

    parser.add_argument(
        "--output-path",
        default=OUTPUT_PATH,
        help="Folder where pre and morphology outputs will be saved."
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=NUM_CLASSES,
        help="Number of segmentation classes."
    )

    parser.add_argument(
        "--device",
        default=str(DEVICE),
        help="Device to use: cpu or cuda."
    )

    args = parser.parse_args()
    args.device = torch.device(args.device)
    return args


if __name__ == "__main__":
    run_morphology_experiment(parse_args())
