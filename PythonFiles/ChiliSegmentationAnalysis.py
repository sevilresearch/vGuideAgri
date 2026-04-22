
import torch
from torch.utils.data import DataLoader
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import heapq
import os
from scipy.ndimage import binary_fill_holes

from ChiliDataset import ChiliData




# ================================
# Morphology Configuration
# ================================

MORPH_OPERATION = "close"
KERNEL_SHAPE = "circle"
KERNEL_SIZE = 5

APPLY_EROSION = False
APPLY_IMFILL = False


# ================================
# Safety Margin
# ================================

SAFETY_KERNEL_SIZE = 7


# ================================
# Kernel Builder
# ================================

def get_kernel():

    if KERNEL_SHAPE == "square":
        return np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)

    elif KERNEL_SHAPE == "circle":
        return cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (KERNEL_SIZE, KERNEL_SIZE)
        )


# ================================
# Safe Class-wise Morphology
# ================================

def apply_morphology(mask):

# Applies morph operations using the parameters defined above
    kernel = get_kernel()

    refinedMask = np.zeros_like(mask)

    for c in range(numClasses):

        classMask = (mask == c).astype(np.uint8)

        if MORPH_OPERATION == "open":
            classMask = cv2.morphologyEx(classMask, cv2.MORPH_OPEN, kernel)

        if APPLY_EROSION:
            classMask = cv2.erode(classMask, kernel)

        if APPLY_IMFILL:
            classMask = binary_fill_holes(classMask).astype(np.uint8)

        refinedMask[classMask == 1] = c

    return refinedMask


# ================================
# A* Path Planner
# ================================

def astar(cost, start, goal):
# Pathing calculation
    h, w = cost.shape
    visited = np.zeros((h,w), dtype=bool)

    pq = []
    heapq.heappush(pq,(0,start))

    parent = {}
    gscore = {start:0}

    directions = [(1,0),(-1,0),(0,1),(0,-1)]

    while pq:

        _, current = heapq.heappop(pq)

        if current == goal:
            break

        if visited[current]:
            continue

        visited[current] = True

        for dy,dx in directions:

            ny = current[0] + dy
            nx = current[1] + dx

            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue

            new_cost = gscore[current] + cost[ny,nx]

            if (ny,nx) not in gscore or new_cost < gscore[(ny,nx)]:

                gscore[(ny,nx)] = new_cost
                priority = new_cost + abs(goal[0]-ny) + abs(goal[1]-nx)

                heapq.heappush(pq,(priority,(ny,nx)))
                parent[(ny,nx)] = current

    path = []
    node = goal

    while node in parent:
        path.append(node)
        node = parent[node]

    return path


# ================================
# Furthest Reachable Pixel
# ================================

def find_furthest_reachable(traversabilityImage, start):

    h, w = traversabilityImage.shape
    visited = np.zeros((h,w), dtype=bool)

    queue = [start]
    visited[start] = True

    furthest = start

    directions = [(1,0),(-1,0),(0,1),(0,-1)]

    while queue:

        y,x = queue.pop(0)

        if y < furthest[0]:
            furthest = (y,x)

        for dy,dx in directions:

            ny = y + dy
            nx = x + dx

            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue

            if visited[ny,nx]:
                continue

            if traversabilityImage[ny,nx] == 0:
                continue

            visited[ny,nx] = True
            queue.append((ny,nx))

    return furthest


 # Start of code
# ================================
# Dataset Setup
# ================================

dataset = "ChiliData"
modelset = "ChiliData"

modelSavesPath = "C:/Python/PyTorchSegmentation/ModelSaves/"
segmentationsPath = "C:/Python/PyTorchSegmentation/Segmentations/"

preFolder = os.path.join(segmentationsPath,"pre")
morphFolder = os.path.join(segmentationsPath,"morph")
pathFolder = os.path.join(segmentationsPath,"path")

os.makedirs(preFolder,exist_ok=True)
os.makedirs(morphFolder,exist_ok=True)
os.makedirs(pathFolder,exist_ok=True)

imageResize = (256,512)

device = torch.device("cpu") # Define device

normalizeTransform = transforms.Compose([
    transforms.Resize(imageResize),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# path definitions for image and model locations

datasetPath = "C:/Python/PyTorchSegmentation/ChiliData/Video 1"
numClasses = 4

testDataset = ChiliData(
    datasetPath,
    split="single",
    transform=normalizeTransform
)

testLoader = DataLoader(
    testDataset,
    batch_size=1,
    shuffle=False
)


# ================================
# Load Model
# ================================

segmentationModel = models.segmentation.deeplabv3_resnet101(pretrained=True)
segmentationModel.classifier = DeepLabHead(2048, numClasses)

segmentationModel.load_state_dict(
    torch.load(
        modelSavesPath +
        "DeeplabV3" +
        modelset +
        "-0-0.963252067565918.pth",
        map_location=device
    )
)

segmentationModel.eval()
segmentationModel.to(device)


# ================================
# Color Lookup Table
# ================================

color_table = np.array([
    [51,221,255],
    [33,160,75],
    [102,255,102],
    [144,107,40]
], dtype=np.uint8)


# ================================
# Traversability Mapping
# ================================

traversabilityLookupTable = [
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,255]
]

for _ in range(256 - len(traversabilityLookupTable)):
    traversabilityLookupTable.append([0,0,0])

traversabilityLookupTable = np.array([traversabilityLookupTable], dtype=np.uint8)


# ================================
# IoU Counters
# ================================

rawIntersectionCounts = torch.zeros(numClasses)
rawUnionCounts = torch.zeros(numClasses)

morphIntersectionCounts = torch.zeros(numClasses)
morphUnionCounts = torch.zeros(numClasses)


# ================================
# Path Metrics
# ================================

cumulativePathingCalculationTime = 0
totalPathLength = 0
totalNumPaths = 0

overallStart = time.time()
imagesTested = 0


# ================================
# Evaluation Loop
# ================================

for i,(testBatch,targetBatch) in enumerate(testLoader):

    individualStart = time.time()

    testBatch = testBatch.to(device)

    with torch.no_grad():
        outputBatch = segmentationModel(testBatch)["out"]

    prediction = outputBatch.argmax(1)[0].cpu()
    target = targetBatch[0].cpu().long()

    rawMask = prediction.numpy()
    filename = os.path.splitext(testDataset.imagesList[i])[0]

    # PRE SEGMENTATION SAVE
    rgbRawMask = color_table[rawMask]
    plt.imsave(os.path.join(preFolder,filename+"_seg_pre.png"),rgbRawMask)


    # IoU BEFORE MORPH
    for c in range(numClasses):

        targetMask = (target == c)
        predMask = (prediction == c)

        inter = torch.logical_and(targetMask,predMask)
        union = torch.logical_or(targetMask,predMask)

        rawIntersectionCounts[c]+=torch.count_nonzero(inter)
        rawUnionCounts[c]+=torch.count_nonzero(union)


    # MORPHOLOGY
    morphMask = apply_morphology(rawMask.astype(np.uint8))
    morphMaskTensor = torch.from_numpy(morphMask)


    # POST MORPH SAVE
    rgbMorphMask = color_table[morphMask]
    plt.imsave(os.path.join(morphFolder,filename+"_seg_morph.png"),rgbMorphMask)


    # IoU AFTER MORPH
    for c in range(numClasses):

        targetMask=(target==c)
        predMask=(morphMaskTensor==c)

        inter=torch.logical_and(targetMask,predMask)
        union=torch.logical_or(targetMask,predMask)

        morphIntersectionCounts[c]+=torch.count_nonzero(inter)
        morphUnionCounts[c]+=torch.count_nonzero(union)


    # TRAVERSABILITY
    segRGB=cv2.cvtColor(morphMask.astype(np.uint8),cv2.COLOR_GRAY2RGB)

    traversabilityImage=cv2.LUT(segRGB,traversabilityLookupTable)
    traversabilityImage=cv2.cvtColor(traversabilityImage,cv2.COLOR_RGB2GRAY)
    traversabilityImage=cv2.threshold(traversabilityImage,1,1,cv2.THRESH_BINARY)[1]


    # SAFETY MARGIN
    kernel=np.ones((SAFETY_KERNEL_SIZE,SAFETY_KERNEL_SIZE),np.uint8)
    traversabilityImage=cv2.erode(traversabilityImage,kernel)


    # =========================
    # SAFE REGION COMPUTATION (MOVED UP)
    # =========================

    h,w=traversabilityImage.shape
    center=w//2

    # TEMP START (for reachability flood fill)
    temp_start=None
    for r in range(h-1,-1,-1):
        if traversabilityImage[r,center]==1:
            temp_start=(r,center)
            break

    if temp_start is None:
        continue

    # FLOOD FILL TO FIND CONNECTED REGION
    visited=np.zeros((h,w),dtype=bool)
    queue=[temp_start]
    visited[temp_start]=True

    directions=[(1,0),(-1,0),(0,1),(0,-1)]

    while queue:
        y,x=queue.pop(0)

        for dy,dx in directions:
            ny=y+dy
            nx=x+dx

            if ny<0 or ny>=h or nx<0 or nx>=w:
                continue

            if visited[ny,nx]:
                continue

            if traversabilityImage[ny,nx]==0:
                continue

            visited[ny,nx]=True
            queue.append((ny,nx))

    # ERODE TO CREATE SAFE (VIABLE) REGION
    marginKernel=np.ones((15,15),np.uint8)

    viableMask=visited.astype(np.uint8)
    viableMask=cv2.erode(viableMask,marginKernel)


    # =========================
    # START (NOW USING SAFE REGION)
    # =========================

    start=None
    for r in range(h-1,-1,-1):
        if viableMask[r,center]==1:
            start=(r,center)
            break

    if start is None:
        continue


    # =========================
    # GOAL (SAFE REGION)
    # =========================

    goal=find_furthest_reachable(viableMask,start)


    # =========================
    # COST MAP (SAFE REGION ONLY)
    # =========================

    costMap=np.where(viableMask==1,1,1000)


    # =========================
    # A*
    # =========================

    pathingStart=time.time()
    path=astar(costMap,start,goal)
    individualPathTime=time.time()-pathingStart


    # =========================
    # VISUALIZATION
    # =========================

    # IMAGE 1 (UNCHANGED)
    allTraversable=np.zeros((h,w,3),dtype=np.uint8)
    allTraversable[traversabilityImage==1]=[255,255,0]

    plt.imsave(os.path.join(pathFolder,filename+"_all_traversable.png"),allTraversable)


    # IMAGE 2 (VIABLE MAP)
    viableMap=np.zeros((h,w,3),dtype=np.uint8)

    # viable = yellow
    viableMap[viableMask==1]=[255,255,0]

    # removed edges = green
    mask=(traversabilityImage==1)&(viableMask==0)
    viableMap[mask]=[0,200,0]

    plt.imsave(os.path.join(pathFolder,filename+"_viable_traversable.png"),viableMap)


    # IMAGE 3 (FINAL PATH — SAFE REGION ONLY)
    pathMap=np.zeros((h,w,3),dtype=np.uint8)

    pathMap[viableMask==1]=[0,200,0]

    for y,x in path:
        pathMap[y,x]=[255,255,0]

    plt.imsave(os.path.join(pathFolder,filename+"_path_map.png"),pathMap)
    # ================================
    # Final Metrics
    # ================================

    rawClassIoU = rawIntersectionCounts / rawUnionCounts
    morphClassIoU = morphIntersectionCounts / morphUnionCounts

    rawIoU = torch.sum(rawIntersectionCounts) / torch.sum(rawUnionCounts)
    morphIoU = torch.sum(morphIntersectionCounts) / torch.sum(morphUnionCounts)

    print("\nRaw IoU per class:", rawClassIoU.tolist())
    print("Morph IoU per class:", morphClassIoU.tolist())

    print("\nRaw Overall IoU:", rawIoU.item())
    print("Morphology Overall IoU:", morphIoU.item())

    print("Improvement:", (morphIoU - rawIoU).item())

    individualTime=time.time()-individualStart

    print("Image:",filename)
    print("Path Length:",len(path))
    print("Path Time:",individualPathTime)
    print("Total Image Time:",individualTime)
    print()

    imagesTested+=1

# ================================
# FINAL DATASET STATISTICS
# ================================

overallTime = time.time() - overallStart

# IoU calculations
rawClassIoU = rawIntersectionCounts / rawUnionCounts
morphClassIoU = morphIntersectionCounts / morphUnionCounts

rawIoU = torch.sum(rawIntersectionCounts) / torch.sum(rawUnionCounts)
morphIoU = torch.sum(morphIntersectionCounts) / torch.sum(morphUnionCounts)

# Path statistics
if totalNumPaths != 0:
    AvgPathLength = totalPathLength / totalNumPaths
else:
    AvgPathLength = 0

if imagesTested != 0:
    AvgRuntimePerImage = overallTime / imagesTested
    AvgPathTimePerImage = cumulativePathingCalculationTime / imagesTested
else:
    AvgRuntimePerImage = 0
    AvgPathTimePerImage = 0


print("\n===============================")
print("DATASET SUMMARY STATISTICS")
print("===============================")

print("\nSegmentation Performance:")
print("Raw IoU per class:", rawClassIoU.tolist())
print("Morph IoU per class:", morphClassIoU.tolist())

print("\nRaw Overall IoU:", rawIoU.item())
print("Morphology Overall IoU:", morphIoU.item())
print("IoU Improvement:", (morphIoU - rawIoU).item())

print("\nPath Planning Performance:")
print("Total Runtime (entire dataset):", overallTime)
print("Average Runtime per Image:", AvgRuntimePerImage)

print("Total Pathing Time:", cumulativePathingCalculationTime)
print("Average Path Time per Image:", AvgPathTimePerImage)

print("Average Path Length:", AvgPathLength)
print("Images Processed:", imagesTested)