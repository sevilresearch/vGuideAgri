
# Traversability Analysis for Agriculture Automation in Chili Rows

## Overview
These files are used for image segmentation and traversability analysis between chili rows. An output video was split into its frames and labeled for ground truth. These
annotations were used to further train the DeepLabV3_resnet101 pretrained model, which was trained using the COCO 2017 dataset. The Chili row video held 2652 frames 
which were split using a 70%/15%/15% split between train, validate, and test respectively. 

## Repository Contents

**1.** The python files used, placed in the PythonFiles folder.

**2.** The images and ground annotations as they were split for training, validation and testing. These are in the images folder.

  * Of note, the images and their annotations are placed within their respective folders and have the same file name. **Do not change these names.**

Output video that the images were taken from. The video folder holds this and other videos used for analysis.

Segementation Model used. This model was trained on the data in this repository.

## Operation Steps
In order to run this process on your own machine, these prerequisite steps are necessary. 

#### Important Notes:
These python files were created using PyCharm.

When downloading the images, ensure that they are placed in a file path outside of the user folder. This will avoid any permission issues.


#### Steps:
**1.** Download this repository.


**2.** Ensure that when you download this, the python files go into your project and the model and images go into the same folder, which does not have to be your 
   python project. Below is the filepath used for the project:

  Images: C:/Python/PytorchSegmentation/ChiliData/... (Replace ... with the Images folder in this repository)

  Model: C:/Python/PytorchSegmentation/... (Replace ... with the ModelSaves folder in this repository)

  Segmentation: C:/Python/PyTorchSegmentation/... (Create a segmentations folder with the following structure and place it in ...)

    Segmentations/  
        Morph/
        Path/
        Pre/
  
   Others: The video and python files can be placed where ever is convenient for you. The files will use the folder system described above, so if you change these
   locations then you need to update the lines that reference these in each folder.



**3.** (Only do this if you changed the file directory from what was reccomended in step 2) Rewrite the lines in each file labeled below to reflect the paths you chose:

**ChiliDataset:**

Lines 23 and 24:

These two lines involve naming. If you didnt change the names of the folders for the image and annotations, you can disregard. otherwise, this should reflect the names.

**ChiliSegmentationAnalysis:**

Line 194: This should be the location of your model

Line 195: This is an ouput file that the segmentations and path outputs go. layout described above

Line 220: This line is the location of your images.

*Note:* Line 225 changes which folder in the splits is used for analysis. Single is for specific images, test val and train were for assessing and training the model.
