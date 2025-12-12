import os
import SimpleITK as sitk
import scipy.ndimage as nd
import numpy as np
import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)
import itk
itk.ProcessObject.SetGlobalWarningDisplay(False)
from monai.transforms import (
    LoadImaged,
    Compose,
    CropForegroundd,
    Resized,
    SpatialPadd,
    EnsureChannelFirstd,
    Spacingd,
    Lambda,
    ScaleIntensityRanged,
)

def reorder(data):
    data["image"] = np.transpose(data["image"], (0, 3, 2, 1))
    return data

def get_training_transform_pipeline() -> Compose:
    """
    Returns the training transformation pipeline using MONAI transforms.
    The pipeline explanation is included in the comments below. Generally,
    it takes a dictionary with key "image" and applies a series of transformations
    to prepare the data for training.
    
    Example Input:
    ```
    {"image": <image_data> }
    ```
    Example Output:
    ```
    [
        # Creates two crops of the image so each input results in two training samples
        # image 1 and image 2 are different augmentations of the same crop
        { # Crop 1, 
            "image": <transformed_image_1>,
            "gt_image": <ground_truth_image>,
            "image_2": <transformed_image_2>
        },
        { # Crop 2
            "image": <transformed_image_1>,
            "gt_image": <ground_truth_image>,
            "image_2": <transformed_image_2>
        }
    ]
    ```
    """
    
    return Compose(
        [
            # Read the image from file
            LoadImaged(keys=["image"], reader="itkreader"),
            # Ensures the channel dimension is first, if it's grey-scale image adds a channel dim
            # e.g., (H, W, D) -> (1, H, W, D)
            EnsureChannelFirstd(keys=["image"]),
            # Ensures the image is spaced correctly, i.e., each pixel represents 2mm x 2mm x 2mm
            Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode=("bilinear")),
            # Scales intensity to [0.0, 1.0] range clipping values outside [-57, 164]
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),
            Lambda(func=reorder),
            # Crops the foreground of the image, trimming out black space to reduce memory
            CropForegroundd(keys=["image"], source_key="image"),
            # Checks if the image is at least 64x128x128, if not pads with zeros
            SpatialPadd(keys=["image"], spatial_size=(64,128,128)),
            # Randomly extracts 2 samples of size 64x128x128 from the volume
            Resized(
                keys=["image"], 
                spatial_size=(64, 128, 128), 
                mode="trilinear" # Important for 3D medical images to stay smooth
            )
        ]
    )

def get_label_transform_pipeline() -> Compose:
    return Compose(
            [
            # Read the image from file
            LoadImaged(keys=["image"], reader="itkreader"),
            # Ensures the channel dimension is first, if it's grey-scale image adds a channel dim
            # e.g., (H, W, D) -> (1, H, W, D)
            EnsureChannelFirstd(keys=["image"]),
            # Ensures the image is spaced correctly, i.e., each pixel represents 2mm x 2mm x 2mm
            Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode=("bilinear")),
            # Scales intensity to [0.0, 1.0] range clipping values outside [-57, 164]
            
            # Not included for label images
            # ScaleIntensityRanged(
            #     keys=["image"],
            #     a_min=-57,
            #     a_max=164,
            #     b_min=0.0,
            #     b_max=1.0,
            #     clip=True
            # ),
            # Crops the foreground of the image, trimming out black space to reduce memory
            CropForegroundd(keys=["image"], source_key="image"),
            # Checks if the image is at least 64x128x128, if not pads with zeros
            SpatialPadd(keys=["image"], spatial_size=(64,128,128)),
            # Randomly extracts 2 samples of size 64x128x128 from the volume
            Lambda(func=reorder),
            CropForegroundd(keys=["image"], source_key="image"),
            # NOTE: SpatialPadd and RandSpatialCrop are removed.
            # We replace them with Resized to force the shape.
            Resized(
                keys=["image"], 
                spatial_size=(64, 128, 128), 
                mode="trilinear" # Important for 3D medical images to stay smooth
            )
        ]
        )

def one_hot(img, C):
    out = np.zeros((C, img.shape[0], img.shape[1], img.shape[2]))
    for i in range(C):
        out[i, ...] = img == i
    return out

def writeToNpy(path, target, training=False):

    for folder in os.listdir(path):
        frames = []
        if not os.path.isdir(os.path.join(path, folder)):
            continue
        for f in os.listdir(os.path.join(path, folder)):
            if f.find('patient') == -1 or f.find('gt') != -1 or f.find("frame") == -1:
                continue
            idx = f[16:18]
            if idx in frames:
                continue
            else:
                frames.append(idx)

        if len(frames) != 2:
            raise RuntimeError(f'got {len(frames)} why is not 2')

        if frames[0] < frames[1]:
            MinF = frames[0]
            MaxF = frames[1]
        else:
            MinF = frames[1]
            MaxF = frames[0]

        F1 = os.path.join(path, folder, folder + '_frame' + MinF)
        F2 = os.path.join(path, folder, folder + '_frame' + MaxF)
        
        load_transform_pipeline = get_training_transform_pipeline()
        load_label_pipeline = get_label_transform_pipeline()

        xName = F1 + '.nii.gz'
        yName = F2 + '.nii.gz'
        xSegName = F1 + '_gt.nii.gz'
        ySegName = F2 + '_gt.nii.gz'

        x = load_transform_pipeline({"image": xName})
        y = load_transform_pipeline({"image": yName})
        xSeg = load_label_pipeline({"image": xSegName})
        ySeg = load_label_pipeline({"image": ySegName})

        x = x['image'].numpy()[0, :, :, :]
        y = y['image'].numpy()[0, :, :, :]
        xSeg = xSeg['image'].numpy()[0, :, :, :]
        ySeg = ySeg['image'].numpy()[0, :, :, :]

        if training:
            newXseg = one_hot(xSeg, C=4)
            newYseg = one_hot(ySeg, C=4)
            np.savez(os.path.join(target, folder), x=x, y=y, xSeg = newXseg, ySeg=newYseg)
        else:
            np.savez(os.path.join(target, folder), x=x, y=y)
# pass
os.makedirs('./npdata/training', exist_ok=True)
os.makedirs('./npdata/testing', exist_ok=True)
os.makedirs('./npdata/validation', exist_ok=True)
writeToNpy('./ACDC/database/training', './npdata/training', training=True)
writeToNpy('./ACDC/database/testing', './npdata/testing', training=True)
writeToNpy('./ACDC/database/validation', './npdata/validation', training=True)