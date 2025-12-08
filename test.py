import matplotlib.pyplot as plt
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, 
    ScaleIntensityRanged, CropForegroundd, SpatialPadd, 
    RandSpatialCropSamplesd, Resized, Lambda
)
import numpy as np
def reorder(data):
    data["image"] = np.transpose(data["image"], (0, 3, 2, 1))
    return data
load_pipeline = Compose([
    LoadImaged(keys=["image"], reader="ITKReader")
])
# 1. The Original Pipeline (Cropping)
# ---------------------------------------------------------
crop_pipeline = Compose([
    LoadImaged(keys=["image"], reader="ITKReader"),
    EnsureChannelFirstd(keys=["image"]),
    Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode=("trilinear")),
    ScaleIntensityRanged(
        keys=["image"], a_min=-57, a_max=164, 
        b_min=0.0, b_max=1.0, clip=True
    ),
    Lambda(func=reorder),
    CropForegroundd(keys=["image"], source_key="image"),
    SpatialPadd(keys=["image"], spatial_size=(64,128,128)),
    # Returns a LIST of dictionaries (num_samples=2)
    RandSpatialCropSamplesd(keys=["image"], roi_size=(64,128,128), random_size=False, num_samples=1), 
])

# 2. The New Pipeline (Zoom to Fit / Resizing)
# ---------------------------------------------------------
zoom_pipeline = Compose([
    LoadImaged(keys=["image"], reader="ITKReader"),
    EnsureChannelFirstd(keys=["image"]),
    Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode=("bilinear")),
    ScaleIntensityRanged(
        keys=["image"], a_min=-57, a_max=164, 
        b_min=0.0, b_max=1.0, clip=True
    ),
    Lambda(func=reorder),
    CropForegroundd(keys=["image"], source_key="image"),
    # NOTE: SpatialPadd and RandSpatialCrop are removed.
    # We replace them with Resized to force the shape.
    Resized(
        keys=["image"], 
        spatial_size=(64, 128, 128), 
        mode="trilinear" # Important for 3D medical images to stay smooth
    )
])

# 3. Helper function to plot
# ---------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import LoadImage
import SimpleITK as sitk

def visualize_three_way(image_path):
    raw_image = load_pipeline({"image": image_path})["image"]
    
    # 2. Run the Pipelines (Reusing your previous pipeline definitions)
    data = {"image": image_path}
    
    # Run Crop Pipeline
    crop_result = crop_pipeline(data)[0]["image"] # Get first sample
    
    # Run Zoom Pipeline
    zoom_result = zoom_pipeline(data)["image"]

    # 3. Convert to Numpy and strip Channel dimension
    # Result shapes should be (Depth, Height, Width)
    vol_raw = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    vol_crop = sitk.GetArrayFromImage(sitk.GetImageFromArray(crop_result[0].numpy()))
    vol_zoom = sitk.GetArrayFromImage(sitk.GetImageFromArray(zoom_result[0].numpy()))

    # 4. Calculate the "Middle" Slice for each volume
    # We use percentages because the raw volume might have 200 slices 
    # while the zoom/crop versions only have 64.
    slice_idx_raw = vol_raw.shape[0] // 2
    slice_idx_crop = vol_crop.shape[0] // 2
    slice_idx_zoom = vol_zoom.shape[0] // 2

    # 5. Extract the 2D Slices
    img_raw = vol_raw[slice_idx_raw, :, :]
    img_crop = vol_crop[slice_idx_crop, :, :]
    img_zoom = vol_zoom[slice_idx_zoom, :, :]

    # 6. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot Raw
    axes[0].set_title(f"Original Unaltered\nShape: {vol_raw.shape}\nSlice: {slice_idx_raw}")
    # We do not specify vmin/vmax here to let matplotlib auto-scale the raw intensities
    axes[0].imshow(img_raw, cmap="gray")
    axes[0].axis("off")

    # Plot Cropped
    axes[1].set_title(f"Cropped (MONAI)\nShape: {vol_crop.shape}\nSlice: {slice_idx_crop}")
    axes[1].imshow(img_crop, cmap="gray", vmin=0, vmax=1) # Scaled 0-1
    axes[1].axis("off")

    # Plot Zoomed
    axes[2].set_title(f"Zoomed (Resized)\nShape: {vol_zoom.shape}\nSlice: {slice_idx_zoom}")
    axes[2].imshow(img_zoom, cmap="gray", vmin=0, vmax=1) # Scaled 0-1
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

# Usage
# visualize_three_way("path/to/image.nii.gz")
# ---------------------------------------------------------
# Usage: Replace with your actual file path
# visualize_three_way("ACDC/database/training/patient001/patient001_frame01.nii.gz")
# exit()
from new_models import MAETransformer
import torch
model = MAETransformer(
        img_size=(64,128,128),
        patch_size=16,
        encoder_dim=768,
        mlp_dim=3072,
        masking_ratio = 0.75,   # the paper recommended 75% masked patches
        decoder_dim = 512,      # paper showed good results with just 512
        dec_num_layers = 6,       # anywhere from 1 to 8
        dec_num_heads=4,
        enc_num_layers=12,
        enc_num_heads=12
    )
weights = torch.load("pretrain_experiments/full_mse_loss/mae_pretrained_model.pth")
model.load_state_dict(weights)
loaded_example_image = zoom_pipeline({"image": "ACDC/database/training/patient001/patient001_frame01.nii.gz"})
model.eval()
raw_image = loaded_example_image["image"]
reconstruction, _ = model(raw_image.unsqueeze(0))
reconstruction = torch.squeeze(reconstruction, 0).squeeze(0)
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
raw_image = raw_image.squeeze(0)
print(torch.nn.functional.mse_loss(reconstruction, raw_image))
# Plot Raw
axes[0].set_title(f"Original Unaltered\nShape: {raw_image.shape}\nSlice: {32}")
# We do not specify vmin/vmax here to let matplotlib auto-scale the raw intensities
axes[0].imshow(raw_image[1, :, :].cpu(), cmap="gray")
axes[0].axis("off")
axes[1].imshow(reconstruction[1, :, :].cpu().detach(), cmap="gray")
axes[1].axis("off")
plt.tight_layout()
plt.show()