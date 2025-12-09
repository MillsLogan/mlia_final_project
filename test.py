import matplotlib.pyplot as plt
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, 
    ScaleIntensityRanged, CropForegroundd, SpatialPadd, 
    RandSpatialCropSamplesd, Resized, Lambda
)
import numpy as np

from tools import utils
from train import RegistrationNet
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


import matplotlib.pyplot as plt
import numpy as np

def visualize_deformation_with_seg(seg_tensor, flow_tensor, slice_idx=None, axis=1, stride=4):
    """
    seg_tensor:  (4, 64, 128, 128) - One-hot encoded segmentation
    flow_tensor: (3, 64, 128, 128) - Deformation field (Z, Y, X displacements usually)
    slice_idx:   The index of the slice to view. If None, picks the middle.
    axis:        The axis to slice along (0=Depth/64, 1=Height/128, 2=Width/128). 
                 Note: Since your data is (C, D, H, W), axis 0 refers to D.
    stride:      Skip every Nth arrow to prevent clutter.
    """
    
    # --- 1. PREPROCESSING ---
    
    # Convert Segmentation from (4, D, H, W) -> (D, H, W) using Argmax
    # This collapses the 4 channels into a single map with values 0, 1, 2, 3
    if seg_tensor.shape[0] == 4:
        seg_mask = np.argmax(seg_tensor, axis=0) 
    else:
        seg_mask = seg_tensor

    # Convert Flow from (3, D, H, W) -> (D, H, W, 3) 
    # We move the channel dim to the end for easier slicing
    flow_permuted = np.moveaxis(flow_tensor, 0, -1)
    
    # Handle default slice index (middle of the volume)
    if slice_idx is None:
        slice_idx = seg_mask.shape[axis] // 2

    # --- 2. EXTRACT SLICES ---
    
    # We need to handle slicing based on which axis (D, H, or W) we are looking at.
    # We also need to pick the correct 2D flow vectors for that plane.
    
    if axis == 0: # Slicing Depth (Viewing the H-W plane) - The most common view
        seg_slice = seg_mask[slice_idx, :, :]
        # Flow vector 1 is usually Y (Height), Vector 2 is X (Width)
        flow_u = flow_permuted[slice_idx, :, :, 2] # X displacement
        flow_v = flow_permuted[slice_idx, :, :, 1] # Y displacement
        xlabel, ylabel = "Width", "Height"

    elif axis == 1: # Slicing Height (Viewing the D-W plane)
        seg_slice = seg_mask[:, slice_idx, :]
        flow_u = flow_permuted[:, slice_idx, :, 2] # X displacement
        flow_v = flow_permuted[:, slice_idx, :, 0] # Z displacement
        xlabel, ylabel = "Width", "Depth"
        
    elif axis == 2: # Slicing Width (Viewing the D-H plane)
        seg_slice = seg_mask[:, :, slice_idx]
        flow_u = flow_permuted[:, :, slice_idx, 1] # Y displacement
        flow_v = flow_permuted[:, :, slice_idx, 0] # Z displacement
        xlabel, ylabel = "Height", "Depth"

    # --- 3. VISUALIZATION ---
    
    plt.figure(figsize=(10, 8))
    
    # A. Plot Magnitude Heatmap (Optional background)
    magnitude = np.sqrt(flow_u**2 + flow_v**2)
    plt.imshow(magnitude, cmap='gray', alpha=0.3, origin='lower')
    
    # B. Overlay Segmentation Contours
    # We assume classes are 0 (Bg), 1 (RV), 2 (Myo), 3 (LV)
    # We define specific colors for each class
    colors = ['black', 'red', 'lime', 'blue'] 
    labels = ['Background', 'RV', 'Myo', 'LV']
    
    # Loop through classes 1, 2, 3 to draw contours
    for i in range(1, 4):
        # Create a binary mask for the specific class
        class_mask = (seg_slice == i).astype(float)
        if np.any(class_mask):
            plt.contour(class_mask, levels=[0.5], colors=[colors[i]], linewidths=2.5)
            # Dummy plot for legend
            plt.plot([], [], color=colors[i], label=labels[i])

    # C. Plot Deformation Vectors (Quiver)
    # We use meshgrid to define the arrow positions
    h, w = seg_slice.shape
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    
    # Apply Stride (downsample)
    plt.quiver(x_grid[::stride, ::stride], 
               y_grid[::stride, ::stride], 
               flow_u[::stride, ::stride], 
               flow_v[::stride, ::stride],
               color='orange', 
               angles='xy', scale_units='xy', scale=1, 
               width=0.002, alpha=0.8, label='Deformation')

    plt.title(f"Deformation Field + Segmentation (Slice {slice_idx} along Axis {axis})")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper right')
    plt.show()

# --- EXAMPLE USAGE ---
# Assuming you have your numpy arrays or torch tensors ready:
# If they are torch tensors, use .cpu().numpy() first.

# Example inputs (Random noise for demonstration)
# seg_data = np.random.randint(0, 4, size=(4, 64, 128, 128)) # Warning: this is not one-hot, just dummy
# flow_data = np.random.randn(3, 64, 128, 128) * 2

# visualize_deformation_with_seg(seg_data, flow_data, slice_idx=32, axis=0, stride=5)

# Usage
# visualize_three_way("path/to/image.nii.gz")
# ---------------------------------------------------------
# Usage: Replace with your actual file path
# visualize_three_way("ACDC/database/training/patient001/patient001_frame01.nii.gz")
# exit()
from new_models import MAETransformer
import torch
# model = MAETransformer(
#         img_size=(64,128,128),
#         patch_size=16,
#         encoder_dim=768,
#         mlp_dim=3072,
#         masking_ratio = 0.75,   # the paper recommended 75% masked patches
#         decoder_dim = 512,      # paper showed good results with just 512
#         dec_num_layers = 6,       # anywhere from 1 to 8
#         dec_num_heads=4,
#         enc_num_layers=12,
#         enc_num_heads=12
#     )
# weights = torch.load("mae_pretrained_model.pth")
model = RegistrationNet(img_size=(64,128,128), in_channels=2, enc_num_layers=12, enc_num_heads=12, encoder_dim=768, mlp_dim=3072, dec_num_layers=6, dec_num_heads=4, decoder_dim=512, patch_size=16, masking_ratio=0.75)
weights = torch.load("Dice-0.598.pth.tar", weights_only=False)
del weights['optimizer']
del weights['epoch']
del weights['best_mse']
model.load_state_dict(weights['state_dict'])
img = np.load("npdata/training/patient001.npz")
frame_1 = img['x'].astype(np.float32)  # Moving
frame_2 = img['y'].astype(np.float32)  # Fixed
seg_x = img['xSeg'].astype(np.float32)
seg_y = img['ySeg'].astype(np.float32)

model.eval()
raw_image = torch.cat((torch.from_numpy(frame_1).unsqueeze(0).unsqueeze(0), torch.from_numpy(frame_2).unsqueeze(0).unsqueeze(0)), dim=1)
deformation = model(raw_image)  # Add batch dimension
# Usage: Plot middle slice
flow_tensor = deformation.squeeze(0).cpu().detach().numpy()
print(f"Max Flow: {np.max(flow_tensor)}")
print(f"Min Flow: {np.min(flow_tensor)}")
print(f"Mean Flow: {np.mean(np.abs(flow_tensor))}")
exit()
visualize_deformation_with_seg(seg_x, deformation.squeeze(0).cpu().detach().numpy(), slice_idx=32, axis=0, stride=4)
exit()
reg_model = utils.register_model((64,128,128), 'bilinear')
output = reg_model([raw_image.unsqueeze(0).cuda().float(), deformation.cuda()])
reconstruction = output.squeeze(0)
# Visualize deformed image

reconstruction = torch.squeeze(reconstruction, 0).squeeze(0).cuda()
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
raw_image = raw_image.squeeze(0)
print(torch.nn.functional.mse_loss(reconstruction, raw_image.cuda()))
# Plot Raw
axes[0].set_title(f"Original Unaltered\nShape: {raw_image.shape}\nSlice: {32}")
# We do not specify vmin/vmax here to let matplotlib auto-scale the raw intensities
axes[0].imshow(raw_image[1, 1, :, :].cpu(), cmap="gray")
axes[0].axis("off")
axes[1].imshow(reconstruction[1, :, :].cpu().detach(), cmap="gray")
axes[1].axis("off")
plt.tight_layout()
plt.show()