import matplotlib.pyplot as plt
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, 
    ScaleIntensityRanged, CropForegroundd, SpatialPadd, 
    RandSpatialCropSamplesd, Resized, Lambda, OneOf, RandCoarseDropoutd
)
import numpy as np
import torch
import tools.utils as utils
from train import RegistrationNet
from new_models import MAETransformer
from torchvision import transforms
from data_pre import datasets, trans
import SimpleITK as sitk



def reorder(data):
    data["image"] = np.transpose(data["image"], (0, 3, 2, 1))
    return data
load_pipeline = Compose([
    LoadImaged(keys=["image"], reader="ITKReader")
])

crop_pipeline = Compose([
    LoadImaged(keys=["image"], reader="ITKReader"),
    EnsureChannelFirstd(keys=["image"]),
    Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode=("trilinear")),
    ScaleIntensityRanged(
        keys=["image"], a_min=-57, a_max=164, 
        b_min=0.0, b_max=1.0, clip=True
    ),
    # Lambda(func=reorder),
    CropForegroundd(keys=["image"], source_key="image"),
    SpatialPadd(keys=["image"], spatial_size=(64,128,128)),
    # Returns a LIST of dictionaries (num_samples=2)
    RandSpatialCropSamplesd(keys=["image"], roi_size=(64,128,128), random_size=False, num_samples=1),
    OneOf(transforms=[
            RandCoarseDropoutd(keys=["image"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True,
                               max_spatial_size=32),
            RandCoarseDropoutd(keys=["image"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False,
                               max_spatial_size=64),
            ]
        ),
])

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



def visualize_three_way(image_path):
    # raw_image = load_pipeline({"image": image_path})["image"]
    
    data = {"image": image_path}
    
    # Run Crop Pipeline
    crop_result = crop_pipeline(data)[0]["image"] # Get first sample
    
    # Run Zoom Pipeline
    zoom_result = zoom_pipeline(data)["image"]

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
    axes[0].set_title(f"Original Unaltered")
    axes[0].imshow(img_raw, cmap="gray")
    axes[0].axis("off")

    # Plot Cropped
    axes[1].set_title(f"Original Preprocessing Pipeline")
    axes[1].imshow(img_crop, cmap="gray", vmin=0, vmax=1) # Scaled 0-1
    axes[1].axis("off")

    # Plot Zoomed
    axes[2].set_title(f"Our Preprocessing Pipeline (Zoom to Fit)")
    axes[2].imshow(img_zoom, cmap="gray", vmin=0, vmax=1) # Scaled 0-1
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("preprocessing_comparison.png")
    plt.show()

    
def get_final_results(model, save_path: str):
    reg_model_val = utils.register_model((64,128,128), 'nearest')
    test_composed = transforms.Compose([trans.Seg_norm(), #rearrange segmentation label to 4 class
                                        trans.NumpyType((np.float32, np.int16)),
                                            ])
    test_dataset = datasets.CardiacInferDataset('./npdata/testing', transforms=test_composed)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    x_flow_values = []
    y_flow_values = []
    z_flow_values = []
    seg_2_dice_values = []
    seg_3_dice_values = []
    seg_4_dice_values = []
    seg_2_hd_values = []
    seg_3_hd_values = []
    seg_4_hd_values = []
    model = model.cuda()
    for x, y, x_seg, y_seg in test_loader:
        x = x.cuda()
        y = y.cuda()
        x_seg = x_seg.cuda()
        y_seg = y_seg.cuda()
        x_seg2 = x_seg[:, :, 1, ...]
        x_seg3 = x_seg[:, :, 2, ...]
        x_seg4 = x_seg[:, :, 3, ...]
        y_seg2 = y_seg[:, :, 1, ...]
        y_seg3 = y_seg[:, :, 2, ...]
        y_seg4 = y_seg[:, :, 3, ...]
        inputs = torch.cat((x, y), dim=1)
        deformation = model(inputs)
        x_flow_values.append(deformation[0, 0, ...].detach().cpu().numpy())
        y_flow_values.append(deformation[0, 1, ...].detach().cpu().numpy())
        z_flow_values.append(deformation[0, 2, ...].detach().cpu().numpy())

        deformation_val2 = reg_model_val([x_seg2, deformation])
        deformation_val3 = reg_model_val([x_seg3, deformation])
        deformation_val4 = reg_model_val([x_seg4, deformation])

        dice_2 = utils.dice_val(deformation_val2, y_seg2).cpu().detach().numpy()
        dice_3 = utils.dice_val(deformation_val3, y_seg3).cpu().detach().numpy()
        dice_4 = utils.dice_val(deformation_val4, y_seg4).cpu().detach().numpy()
        seg_2_dice_values.append(dice_2)
        seg_3_dice_values.append(dice_3)
        seg_4_dice_values.append(dice_4)
        hd_2 = utils.hd(deformation_val2.detach().cpu().numpy(), y_seg2.detach().cpu().numpy())
        hd_3 = utils.hd(deformation_val3.detach().cpu().numpy(), y_seg3.detach().cpu().numpy())
        hd_4 = utils.hd(deformation_val4.detach().cpu().numpy(), y_seg4.detach().cpu().numpy())
        seg_2_hd_values.append(hd_2)
        seg_3_hd_values.append(hd_3)
        seg_4_hd_values.append(hd_4)
    import pandas as pd
    df = pd.DataFrame({
        'x_flow_mag_avg': [np.mean([np.mean(np.abs(f)) for f in x_flow_values])], # Length 1
        'y_flow_mag_avg': [np.mean([np.mean(np.abs(f)) for f in y_flow_values])], # Length 1
        'z_flow_mag_avg': [np.mean([np.mean(np.abs(f)) for f in z_flow_values])], # Length 1
        'x_flow_mag_std': [np.std([np.std(np.abs(f)) for f in x_flow_values])], # Length 1
        'y_flow_mag_std': [np.std([np.std(np.abs(f)) for f in y_flow_values])], # Length 1
        'z_flow_mag_std': [np.std([np.std(np.abs(f)) for f in z_flow_values])], # Length 1
        'seg_2_dice': [np.mean(seg_2_dice_values)], # Length 1
        'seg_3_dice': [np.mean(seg_3_dice_values)], # Length 1
        'seg_4_dice': [np.mean(seg_4_dice_values)], # Length 1
        'dice_avg': [(np.mean(seg_2_dice_values)+np.mean(seg_3_dice_values)+np.mean(seg_4_dice_values))/3], # Length 1
        'seg_2_hd': [np.mean(seg_2_hd_values)], # Length 1
        'seg_3_hd': [np.mean(seg_3_hd_values)], # Length 1
        'seg_4_hd': [np.mean(seg_4_hd_values)], # Length 1
        'hd_avg': [(np.mean(seg_2_hd_values)+np.mean(seg_3_hd_values)+np.mean(seg_4_hd_values))/3], # Length 1
    })
    df.to_csv(save_path, index=False)

def get_large_model(use_pretrained: bool = True):
    model = RegistrationNet(img_size=(64,128,128), 
                        in_channels=2, 
                        out_channels=3, 
                        enc_num_layers=24, 
                        enc_num_heads=16, 
                        encoder_dim=1024, 
                        mlp_dim=4096, 
                        dec_num_layers=6, 
                        dec_num_heads=4, 
                        decoder_dim=512, 
                        patch_size=16, 
                        masking_ratio=0.)
    if use_pretrained:
        weights = torch.load("final_models/registration/large_pretrain.pth.tar", weights_only=False)
    else:
        weights = torch.load("final_models/registration/large_random.pth.tar", weights_only=False)
    del weights['optimizer']
    del weights['epoch']
    del weights['best_mse']
    model.load_state_dict(weights['state_dict'])
    return model

def get_huge_model(use_pretrained: bool = True):
    model = RegistrationNet(img_size=(64,128,128), 
                        in_channels=2, 
                        out_channels=3, 
                        enc_num_layers=32, 
                        enc_num_heads=16, 
                        encoder_dim=1280, 
                        mlp_dim=5120, 
                        dec_num_layers=6, 
                        dec_num_heads=4, 
                        decoder_dim=512, 
                        patch_size=16, 
                        masking_ratio=0.)
    if use_pretrained:
        weights = torch.load("final_models/registration/huge_pretrain.pth.tar", weights_only=False)
    else:
        weights = torch.load("final_models/registration/huge_random.pth.tar", weights_only=False)
    del weights['optimizer']
    del weights['epoch']
    del weights['best_mse']
    model.load_state_dict(weights['state_dict'])
    return model

def get_base_model(use_pretrained: bool = True):
    model = RegistrationNet(img_size=(64,128,128), 
                        in_channels=2, 
                        out_channels=3, 
                        enc_num_layers=12, 
                        enc_num_heads=12, 
                        encoder_dim=768, 
                        mlp_dim=3072, 
                        dec_num_layers=6, 
                        dec_num_heads=4, 
                        decoder_dim=512, 
                        patch_size=16, 
                        masking_ratio=0.)
    if use_pretrained:
        weights = torch.load("final_models/registration/large_pretrain.pth.tar", weights_only=False)
    else:
        weights = torch.load("final_models/registration/large_random.pth.tar", weights_only=False)
    del weights['optimizer']
    del weights['epoch']
    del weights['best_mse']
    model.load_state_dict(weights['state_dict'])
    return model

def base_mae_model():
    model = MAETransformer(img_size=(64,128,128), 
                        enc_num_layers=12, 
                        enc_num_heads=12, 
                        encoder_dim=768, 
                        mlp_dim=3072, 
                        dec_num_layers=6, 
                        dec_num_heads=4, 
                        decoder_dim=512, 
                        patch_size=16, 
                        masking_ratio=0.75)
    weights = torch.load("final_models/mae/base_mae.pth")
    model.load_state_dict(weights)
    return model

def large_mae_model():
    model = MAETransformer(img_size=(64,128,128), 
                        enc_num_layers=24, 
                        enc_num_heads=16, 
                        encoder_dim=1024, 
                        mlp_dim=4096, 
                        dec_num_layers=6, 
                        dec_num_heads=4, 
                        decoder_dim=512, 
                        patch_size=16, 
                        masking_ratio=0.75)
    weights = torch.load("final_models/mae/large_mae.pth")
    model.load_state_dict(weights)
    return model

def huge_mae_model():
    model = MAETransformer(img_size=(64,128,128), 
                        enc_num_layers=32, 
                        enc_num_heads=16, 
                        encoder_dim=1280, 
                        mlp_dim=5120, 
                        dec_num_layers=6, 
                        dec_num_heads=4, 
                        decoder_dim=512, 
                        patch_size=16, 
                        masking_ratio=0.75)
    weights = torch.load("final_models/mae/huge_mae.pth")
    model.load_state_dict(weights)
    return model

def reconstruction_results(model, save_path: str, get_image: bool=False):
    test_composed = transforms.Compose([trans.Seg_norm(), #rearrange segmentation label to 4 class
                                        trans.NumpyType((np.float32, np.int16)),
                                            ])
    test_dataset = datasets.CardiacInferDataset('./npdata/testing', transforms=test_composed)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    mse_values = []
    model = model.cuda()
    model.eval()
    for x, y, _, _ in test_loader:
        x = x.cuda()
        y = y.cuda()
        x_out, _ = model(x, False)
        y_out, _ = model(y, False)
        mse_values.append(torch.nn.MSELoss()(x_out, x).cpu().detach().numpy())
        mse_values.append(torch.nn.MSELoss()(y_out, y).cpu().detach().numpy())
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        if get_image:
            axs[0].imshow(x[0,0,32,:,:].cpu().detach().numpy(), cmap='gray')
            axs[0].set_title('Original Image Slice')
            axs[0].axis('off')
            axs[1].imshow(x_out[0,0,32,:,:].cpu().detach().numpy(), cmap='gray')
            axs[1].set_title('Reconstructed Image Slice')
            axs[1].axis('off')
            plt.tight_layout()
            plt.savefig(f'{save_path}', dpi=300)
            plt.close()
            exit()
    import pandas as pd
    df = pd.DataFrame({
        'reconstruction_mse_avg': [np.mean(mse_values)], # Length 1
        'reconstruction_mse_std': [np.std(mse_values)], # Length 1
    })
    df.to_csv(save_path, index=False)

if __name__ == "__main__":
    # MAE Reconstruction Results
    model = base_mae_model()
    reconstruction_results(model, "results/base_mae_reconstruction_result.csv", get_image=False)
    reconstruction_results(model, "results/base_mae_reconstruction_image.png", get_image=True)
    model = large_mae_model()
    reconstruction_results(model, "results/large_mae_reconstruction_result.csv", get_image=False)
    reconstruction_results(model, "results/large_mae_reconstruction_image.png", get_image=True)
    model = huge_mae_model()
    reconstruction_results(model, "results/huge_mae_reconstruction_result.csv", get_image=False)
    reconstruction_results(model, "results/huge_mae_reconstruction_image.png", get_image=True)
    
    # Registration Results
    # base_model = get_base_model(use_pretrained=False)
    # get_final_results(base_model, "results/base_model_results.csv")
    # del base_model
    # large_model = get_large_model(use_pretrained=False)
    # get_final_results(large_model, "results/large_model_results.csv")
    # del large_model
    # huge_model = get_huge_model(use_pretrained=False)
    # get_final_results(huge_model, "results/huge_model_results.csv")
    # del huge_model
    # pretrained_base_model = get_base_model(use_pretrained=True)
    # get_final_results(pretrained_base_model, "results/pretrained_base_model_results.csv")
    # del pretrained_base_model
    # pretrained_large_model = get_large_model(use_pretrained=True)
    # get_final_results(pretrained_large_model, "results/pretrained_large_model_results.csv")
    # del pretrained_large_model
    # pretrained_huge_model = get_huge_model(use_pretrained=True)
    # get_final_results(pretrained_huge_model, "results/pretrained_huge_model_results.csv")
    # del pretrained_huge_model