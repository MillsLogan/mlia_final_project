JSON_PATH = "Pretrain_MAE/json_files/cardiac/Cardiac_dataset.json"
DATA_ROOT = "ACDC/database"
LOG_DIR = "pretrain_experiments/full_mse_loss"
VERBOSE = True

import os
import json
import torch
import numpy as np
import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)
import itk
itk.ProcessObject.SetGlobalWarningDisplay(False)
from monai.utils.misc import set_determinism
from monai.data import Dataset as monaiDataset
from monai.data import DataLoader as monaiDataLoader
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

# from Pretrain_MAE.MAE_3d import MAE
from new_models import MAETransformer


def get_image_path(json_entry: str) -> str:
    """
    Given the JSON entry from the dataset, for a patient, return the full file path.
    Input: "training/patient001_frame01.nii.gz"
    Output: "ACDC/database/training/patient001/patient001_frame01.nii.gz"

    Args:
        json_entry (str): The JSON entry for a patient, e.g., "training/patient001_frame01.nii.gz".
    Returns:
        file_path (str): The full file path to the image.
    Raises:
        AssertionError: If the constructed file path does not exist.
    """


    folder, file = json_entry.split("/")
    patient_id = file.split("_")[0] # patient001
    file_path = os.path.join(DATA_ROOT, folder, patient_id, file)
    assert os.path.exists(file_path), f"File not found: {file_path}"
    return file_path

def load_dataset(print_ds_size: bool=VERBOSE) -> tuple[dict, dict]:
    """
    Loads and formats the dataset from the JSON file. Configures the path and returns
    training and validation datasets as dictionaries.

    Returns:
        training (dict): The training dataset entries.
        validation (dict): The validation dataset entries.
    """
    assert os.path.exists(JSON_PATH), f"JSON file not found: {JSON_PATH}"
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    assert data.get("training") is not None, "JSON file does not contain 'training' key."
    assert data.get("validation") is not None, "JSON file does not contain 'validation' key."

    for split in ["training", "validation"]:
        for i in range(len(data[split])):
            json_entry = data[split][i]["image"]
            file_path = get_image_path(json_entry)
            data[split][i]["image"] = file_path

    if print_ds_size:
        print(f"Training dataset size: {len(data['training'])} samples")
        print(f"Validation dataset size: {len(data['validation'])} samples")

    return data["training"], data["validation"]

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

    def reorder(data):
        data["image"] = np.transpose(data["image"], (0, 3, 2, 1))
        return data
    
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

def train_model_with_masked_error(
    model: MAETransformer,
    train_loader: monaiDataLoader,
    val_loader: monaiDataLoader,
    optimizer: torch.optim.Optimizer,
    max_epochs: int,
    val_interval: int
) -> None:
    # Get both for full image reconstruction, but only use MSE for backprop for now
    mse_loss = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss()

    # Lists to track losses
    masked_recon_loss_values = []
    full_recon_mse_loss_values = []
    full_recon_mae_loss_values = []
    
    val_masked_loss_values = []
    val_full_mse_loss_values = []
    val_full_mae_loss_values = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_masked_recon_loss = 0
        epoch_full_mse_recon_loss = 0
        epoch_full_mae_recon_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            
            inputs = batch_data["image"].to(device)

            optimizer.zero_grad()

            # The loss is unused in the original code
            outputs, masked_recon_loss = model(inputs)
            
            epoch_masked_recon_loss += masked_recon_loss.item()
            masked_recon_loss.backward()

            # Reconstruction loss between outputs and ground truth, not just the masked regions
            full_recon_mse_loss = mse_loss(outputs, inputs)
            full_recon_mae_loss = mae_loss(outputs, inputs)
            # Don't use the full reconstruction loss for backpropagation
            epoch_full_mse_recon_loss += full_recon_mse_loss.detach().item()
            epoch_full_mae_recon_loss += full_recon_mae_loss.detach().item()
            optimizer.step()

            if step % 10 == 0:
                print(f"{step}/{len(train_loader)}, train_loss: {masked_recon_loss.item():.4f}")

        epoch_masked_recon_loss /= step
        epoch_full_mse_recon_loss /= step
        epoch_full_mae_recon_loss /= step
        masked_recon_loss_values.append(epoch_masked_recon_loss)
        full_recon_mse_loss_values.append(epoch_full_mse_recon_loss)
        full_recon_mae_loss_values.append(epoch_full_mae_recon_loss)
        print(f"epoch {epoch + 1} average masked recon loss: {epoch_masked_recon_loss:.4f}, average full MSE recon loss: {epoch_full_mse_recon_loss:.4f}, average full MAE recon loss: {epoch_full_mae_recon_loss:.4f}")
        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_full_mse_loss = 0
            val_full_mae_loss = 0
            val_masked_loss = 0
            val_step = 0
            with torch.no_grad():
                for val_data in val_loader:
                    val_step += 1
                    val_inputs = val_data["image"].to(device)

                    val_reconstructions, val_masked_recon_loss = model(val_inputs)
                    val_mse_loss_batch = mse_loss(val_reconstructions, val_inputs)
                    val_mae_loss_batch = mae_loss(val_reconstructions, val_inputs)
                    val_full_mse_loss += val_mse_loss_batch.item()
                    val_full_mae_loss += val_mae_loss_batch.item()
                    val_masked_loss += val_masked_recon_loss.item()
                val_full_mse_loss /= val_step
                val_full_mae_loss /= val_step
                val_masked_loss /= val_step
                val_full_mse_loss_values.append(val_full_mse_loss)
                val_full_mae_loss_values.append(val_full_mae_loss)
                val_masked_loss_values.append(val_masked_loss)
                print(f"validation masked loss: {val_masked_loss:.4f}")
                print(f"validation full recon loss (MSE): {val_full_mse_loss:.4f}, validation full MAE recon loss: {val_full_mae_loss:.4f}")

    os.makedirs(LOG_DIR, exist_ok=True)

    np.savez(
        os.path.join(LOG_DIR, "training_losses.npz"),
        masked_recon_loss=np.array(masked_recon_loss_values),
        full_mse_loss=np.array(full_recon_mse_loss_values),
        full_mae_loss=np.array(full_recon_mae_loss_values),
    )

    np.savez(
        os.path.join(LOG_DIR, "validation_losses.npz"),
        val_masked_loss=np.array(val_masked_loss_values),
        val_full_mse_loss=np.array(val_full_mse_loss_values),
        val_full_mae_loss=np.array(val_full_mae_loss_values),
    )

    torch.save(
        model.state_dict(),
        os.path.join(LOG_DIR, "mae_pretrained_model.pth")
    )


def train_model_with_full_mse_error(
    model: MAETransformer,
    train_loader: monaiDataLoader,
    val_loader: monaiDataLoader,
    optimizer: torch.optim.Optimizer,
    max_epochs: int,
    val_interval: int
) -> None:
    # Get both for full image reconstruction, but only use MSE for backprop for now
    mse_loss = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss()

    # Lists to track losses
    masked_recon_loss_values = []
    full_recon_mse_loss_values = []
    full_recon_mae_loss_values = []
    
    val_masked_loss_values = []
    val_full_mse_loss_values = []
    val_full_mae_loss_values = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_masked_recon_loss = 0
        epoch_full_mse_recon_loss = 0
        epoch_full_mae_recon_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            
            inputs = batch_data["image"].to(device)          # masked image
            
            # gt_images = batch_data["gt_image"].to(device)    # ground truth image
            # inputs_2 = batch_data["image_2"].to(device)      # second masked image

            # This is different from the original code
            # This stacks the two images along the batch dimension
            # So if batch size is 2, inputs will have 4 images: [img1, img2, img1_2, img2_2]
            # This allows processing both images in one forward pass
            # inputs = torch.cat((inputs, inputs_2), dim=0)

            optimizer.zero_grad()

            # The loss is unused in the original code
            outputs, masked_recon_loss = model(inputs)
            
            epoch_masked_recon_loss += masked_recon_loss.detach().item()
            # masked_recon_loss.backward()

            # Reconstruction loss between outputs and ground truth, not just the masked regions
            full_recon_mse_loss = mse_loss(outputs, inputs)
            full_recon_mse_loss.backward() # Backprop using full MSE loss
            full_recon_mae_loss = mae_loss(outputs, inputs)
            # Don't use the full reconstruction loss for backpropagation
            epoch_full_mse_recon_loss += full_recon_mse_loss.item()
            epoch_full_mae_recon_loss += full_recon_mae_loss.detach().item()
            optimizer.step()

            if step % 10 == 0:
                print(f"{step}/{len(train_loader)}, train_loss: {(full_recon_mse_loss / step):.4f}")

        epoch_masked_recon_loss /= step
        epoch_full_mse_recon_loss /= step
        epoch_full_mae_recon_loss /= step
        masked_recon_loss_values.append(epoch_masked_recon_loss)
        full_recon_mse_loss_values.append(epoch_full_mse_recon_loss)
        full_recon_mae_loss_values.append(epoch_full_mae_recon_loss)
        print(f"epoch {epoch + 1} average masked recon loss: {epoch_masked_recon_loss:.4f}, average full MSE recon loss: {epoch_full_mse_recon_loss:.4f}, average full MAE recon loss: {epoch_full_mae_recon_loss:.4f}")
        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_full_mse_loss = 0
            val_full_mae_loss = 0
            val_masked_loss = 0
            val_step = 0
            with torch.no_grad():
                for val_data in val_loader:
                    val_step += 1
                    val_inputs = val_data["image"].to(device)

                    val_reconstructions, val_masked_recon_loss = model(val_inputs)
                    val_mse_loss_batch = mse_loss(val_reconstructions, val_inputs)
                    val_mae_loss_batch = mae_loss(val_reconstructions, val_inputs)
                    val_full_mse_loss += val_mse_loss_batch.item()
                    val_full_mae_loss += val_mae_loss_batch.item()
                    val_masked_loss += val_masked_recon_loss.item()
                val_full_mse_loss /= val_step
                val_full_mae_loss /= val_step
                val_masked_loss /= val_step
                val_full_mse_loss_values.append(val_full_mse_loss)
                val_full_mae_loss_values.append(val_full_mae_loss)
                val_masked_loss_values.append(val_masked_loss)
                print(f"validation masked loss: {val_masked_loss:.4f}")
                print(f"validation full recon loss (MSE): {val_full_mse_loss:.4f}, validation full MAE recon loss: {val_full_mae_loss:.4f}")

    os.makedirs(LOG_DIR, exist_ok=True)

    np.savez(
        os.path.join(LOG_DIR, "training_losses.npz"),
        masked_recon_loss=np.array(masked_recon_loss_values),
        full_mse_loss=np.array(full_recon_mse_loss_values),
        full_mae_loss=np.array(full_recon_mae_loss_values),
    )

    np.savez(
        os.path.join(LOG_DIR, "validation_losses.npz"),
        val_masked_loss=np.array(val_masked_loss_values),
        val_full_mse_loss=np.array(val_full_mse_loss_values),
        val_full_mae_loss=np.array(val_full_mae_loss_values),
    )

    torch.save(
        model.state_dict(),
        os.path.join(LOG_DIR, "mae_pretrained_model.pth")
    )


def main():
    # Load dataset
    train_ds, val_ds = load_dataset()

    # Set determinism for reproducibility
    set_determinism(seed=123)

    training_transforms = get_training_transform_pipeline()

    # Base model from paper
    # Patch size: 16
    # Encdoer dim: 768
    # MLP dim: 3072
    # ViT layers: 12 - encoder depth
    # ViT head: 12 - encoder heads
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
    # It should be noted that this model has 76,836,625
    # The encoder alone has 66,140,160
    # The paper mentions 63.837M parameters for the base model
    # I'm assuming they meant the encoder only, but there is still a discrepancy of ~2.3M parameters
    
    # Training parameters
    max_epochs = 1 # From paper
    val_interval = 1 # From code, after how many epochs to validate
    batch_size = 2 # From paper
    lr = 1e-4 # From paper
    workers = 0 # From code, number of workers for data loading
    
    # From code, not mentioned in paper, except in Figure 12
    # contrastive_loss = ContrastiveLoss(temperature=0.05)

    # Paper does mention using Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train Dataset and DataLoader
    train_dataset = monaiDataset(data=train_ds, transform=training_transforms)
    train_loader = monaiDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    # Validation Dataset and DataLoader
    val_dataset = monaiDataset(data=val_ds, transform=training_transforms)
    val_loader = monaiDataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=workers)

    # train_model_with_masked_error(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     optimizer=optimizer,
    #     max_epochs=max_epochs,
    #     val_interval=val_interval
    # )

    train_model_with_full_mse_error(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        max_epochs=max_epochs,
        val_interval=val_interval
    )

if __name__ == "__main__":
    main()