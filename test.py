import numpy as np
import nibabel as nib
import SimpleITK as sitk
import os

def is_sform_orthonormal(img_path, tolerance=1e-6):
    """
    Checks if the sform matrix of a NIfTI image is orthonormal.
    """
    if not os.path.exists(img_path):
        print(f"Error: File not found at {img_path}")
        return False

    try:
        img = sitk.ReadImage(img_path)
        # The sform is a 4x4 affine matrix. 
        # The top-left 3x3 submatrix (the rotation/zoom part) should be orthonormal.
        sform_rotation_zoom = img.affine[:3, :3]
        
        # Check if the matrix is square
        if sform_rotation_zoom.shape[0] != sform_rotation_zoom.shape[1]:
            print("Sform rotation part is not square.")
            return False
            
        # Compute M^T @ M (matrix multiplication of transpose and original)
        mt_m = sform_rotation_zoom.T @ sform_rotation_zoom
        
        # Create an identity matrix of the same size
        identity = np.eye(3)
        
        # Check if mt_m is approximately equal to the identity matrix due to floating point inaccuracies
        is_orthonormal = np.allclose(mt_m, identity, atol=tolerance)
        
        if is_orthonormal:
            print("The sform matrix is orthonormal (within tolerance).")
        else:
            print("The sform matrix is not orthonormal.")
            # Optional: print the difference to see how far it is from identity
            print("Difference (M^T @ M - I):")
            print(mt_m - identity)
            
        return is_orthonormal
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

from models import MAE_Transformer
from Models_configs import get_config
model = MAE_Transformer(get_config(), img_size=(64, 128, 128))
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params}")