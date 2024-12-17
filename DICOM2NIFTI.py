import os
import pydicom
import nibabel as nib
import numpy as np

# Specify the input directory containing DICOM files
dicom_dir = 'E:\Projects\GLCM_Analysis_of_Brain\data\T1'

# Load DICOM series
dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir)]
dicom_files.sort()  # Ensure proper ordering of slices
dicom_slices = [pydicom.dcmread(f) for f in dicom_files]

# Extract necessary DICOM metadata
voxel_data = [s.pixel_array for s in dicom_slices]
voxel_data = np.stack(voxel_data, axis=-1)
voxel_data = np.rot90(voxel_data, k=1, axes=(0, 1))

# Get voxel spacing information
voxel_spacing = dicom_slices[0].PixelSpacing
slice_thickness = dicom_slices[0].SliceThickness 

# Create a NIfTI image object

nifti_img = nib.Nifti1Image(voxel_data, np.eye(4))

# Set the voxel spacing information in the image header
nifti_img.header.set_zooms((float(voxel_spacing[0]), float(voxel_spacing[1]), float(slice_thickness)))


# Set the orientation information in the image header
nifti_img.header.set_qform(np.eye(4), code='scanner')
nifti_img.header.set_sform(np.eye(4), code='scanner')

# Save the NIfTI image to a file
nifti_filepath = 'E:\Projects\GLCM_Analysis_of_Brain\data\T1\image.nii.gz'
nib.save(nifti_img, nifti_filepath)

print('DICOM series converted to NIfTI:', nifti_filepath)