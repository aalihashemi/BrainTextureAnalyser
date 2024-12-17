'''

import nibabel as nib
import numpy as np
from brainextractor import BrainExtractor

input_img = nib.load('E:\Projects\GLCM_Analysis_of_Brain\data\image.nii.gz')

bet = BrainExtractor(img=input_img)
bet.run()

bet.save_mask("E:\\Projects\\GLCM_Analysis_of_Brain\data\\brainextracted_mask.nii.gz")
mask_img = nib.load('E:\\Projects\\GLCM_Analysis_of_Brain\\data\\brainextracted_mask.nii.gz')

input_array = input_img.get_fdata()
mask_array = mask_img.get_fdata()
result_array = input_array * mask_array

result_img = nib.Nifti1Image(result_array, input_img.affine)
nib.save(result_img, "E:\\Projects\\GLCM_Analysis_of_Brain\data\\masked_image.nii.gz")
'''

'''
import numpy as np
import matplotlib.pyplot as plt
from dipy.data import fetch_tissue_data, read_tissue_data
from dipy.segment.tissue import TissueClassifierHMRF
import nibabel as nib

t1 = nib.load('E:\Projects\GLCM_Analysis_of_Brain\data\image.nii.gz').get_fdata()

fig = plt.figure()
a = fig.add_subplot(1, 2, 1)
img_ax = np.rot90(t1[..., 89])
imgplot = plt.imshow(img_ax, cmap="gray")
a.axis('off')
a.set_title('Axial')
a = fig.add_subplot(1, 2, 2)
img_cor = np.rot90(t1[:, 128, :])
imgplot = plt.imshow(img_cor, cmap="gray")
a.axis('off')
a.set_title('Coronal')
plt.savefig('t1_image.png', bbox_inches='tight', pad_inches=0)

import time
t0 = time.time()
nclass = 3
beta = 0.1
hmrf = TissueClassifierHMRF(verbose=True)
initial_segmentation, final_segmentation, PVE = hmrf.classify(t1, nclass, beta, max_iter=2)

t1 = time.time()
total_time = t1-t0
print('Total time:' + str(total_time))

fig = plt.figure()
a = fig.add_subplot(1, 2, 1)
img_ax = np.rot90(final_segmentation[..., 89])
imgplot = plt.imshow(img_ax)
a.axis('off')
a.set_title('Axial')
a = fig.add_subplot(1, 2, 2)
img_cor = np.rot90(final_segmentation[:, 128, :])
imgplot = plt.imshow(img_cor)
a.axis('off')
a.set_title('Coronal')
plt.savefig('final_seg.png', bbox_inches='tight', pad_inches=0)
'''
import ants
import antspynet

import nibabel as nib

t1 = ants.image_read('E:\Projects\GLCM_Analysis_of_Brain\data\image.nii.gz')

# # ANTs-flavored
# seg = antspynet.brain_extraction(t1, modality="t1", verbose=True)
# ants.plot(t1, overlay=seg, overlay_alpha=0.5)
# # FreeSurfer-flavored
# seg = antspynet.brain_extraction(t1, modality="t1nobrainer", verbose=True)
# ants.plot(t1, overlay=seg, overlay_alpha=0.5)
# Combined
seg = antspynet.brain_extraction(t1, modality="t1combined", verbose=True)
ants.plot(t1, overlay=seg, overlay_alpha=0.5)
nib.save(seg, "E:\\Projects\\GLCM_Analysis_of_Brain\data\\masked_image.nii.gz")