'''
import ants

# Load your image
image = ants.image_read('E:\Projects\GLCM_Analysis_of_Brain\data\image.nii.gz')

# Load a template image for normalization
template = ants.image_read('E:\Projects\GLCM_Analysis_of_Brain\data\CTRL17-T1MPRAGE-template-0.5mm.nii.gz')

print("start registration")
# Perform spatial normalization
transformed_image = ants.registration(fixed=template , moving=image, type_of_transform='SyN')

# Save the normalized image
ants.image_write(transformed_image['warpedmovout'], 'E:\Projects\GLCM_Analysis_of_Brain\data\spatialnor.nii.gz')
'''
# Import necessary modules
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.data.fetcher import fetch_syn_data, read_syn_data
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)
import nibabel as nib

# Load the data
static_img = nib.load('E:\Projects\GLCM_Analysis_of_Brain\data\image.nii.gz')
static = static_img.get_data()
static_grid2world = static_img.affine

moving_img = nib.load('E:\Projects\GLCM_Analysis_of_Brain\data\CTRL17-T1MPRAGE-template-0.5mm.nii.gz')
moving = moving_img.get_data()
moving_grid2world = moving_img.affine

# Pre-align the data
c_of_mass = transform_centers_of_mass(static, static_grid2world,
                                      moving, moving_grid2world)

nbins = 32
sampling_prop = None
metric = MutualInformationMetric(nbins, sampling_prop)

level_iters = [10000, 1000, 100]
sigmas = [3.0, 1.0, 0.0]
factors = [4, 2, 1]
affreg = AffineRegistration(metric=metric,
                            level_iters=level_iters,
                            sigmas=sigmas,
                            factors=factors)

transform = TranslationTransform3D()
params0 = None
translation = affreg.optimize(static, moving, transform, params0,
                              static_grid2world, moving_grid2world)

transform = RigidTransform3D()
rigid = affreg.optimize(static, moving, transform, params0,
                        static_grid2world, moving_grid2world,
                        starting_affine=translation.affine)

transform = AffineTransform3D()
affine = affreg.optimize(static, moving, transform, params0,
                         static_grid2world, moving_grid2world,
                         starting_affine=rigid.affine)

# Save the result
resampled = affine.transform(moving)
new_img = nib.Nifti1Image(resampled, static_grid2world)
nib.save(new_img, 'path_to_save_your_result.nii.gz')
