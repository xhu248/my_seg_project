import dicom2nifti
import os

folder = 'echo speed -1'
output_folder = 'data/tapvc_dataset'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

l = os.path.join
subdir = [(l(folder, i), i) for i in os.listdir(folder)]

for dic_dir in subdir:
    output_file = os.path.join(output_folder, dic_dir[1]+'.nii.gz')
    dicom2nifti.dicom_series_to_nifti(dic_dir[0], output_file, reorient_nifti=True)
    print(dic_dir)

