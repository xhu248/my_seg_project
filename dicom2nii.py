import dicom2nifti
import dicom2nifti.settings as settings
import shutil
import os

settings.disable_validate_slice_increment()
orig_folder = '/home/xinrong/tmp/19-11-22-tapvc+2'
folder = 'echo speed -1'
l = os.path.join

output_folder = 'data/tapvc_dataset'
images_folder = l(output_folder, 'images')

if not os.path.exists(output_folder):
    os.mkdir(output_folder)
if not os.path.exists(images_folder):
    os.mkdir(images_folder)

# move dir from original folder to folder
"""
for f in os.listdir(orig_folder):
    if not os.path.exists(l(folder, f)):
        shutil.move(l(orig_folder, f), folder)

"""

subdir = [(l(orig_folder, i), i) for i in os.listdir(orig_folder)]

for dic_dir in subdir:
    output_file = os.path.join(images_folder, dic_dir[1] + '.nii.gz')
    # 488 is no cubic image not os.path.exists(output_file)
    if not os.path.exists(output_file) and '488' != dic_dir[1]:
        print(dic_dir)
        dicom2nifti.dicom_series_to_nifti(dic_dir[0], output_file, reorient_nifti=True)



