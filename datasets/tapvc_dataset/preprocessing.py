#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

from medpy.io import load
import os
import numpy as np

import pandas as pd


from utilities.file_and_folder_operations import subfiles
import torch
import torch.nn.functional as F

def reshape_array(numpy_array):
    shape = numpy_array.shape[1]
    slice_img = numpy_array[:, :, :, 0].reshape(1, 2, shape, shape)
    slice_len = np.shape(numpy_array)[3]
    for k in range(1, slice_len):
        slice_array = numpy_array[:, :, :, k].reshape(1, 2, shape, shape)
        slice_img = np.concatenate((slice_img, slice_array))

    return slice_img


def preprocess_data(root_dir):
    image_dir = os.path.join(root_dir, 'images')
    # label_dir = os.path.join(root_dir, 'labels')
    output_dir = os.path.join(root_dir, 'preprocessed')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    nii_files = subfiles(image_dir, suffix=".nii.gz", join=False)

    for f in nii_files:
        file_dir = os.path.join(output_dir, f.split('.')[0]+'.npy')
        if not os.path.exists(file_dir) and '362' not in f and '387' not in f and '411' not in f:
            image, _ = load(os.path.join(image_dir, f))
            # normalize images
            image = (image - image.min()) / (image.max() - image.min())
            target_shape = (image.shape[0]//2, image.shape[1]//2, image.shape[2]//2)   # reduce x, y , z by 2

            image_tensor = torch.from_numpy(image[None, None]).to(device)
            new_image = F.interpolate(image_tensor, size=target_shape, mode='trilinear')
            new_image = new_image.cpu().numpy().squeeze()
            print(new_image.shape)
            # image = reshape(image, append_value=0, new_shape=(64, 64, 64))
            # label = reshape(label, append_value=0, new_shape=(64, 64, 64))

            np.save(os.path.join(output_dir, f.split('.')[0] + '.npy'), new_image)

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    npy_files = subfiles(output_dir, suffix='.npy', join=False)
    for f in npy_files:
        numpy_array = np.load(os.path.join(output_dir, f)).squeeze()
        if numpy_array.shape != target_shape:
            image_tensor = torch.from_numpy(numpy_array[None, None]).to(device)
            new_image = F.interpolate(image_tensor, size=target_shape, mode='trilinear')
            new_image = new_image.cpu().numpy().squeeze()
            print(new_image.shape)
            np.save(os.path.join(output_dir, f), new_image)
            print('saving new image: ', f)
        else:
            np.save(os.path.join(output_dir, f), numpy_array)
            print('saving squeezed image', f, numpy_array.shape)

            # result = np.stack((image, label))
            # result = reshape_array(image)

            np.save(os.path.join(output_dir, f.split('.')[0] + '.npy'), image)
            print(f)
    """

def remove_nii_file(root_dir, excel_path):
    image_dir = os.path.join(root_dir, 'images')
    df = pd.read_excel(excel_path)
    tapvc_type = df["tapvc_type"]
    new_number = df["new number"]

    length = len(new_number)
    supre_list = []
    for k in range(length):
        if tapvc_type[k] == 0:
            supre_patient_number = str(new_number[k])
            nii_file = supre_patient_number + ".nii.gz"
            supre_list.append(nii_file)

    files_list = os.listdir(image_dir)
    for f in files_list:
        if f not in supre_list:
            file_path = os.path.join(image_dir, f)
            os.remove(file_path)
            print("remove file: ", f)


if __name__ == "__main__":
    root_dir = "/home/xinronghu/tmp/tapvc_project/data/tapvc_dataset"
    excel_path = '/home/xinronghu/tmp/tapvc_project/data/tapvc_dataset/pvo_classification.xlsx'

    # remove_nii_file(root_dir, excel_path)
    preprocess_data(root_dir)

