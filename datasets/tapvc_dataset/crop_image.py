import torch.nn.functional as F
import torch
import os

import matplotlib.pyplot as plt
import numpy as np


def image_reshape(image, target_size=(256, 256)):
    shape = image.shape

    image = torch.tensor(image).reshape((1, 1, shape[0], shape[1]))
    reshape_image = F.interpolate(image, size=target_size, mode='nearest')
    reshape_image = reshape_image.squeeze().numpy()
    return reshape_image


# the thrshold is 0.2

def crop_slice(slice_orig):
    length = slice_orig.shape[0]
    # find the crop index in horizontal direction
    h_index = []
    for k in range(length - 1):
        max_0 = max(slice_orig[k])
        if max_0 > 0.2:
            h_index.append(k)
    upper_bound = max(h_index)
    lower_bound = min(h_index)
    slice_crop = slice_orig[lower_bound:upper_bound]

    # find the crop index in. vertical direction
    v_index = []
    for k in range(length - 1):
        max_0 = max(slice_orig[:, k])
        if max_0 > 0.2:
            v_index.append(k)
    upper_bound = max(v_index)
    lower_bound = min(v_index)
    slice_crop = slice_crop[:, lower_bound:upper_bound]

    # print(v_index)

    return slice_crop


# input is the original numpy array of tapvc image, whose shape is (512, 512, k)
# crop the 3d image slice by slice across the third dimension
# main function, call crop_slice and reshape_image
def crop_image(image):
    length = image.shape[2]
    image_crop = np.copy(image)
    for k in range(length):
        slice_orig = image[:, :, k]
        slice_crop = crop_slice(slice_orig)
        print(slice_crop.shape)
        slice_reshape = image_reshape(slice_crop)
        image_crop[:, :, k] = slice_reshape

    return image_crop

if __name__ == "__main__":
    crop_dir = "/home/xinronghu/tmp/tapvc_project/data/tapvc_dataset/cropped"
    processed_dir = "/home/xinronghu/tmp/tapvc_project/data/tapvc_dataset/preprocessed"

    if not os.path.exists(crop_dir):
        os.mkdir(crop_dir)
        print("Making", crop_dir)

    l = os.path.join
    files = [f for f in os.listdir(processed_dir)]

    for f in files:
        processed_path = l(processed_dir, f)
        cropped_path = l(crop_dir, f)
        images = np.load(processed_path)
        cropped_image = crop_image(images)
        print(f, cropped_image.shape)

        np.save(cropped_path, cropped_image)

    print("Cropping finished!")
