import matplotlib
import matplotlib.pyplot as plt

import os
import numpy as np
import torch
import torch.nn.functional as F
from configs.Config_tapvc import get_config
from utilities.file_and_folder_operations import subfiles

if __name__ == '__main__':
    c = get_config()

    base_dir =os.path.join(c.data_root_dir, 'CHD_segmentation_dataset')
    original_image = os.path.join(base_dir, "preprocessed/")
    pred_dir = os.path.join(c.base_dir, c.dataset_name
                                             + '_' + str(
        c.batch_size) + c.cross_vali_result_all_dir + '_20190425-213808')
    

    test_num = c.dataset_name + '_006'
    image_dir = os.path.join(pred_dir, 'results', 'pred_' + test_num + '.npy')


    # all_image = np.load(image_dir)[25]


    plt.figure(1)
    for i in range(np.shape(all_image)[0]):
        plt.subplot(1,3,i+1)
        plt.imshow(train_image[i], cmap='gray')
        if i == 0:
            plt.xlabel('original image')
        elif i == 1:
            plt.xlabel('label image')
        else:
            plt.xlabel('segmented image')

    if not os.path.exists(os.path.join(pred_dir, 'images')):
        os.makedirs(os.path.join(pred_dir, 'images'))

    plt.savefig(os.path.join(pred_dir, 'images') + '/_006_25.jpg')
    plt.show()

"""
    n = 0
    k = 10
    chd_files = subfiles(os.path.join(c.data_root_dir, 'CHD_segmentation_dataset/preprocessed'),
                         suffix='.npy', join=True)
    org_files = subfiles(os.path.join(c.data_root_dir, 'tapvc_dataset/preprocessed'),
                         suffix='.npy', join=True)
    seg_files = subfiles(c.seg_dir, suffix='.npy', join=True)

    ############ original image and target ########################
    file = org_files[n]
    seg_file = seg_files[n]
    chd_file = chd_files[n]
    data = np.load(file)
    seg_data = np.load(seg_file)
    chd_data = np.load(chd_file)
    print(data.max())
    print(chd_data.max())
    print(seg_data.max())
    image = data[:, :, k]
    seg_image = seg_data[:, :, k]
    chd_image = chd_data[k, 0]
    chd_label = chd_data[k, 1]


    ############ down scale using interpolation ########################
    
    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.title('image:%d,  slice:%d, original image' % (n, k))
    plt.imshow(image, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.title('image:%d,  slice:%d, after segmentation' % (n, k))
    plt.imshow(seg_image, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.title('image:%d,  slice:%d, chd image' % (n, k))
    plt.imshow(chd_image, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.title('image:%d,  slice:%d, chd label' % (n, k))
    plt.imshow(chd_label, cmap='gray')
    plt.show()
    
"""

"""
    ############ down scale using max-pooling ########################
    file_64 = pred_32_files[n]
    pred_64 = np.load(os.path.join(c.stage_1_dir_32, file_64))[:, 0:8]
    target_64 = np.load(os.path.join(c.stage_1_dir_32, file_64))[:, 8:9]

    pred_64 = torch.tensor(pred_64).float()
    target_64 = torch.tensor(target_64).long()

    # 32*32 image and target
    data_32 = np.load(os.path.join(c.scaled_image_32_dir, 'ct_1010_image.npy'))
    image_32 = data_32[:, 0]
    target_32 = data_32[:, 1]

    soft_max = F.softmax(pred_64[k:k + 1], dim=1)
    cf_img = torch.max(soft_max, 1)[0].numpy()
    pred_img = torch.argmax(soft_max, dim=1)

    # plot target
    plt.figure(2)
    plt.subplot(2, 2, 1)
    plt.title('image:%d,  slice:%d, confidence' % (n, k))
    plt.imshow(cf_img[0], cmap='gray')
    plt.subplot(2, 2, 2)
    plt.title('image:%d,  slice:%d, target' % (n, k))
    plt.imshow(target_64[k][0], cmap='gray')
    plt.subplot(2, 2, 3)
    plt.title('image:%d,  slice:%d, pred_image' % (n, k))
    plt.imshow(pred_img[0], cmap='gray')
    plt.show()

    plt.figure(3)
    plt.subplot(1, 2, 1)
    plt.title('image:%d,  slice:%d, original image' % (n, k))
    plt.imshow(image_32[k], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('image:%d,  slice:%d, original target' % (n, k))
    plt.imshow(target_32[k], cmap='gray')
    plt.show()

"""

