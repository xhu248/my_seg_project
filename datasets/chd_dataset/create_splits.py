import pickle
from utilities.file_and_folder_operations import subfiles
from datasets.chd_dataset.load_excel import load_excel

import os
import numpy as np


def create_splits(excel_path, output_dir, image_dir):
    npy_files = subfiles(image_dir, suffix=".npy", join=False)
    pvo_list = []
    non_pvo_list = []
    tapvc_dict = load_excel(excel_path, image_dir)

    for file in npy_files:
        if tapvc_dict[file] == 1:
            pvo_list.append(file)
        else:
            non_pvo_list.append(file)

    print('Number of pvo: %d, Number of non_pvo: %d'%(len(pvo_list), len(non_pvo_list)))

    trainset_size = len(npy_files)*50//100
    valset_size = len(npy_files)*25//100
    testset_size = len(npy_files)*25//100

    splits = []
    for split in range(0, 5):
        image_list = npy_files.copy()
        trainset = []
        valset = []
        testset = []
        for i in range(0, trainset_size):
            patient = np.random.choice(image_list)
            image_list.remove(patient)
            trainset.append(patient[:-4])
        for i in range(0, valset_size):
            patient = np.random.choice(image_list)
            image_list.remove(patient)
            valset.append(patient[:-4])
        for i in range(0, testset_size):
            patient = np.random.choice(image_list)
            image_list.remove(patient)
            testset.append(patient[:-4])
        split_dict = dict()
        split_dict['train'] = trainset
        split_dict['val'] = valset
        split_dict['test'] = testset

        splits.append(split_dict)

    with open(os.path.join(output_dir, 'splits.pkl'), 'wb') as f:
        pickle.dump(splits, f)

# divide the processed .npy data into three parts, and do it four times