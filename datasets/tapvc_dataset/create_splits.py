import pickle
from utilities.file_and_folder_operations import subfiles
from datasets.tapvc_dataset.load_excel import load_excel

import os
import numpy as np


def create_splits(excel_path, output_dir, image_dir, do_balancement=False):
    npy_files = subfiles(image_dir, suffix=".npy", join=False)
    pvo_list = []
    non_pvo_list = []
    tapvc_dict = load_excel(excel_path)

    for file in npy_files:
        if file in tapvc_dict:
            if tapvc_dict[file] == 1:
                pvo_list.append(file)
            else:
                non_pvo_list.append(file)

    print('Number of pvo: %d, Number of non_pvo: %d'%(len(pvo_list), len(non_pvo_list)))

    valset_negative_size = len(pvo_list)*25 // 100
    testset_negative_size = len(pvo_list)*25 // 100
    trainset_negative_size = len(pvo_list) - valset_negative_size - testset_negative_size

    if do_balancement:
        trainset_positive_size = trainset_negative_size * 5
        valset_positive_size = valset_negative_size * 5
        testset_positive_size = testset_negative_size * 5
    else:
        valset_positive_size = len(non_pvo_list) * 25 // 100
        testset_positive_size = len(non_pvo_list) * 25 // 100
        trainset_positive_size = len(non_pvo_list) - valset_positive_size - testset_positive_size

    splits = []
    for split in range(0, 5):
        positive_set = non_pvo_list.copy()
        negative_set = pvo_list.copy()
        trainset = []
        valset = []
        testset = []
        for i in range(0, trainset_positive_size):
            patient = np.random.choice(positive_set)
            positive_set.remove(patient)
            trainset.append(patient[:-4])
        for i in range(0, trainset_negative_size):
            patient = np.random.choice(negative_set)
            negative_set.remove(patient)
            trainset.append(patient[:-4])

        for i in range(0, valset_positive_size):
            patient = np.random.choice(positive_set)
            positive_set.remove(patient)
            valset.append(patient[:-4])
        for i in range(0, valset_negative_size):
            patient = np.random.choice(negative_set)
            negative_set.remove(patient)
            valset.append(patient[:-4])

        for i in range(0, testset_positive_size):
            patient = np.random.choice(positive_set)
            positive_set.remove(patient)
            testset.append(patient[:-4])
        for i in range(0, testset_negative_size):
            patient = np.random.choice(negative_set)
            negative_set.remove(patient)
            testset.append(patient[:-4])
        split_dict = dict()
        split_dict['all'] = get_number_from_file_list(pvo_list) + get_number_from_file_list(non_pvo_list)
        split_dict['train'] = trainset
        split_dict['val'] = valset
        split_dict['test'] = testset

        splits.append(split_dict)

    with open(os.path.join(output_dir, 'splits.pkl'), 'wb') as f:
        pickle.dump(splits, f)


def get_number_from_file_list(file_list):
    number_list = []
    for file in file_list:
        number = file.split(".")[0]
        number_list.append(number)

    return number_list
# divide the processed .npy data into three parts, and do it four times