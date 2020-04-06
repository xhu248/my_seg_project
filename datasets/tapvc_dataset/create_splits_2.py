import pickle
from utilities.file_and_folder_operations import subfiles
from datasets.tapvc_dataset.load_excel import load_excel

import os
import numpy as np

def create_splits(excel_path, output_dir, image_dir, do_balancement=False, fold_num=4):
    npy_files = subfiles(image_dir, suffix=".npy", join=False)
    pvo_list = []
    aug_list = []  # store types of augmentation
    non_pvo_list = []
    tapvc_dict = load_excel(excel_path)

    for file in npy_files:
        suffix = file.split("_")[1]
        aug_type = file.split("_")[0]

        if aug_type not in aug_list:
            aug_list.append(aug_type)

        assert suffix in tapvc_dict
        if suffix not in pvo_list and suffix not in non_pvo_list:
            if tapvc_dict[suffix] == 1:
                pvo_list.append(suffix)
            else:
                non_pvo_list.append(suffix)

    print('Number of pvo: %d, Number of non_pvo: %d ' %(len(pvo_list), len(non_pvo_list)))

    fold_postive_set = []
    fold_negative_set = []
    fold_postive_size = len(pvo_list) // 4
    fold_negative_size = len(non_pvo_list) // 4
    positive_set = pvo_list.copy()
    negative_set = non_pvo_list.copy()
    for k in range(fold_num):
        fold_pset = []
        fold_nset = []
        if k < fold_num - 1:
            for i in range(fold_postive_size):
                patient = np.random.choice(positive_set)
                positive_set.remove(patient)
                fold_pset.append(patient)

            for i in range(fold_negative_size):
                patient = np.random.choice(negative_set)
                negative_set.remove(patient)
                fold_nset.append(patient)
        else:
            fold_pset = positive_set
            fold_nset = negative_set
        fold_postive_set.append(fold_pset)
        fold_negative_set.append(fold_nset)

    splits = []
    for k in range(fold_num):
        split_dict = dict()
        valset = []
        trainset = []

        testset = fold_postive_set[k] + fold_negative_set[k]
        positive_set = pvo_list.copy()
        negative_set = non_pvo_list.copy()
        for item in fold_postive_set[k]:
            positive_set.remove(item)
        for item in fold_negative_set[k]:
            negative_set.remove(item)

        positive_val_size = len(positive_set) // 3
        negative_val_size = len(negative_set) // 3

        for i in range(positive_val_size):
            patient = np.random.choice(positive_set)
            positive_set.remove(patient)
            valset.append(patient)
        for i in range(negative_val_size):
            patient = np.random.choice(negative_set)
            negative_set.remove(patient)
            valset.append(patient)

        trainset = positive_set
        if do_balancement:
            positive_train_size = len(positive_set)
            negative_train_size = positive_train_size * 5
            for i in range(negative_train_size):
                patient = np.random.choice(negative_set)
                negative_set.remove(patient)
                trainset.append(patient)
        else:
            trainset = trainset + negative_set

        trainset = append_aug_files(trainset, aug_list)
        valset = append_aug_files(valset, aug_list)
        testset = append_aug_files(testset, aug_list)

        split_dict["train"] = trainset
        split_dict["val"] = valset
        split_dict["test"] = testset

        splits.append(split_dict)

    print('finished')

    with open(os.path.join(output_dir, 'splits.pkl'), 'wb') as f:
        pickle.dump(splits, f)


def get_number_from_file_list(file_list):
    number_list = []
    for file in file_list:
        number = file.split(".")[0]
        number_list.append(number)

    return number_list
# divide the processed .npy data into three parts, and do it four times

def append_aug_files(target_set, aug_list):
    new_set = []
    for item in target_set:
        for aug in  aug_list:
            new_item = aug + "_" + item
            new_set.append(new_item)

    return new_set

