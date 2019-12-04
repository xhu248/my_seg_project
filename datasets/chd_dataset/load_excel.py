# return a dictionary mapping file name to whether tapvc
# and create splits based on number of pvo
import pandas as pd
import os
import numpy as np


def load_excel(excel_path, do_random=False):
    df = pd.read_excel(excel_path)
    new_number = df['new number']
    pvo = df['pvo']
    tapvc_dict = {}
    pvo_list = []

    # mapping new number to whether pvo
    length = len(new_number)
    pvo_number = sum(pvo)
    if not do_random:
        for k in range(length):
            file_name = str(new_number[k]) + '.npy'
            tapvc_dict[file_name] = pvo[k]
    else:
        random_pvo = np.concatenate((np.ones(pvo_number), np.zeros(length - pvo_number)))
        np.random.shuffle(random_pvo)
        for k in range(length):
            file_name = str(new_number[k]) + '.npy'
            tapvc_dict[file_name] = random_pvo[k]

    return tapvc_dict
