# return a dictionary mapping file name to whether tapvc
import pandas as pd
import os

def load_excel(file_path, image_dir):
    df = pd.read_excel('file_path')
    new_number = df['new number']
    pvo = df['pvo']
    pvo_dict = {}

    # mapping new number to whether pvo
    length = len(new_number)
    for k in length:
        file_name = new_number[k] + '.nii,gz'
        pvo_dict[file_name] = pvo[k]

    return pvo_dict