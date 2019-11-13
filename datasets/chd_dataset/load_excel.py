# return a dictionary mapping file name to whether tapvc
# and create splits based on number of pvo
import pandas as pd
import os

def load_excel(excel_path, image_dir):
    df = pd.read_excel(excel_path)
    new_number = df['new number']
    pvo = df['pvo']
    tapvc_dict = {}
    pvo_list = []

    # mapping new number to whether pvo
    length = len(new_number)
    for k in range(length):
        file_name = str(new_number[k]) + '.npy'
        tapvc_dict[file_name] = pvo[k]

    return tapvc_dict
