import json
import os
from configs.Config_chd import get_config

c = get_config()

image_dir = c.data_dir

image_dict = {}
image_list = []

for i in os.listdir(image_dir):
    image_list.append(i)
    print(i)

image_dict['all_images'] = image_list

with open('tapvc_dict.json', 'w') as f:
    json.dump(image_dict, f)
    print('dict saved to json')

