from collections import defaultdict

from medpy.io import load
import os
from configs.Config_chd import get_config
import numpy as np
from datasets.three_dim.data_augmentation import get_transforms

if __name__ == "__main__":
    c = get_config()
    data_path = c.data_dir
    files = os.listdir(data_path)
    file = files[1]

    image = np.load(os.path.join(data_path, file))

    transform = get_transforms('train', target_size=(256, 256, 256))
    new_image = transform(image)
    print(image)
