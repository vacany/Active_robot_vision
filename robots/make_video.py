import os.path

import matplotlib.pyplot as plt
import numpy as np
import glob
from PIL import Image

rgb_paths = sorted(glob.glob(os.path.expanduser("~") + '/data/robots/raw_rgb/*.png'))
seg_paths = sorted(glob.glob(os.path.expanduser("~") + '/data/robots/hrnet_seg/*.png'))

for idx in range(len(rgb_paths)):
    print(idx)
    rgb = Image.open(rgb_paths[idx]).resize((1280, 720))
    seg = Image.open(seg_paths[idx]).resize((1280, 720))

    fig, ax = plt.subplots(1,2, figsize=(10, 4), dpi=300)


    ax[0].imshow(np.asarray(rgb))
    ax[0].set_title('RGB image from Realsense')
    ax[1].imshow(np.asarray(seg))
    ax[1].set_title('Segmentation from HRNet based on RGB')

    # plt.show()
    plt.savefig(os.path.expanduser("~") + '/patrik_data/delft_toy/to_show/' + os.path.basename(rgb_paths[idx]))
    plt.close()

