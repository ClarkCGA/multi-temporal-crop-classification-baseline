import os
import random
import numbers
import math
import itertools
import time
import numpy as np
import pandas as pd
import rasterio
import pickle
import torch
import matplotlib.pyplot as plt
from normalization import do_normalization


def load_data(data_path, usage, is_label=False, apply_normalization=False, dtype=np.float32, verbose=False):
    r"""
    Open data using gdal, read it as an array and normalize it.

    Arguments:
            data_path (string): Full path including filename of the data source we wish to load.
            usage (string): Either "train", "validation", "inference".
            is_label (binary): If True then the layer is a ground truth (category index) and if
                                set to False the layer is a reflectance band.
            apply_normalization (binary): If true min/max normalization will be applied on each band.
            dtype (np.dtype): Data type of the output image chips.
            verbose (binary): if set to true, print a screen statement on the loaded band.

    Returns:
            image: Returns the loaded image as a 32-bit float numpy ndarray.
    """

    # Inform user of the file names being loaded from the Dataset.
    if verbose:
        print('loading file:{}'.format(data_path))

    # open dataset using rasterio library.
    with rasterio.open(data_path, "r") as src:

        if is_label:
            if src.count != 1:
                raise ValueError("Expected Label to have exactly one channel.")
            img = src.read(1)

        else:
            meta = src.meta
            if apply_normalization:
                img = do_normalization(src.read(), bounds=(0, 1), clip_val=1)
                img = img.astype(dtype)
            else:
                img = src.read()
                img = img.astype(dtype)

    if usage in ["train", "validation"]:
        return img
    else:
        return img, meta

def load_pred(data_path, apply_normalization=False, dtype=np.float32):

def make_deterministic(seed=None, cudnn=True):
    """
    Sets the random seed for Python, NumPy, and PyTorch to a fixed value to ensure 
    reproducibility of results. Optionally, sets the seed for the CuDNN backend to 
    ensure reproducibility when training on a GPU.

    Args:
        seed (int): The seed value to use for setting the random seed (default: 1960).
        cudnn (bool): If True, sets the seed for the CuDNN backend to ensure 
            reproducibility when training on a GPU (default: True).
    """
    if seed is None:
        seed = int(time.time()) + int(os.getpid())
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if cudnn:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True