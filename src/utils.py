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
from torch.utils.data import DataLoader
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


def get_labels_distribution(dataset, num_classes=14, ignore_class=0):
    labels_count = torch.zeros(num_classes)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for _, label in dataloader:
        unique, counts = torch.unique(label, return_counts=True)
        for u, c in zip(unique, counts):
            if u != ignore_class:
                labels_count[u] += c

    return labels_count


def plot_labels_distribution(labels_count, num_classes=14, ignore_class=0):
    labels = list(range(num_classes))
    labels.remove(ignore_class)

    plt.bar([str(i) for i in labels], labels_count[labels].numpy())
    plt.xlabel("Class Label")
    plt.ylabel("Frequency")
    plt.title("Class Distribution (ignoring class {})".format(ignore_class))
    plt.show()


def pickle_dataset(dataset, file_path):
    try:
        with open(file_path, "wb") as fp:
            pickle.dump(dataset, fp)
        print(f"Dataset pickled and saved to {file_path}")
    except OSError as e:
        print(f"Error: could not open file {file_path}: {e}")
    except pickle.PickleError as e:
        print(f"Error: could not pickle dataset: {e}")


def load_pickled_dataset(file_path):
    """
    Load pickled dataset from file path.
    """
    dataset = pd.read_pickle(file_path)
    return dataset


def show_random_patches(dataset, sample_num, rgb_bands=(3, 2, 1)):
    """
    Plots a user-defined number of image chips and the corresponding labels.
    Arguments:
    dataset (Dataset object) : Loaded custom dataset.
    sample_num (int) : Number of pairs of image chips and their corresponding 
        labels to be plotted.
    rgb_bands (tuple of int) : List of the rgb bands for visualization indexed 
        from zero.
    Note: The order of the input bands is as follows:
          bands = {0 : "KLS-L band 2 (Blue)",
                   1 : "HLS-L band 3 (Green)",
                   2 : "HLS-L band 4 (Red)",
                   3 : "HLS-L band 5 (NIR)"}
    """
    # Make a deep copy of the dataset for static augmentation.
    static_dataset = list(dataset)

    if len(rgb_bands) != 3 or any(not isinstance(b, int) for b in rgb_bands) or not (1 <= sample_num <= len(static_dataset)):
        del static_dataset
        raise ValueError("'sample_num' or 'rgb_bands' are not properly defined")

    # Select random samples
    sample_indices = np.random.choice(len(static_dataset), size=sample_num, replace=False)

    fig, axs = plt.subplots(nrows=sample_num, ncols=2, figsize=(16, sample_num * 16 / 2), squeeze=False)

    for i, sample_index in enumerate(sample_indices):
        # Get RGB data and normalize if necessary
        rgb_data = static_dataset[sample_index][0][list(rgb_bands),:,:].permute(1, 2, 0)
        if rgb_data.max() > 1:
            rgb_data = rgb_data.int()

        axs[i, 0].set_title(f'Image Patch #{sample_index}')
        axs[i, 0].imshow(rgb_data)

        label_data = static_dataset[sample_index][1]
        im = axs[i, 1].imshow(label_data)
        axs[i, 1].set_title(f'Label Patch #{sample_index}')
        
        # Adding colorbar to each label patch
        plt.colorbar(im, ax=axs[i, 1])

    plt.tight_layout()
    plt.show()
    del static_dataset