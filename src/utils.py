import os
import random
import csv
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
from pathlib import Path


def load_data(data_path, usage, is_label=False, apply_normalization=False, 
              normal_strategy="z_value", stat_procedure="gpb", global_stats=None, 
              dtype=np.float32, verbose=False):
    r"""
    Open data using gdal, read it as an array and normalize it.

    Arguments:
            data_path (string): Full path including filename of the data source we wish to load.
            usage (string): Either "train", "validation", "inference".
            is_label (binary): If True then the layer is a ground truth (category index) and if
                                set to False the layer is a reflectance band.
            apply_normalization (binary): If true min/max normalization will be applied on each band.
            normal_strategy (str): Strategy for normalization. Either 'min_max'
                               or 'z_value'.
            stat_procedure (str): Procedure to calculate the statistics used in normalization.
                              Options:
                                    - 'lab': local tile over all bands.
                                    - 'gab': global over all bands.
                                    - 'lpb': local tile per band.
                                    - 'gpb': global per band.
            global_stats (dict): Optional dictionary containing the 'min', 'max', 'mean', and 'std' arrays 
                                 for each band. If not provided, these values will be calculated from the data.
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
                img = do_normalization(src.read(), normal_strategy, stat_procedure,
                                        bounds=(0, 1), clip_val=1, global_stats=global_stats)
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


def calculate_global_class_weights(src_dir, dataset_name, ignore_index=0, num_classes=14):
    """
    Calculate class weights for a dataset of labels with class imbalances.

    This function calculates the class weights based on the global class distribution
    over the entire dataset based on the inverse frequency ratio formula.

    Parameters:
    src_dir (str): The source directory where the dataset is located.
    dataset_name (str): The name of the dataset.
    ignore_index (int, optional): The class index to be ignored when calculating 
                                  the class weights. Defaults to 0.
    num_classes (int, optional): The total number of classes in the dataset. 
                                 Defaults to 14.

    Returns:
    numpy array: The class weights for the dataset as an array.
    """
    lbl_fnames = [Path(dirpath) / f 
                      for (dirpath, dirnames, filenames) in os.walk(Path(src_dir) / dataset_name) 
                      for f in filenames 
                      if f.endswith(".tif") and "mask" in f]
    lbl_fnames.sort()

    # Initialize the global counts for each class
    global_counts = np.zeros(num_classes)

    for lbl_fn in lbl_fnames:
         with rasterio.open(lbl_fn, "r") as src:
             lbl = src.read(1)
             mask = lbl != ignore_index
             unique, unique_counts = np.unique(lbl[mask], return_counts=True)

             # Add the unique_counts from this image to the global_counts
             for u, uc in zip(unique, unique_counts):
                 global_counts[u] += uc

    # Ignore the class specified by ignore_index when calculating the ratio and weight
    valid_indices = np.arange(num_classes) != ignore_index
    valid_counts = global_counts[valid_indices]
    ratio = valid_counts.astype(float) / np.sum(valid_counts)
    weights = (1. / ratio) / np.sum(1. / ratio)
    
    return weights


def split_dataset(src_dir, dataset_name, split_ratio, out_dir, seed=0):

    lbl_fnames = [Path(dirpath) / f
                  for (dirpath, dirnames, filenames) in os.walk(Path(src_dir) / dataset_name) 
                  for f in filenames 
                  if f.endswith(".tif") and "mask" in f]
    lbl_fnames.sort()

    total_samples = len(lbl_fnames)
    indices = np.arange(total_samples)
    split_index = int(total_samples * split_ratio)

    np.random.seed(seed)
    np.random.shuffle(indices)

    train_indices = indices[:split_index]
    train_lbl_fnames = [lbl_fnames[i] for i in train_indices]
    train_ids = [Path(fname).stem.split('.')[0].split('_')[1:] for fname in train_lbl_fnames]
    train_ids = ["_".join(id) for id in train_ids]

    val_indices = indices[split_index:]
    val_lbl_fnames = [lbl_fnames[i] for i in val_indices]
    val_ids = [Path(fname).stem.split('.')[0].split('_')[1:] for fname in val_lbl_fnames]
    val_ids = ["_".join(id) for id in val_ids]

    # Save train IDs and validation IDs to CSV files
    with open(Path(out_dir) / 'train_ids.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for id in train_ids:
            writer.writerow([id])
            
    with open(Path(out_dir) / 'val_ids.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for id in val_ids:
            writer.writerow([id])

    return train_ids, val_ids