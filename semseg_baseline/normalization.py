import numpy as np


def do_normalization(img, normal_strategy="min_max", bounds=(0, 1), clip_val=None):
    """
    Normalize the input image pixels to a user-defined range based on the
    minimum and maximum statistics of each band and optional clip value.

    Args:
        img (np.ndarray): Stacked image bands with a dimension of (C, H, W).
        normal_strategy (str): Strategy for normalization. Either 'min_max'
                               or 'z_value'.
        bounds (tuple): Lower and upper bound of rescaled values applied to all
                        the bands in the image. Default is (0, 1).
        clip_val (float): Defines how much of the distribution tails to be cut off.
                          Default is None.

    Returns:
        np.ndarray: Normalized image stack of size (C, H, W).

    Notes:
        - Most common bounds for satellite image processing would be (0, 1)
          or (0, 256).
        - Normalization statistics are calculated per band and for each image
          tile separately.
    """

    if normal_strategy not in ["min_max", "z_value"]:
        raise ValueError("Normalization strategy is not recognized.")

    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
        raise ValueError("Normalization bounds should be a tuple or list of length 2.")

    lower_bound, upper_bound = map(float, bounds)

    img_mins = np.nanmin(img, axis=(1, 2))
    img_maxs = np.nanmax(img, axis=(1, 2))

    if normal_strategy == "min_max":
        if clip_val is not None:
            img = np.clip(img, np.nanpercentile(img, clip_val),
                          np.nanpercentile(img, 100 - clip_val))

        normal_img = (upper_bound - lower_bound) * (img - img_mins[:, None, None]) / (
                img_maxs[:, None, None] - img_mins[:, None, None])

    elif normal_strategy == "z_value":
        img_means = np.nanmean(img, axis=(1, 2))
        img_stds = np.nanstd(img, axis=(1, 2))
        normal_img = (img - img_means[:, None, None]) / img_stds[:, None, None]

    return normal_img
