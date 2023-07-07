import numpy as np


def do_normalization(img, normal_strategy="z_value", bounds=(0, 1), clip_val=None, global_stats=True):
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
        global_stats(Binary): If set to 'True' it uses the global statistics for 
                              the whole dataset, otherwise statistics are calculated 
                              per band for each image chip separately.

    Returns:
        np.ndarray: Normalized image stack of size (C, H, W).

    Notes:
        - Most common bounds for satellite image processing would be (0, 1)
          or (0, 256).
        - Normalization statistics are calculated per band and for each image
          tile separately.
        - Notice the global statistics are hard-coded for our HLS dataset and only when
          we use 'global_stats'.
    """

    if normal_strategy not in ["min_max", "z_value"]:
        raise ValueError("Normalization strategy is not recognized.")

    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
        raise ValueError("Normalization bounds should be a tuple or list of length 2.")

    lower_bound, upper_bound = map(float, bounds)

    if normal_strategy == "min_max":
        img_mins = np.nanmin(img, axis=(1, 2))
        img_maxs = np.nanmax(img, axis=(1, 2))
        
        if clip_val is not None:
            img = np.clip(img, np.nanpercentile(img, clip_val),
                          np.nanpercentile(img, 100 - clip_val))

        normal_img = (upper_bound - lower_bound) * (img - img_mins[:, None, None]) / (
                img_maxs[:, None, None] - img_mins[:, None, None])

    elif normal_strategy == "z_value":
        if global_stats:
            img_means = np.array([377.9830716, 653.4058739, 702.1834053, 2573.849324, 2344.568628, 1432.163695] * 3)
            img_stds = np.array([171.5116587, 215.5407551, 364.3545396, 686.5730746, 769.6448444, 675.9192684] * 3)
        else:
            img_means = np.nanmean(img, axis=(1, 2))
            img_stds = np.nanstd(img, axis=(1, 2))
        normal_img = (img - img_means[:, None, None]) / img_stds[:, None, None]

        # shift the values to the positive range --> [1,+inf)
        min_vals = np.min(normal_img, axis=(1, 2))
        for i in range(normal_img.shape[0]):
            if min_vals[i] < 0:
                normal_img[i,:,:] -= min_vals[i]


    return normal_img