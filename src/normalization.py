import numpy as np


def do_normalization(img, normal_strategy, stat_procedure, nodata, clip_val=0, global_stats=None):
    """
    Normalize the input image pixels to a user-defined range based on the
    minimum and maximum statistics of each band and optional clip value.

    Args:
        img (np.ndarray): Stacked image bands with a dimension of (C, H, W).
        normal_strategy (str): Strategy for normalization. Either 'min_max'
                               or 'z_value'.
        stat_procedure (str): Procedure to calculate the statistics used in normalization.
                              Options:
                                    - 'lab': local tile over all bands.
                                    - 'gab': global over all bands.
                                    - 'lpb': local tile per band.
                                    - 'gpb': global per band.
        dtype (np.dtype): Data type of the output image chips.
        nodata (int): Value reserved to show nodata.
        clip_val (float): Defines how much of the distribution tails to be cut off.
                          Default is 0.
        global_stats (dict): Optional dictionary containing the 'min', 'max', 'mean', and 'std' arrays
                      for each band. If not provided, these values will be calculated from the data.

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
    
    if stat_procedure not in ["gpb", "lpb", "gab", "lab"]:
        raise ValueError("Statistics calculation strategy is not recognized.")
    
    if stat_procedure in ["gpb", "gab"]:
        if global_stats is None:
            raise ValueError("Global statistics must be provided for global normalization.")
        else:
            gpb_mins = np.array(global_stats['min'])
            gpb_maxs = np.array(global_stats['max'])
            gpb_means = np.array(global_stats['mean'])
            gpb_stds = np.array(global_stats['std'])
    
    # create a mask of nodata values and replace it with nan for computation.
    img_tmp = np.where(np.isin(img, nodata), np.nan, img)

    if normal_strategy == "min_max":
        
        if (clip_val is not None) and clip_val > 0:
            lower_percentiles = np.nanpercentile(img_tmp, clip_val, axis=(1, 2))
            upper_percentiles = np.nanpercentile(img_tmp, 100 - clip_val, axis=(1, 2))
            for b in range(img.shape[0]):
                img[b] = np.clip(img[b], lower_percentiles[b], upper_percentiles[b])

        if stat_procedure == "gpb":
            if np.any(gpb_maxs == gpb_mins):
                raise ValueError("Division by zero detected: some bands have identical min and max values.")
            
            normal_img = (img - gpb_mins[:, None, None]) / (gpb_maxs[:, None, None] - gpb_mins[:, None, None])
            normal_img = np.clip(normal_img, 0, 1)
        
        elif stat_procedure == "gab":
            gab_min = np.mean(gpb_mins)           
            gab_max = np.mean(gpb_maxs)
            if gab_max == gab_min:
                raise ValueError("Division by zero detected: some images have identical min and max values.")
            
            normal_img = (img - gab_min) / (gab_max - gab_min)
            normal_img = np.clip(normal_img, 0, 1)
        
        elif stat_procedure == "lab":
            lab_min = np.nanmin(img_tmp)
            lab_max = np.nanmax(img_tmp)
            if lab_max == lab_min:
                raise ValueError("Division by zero detected: some images have identical min and max values.")
            
            normal_img = (img - lab_min) / (lab_max - lab_min)
            normal_img = np.clip(normal_img, 0, 1)
        
        else:    # stat_procedure == "lpb"
            lpb_mins = np.nanmin(img_tmp, axis=(1, 2))
            lpb_maxs = np.nanmax(img_tmp, axis=(1, 2))
            if np.any(lpb_maxs == lpb_mins):
                raise ValueError("Division by zero detected: some bands have identical min and max values.")

            normal_img = (img - lpb_mins[:, None, None]) / (lpb_maxs[:, None, None] - lpb_mins[:, None, None])
            normal_img = np.clip(normal_img, 0, 1)

    elif normal_strategy == "z_value":
        
        if stat_procedure == "gpb":
            normal_img = (img - gpb_means[:, None, None]) / gpb_stds[:, None, None]
        
        elif stat_procedure == "gab":
            gab_mean = np.mean(gpb_means)

            num_pixels_per_band = img.shape[1] * img.shape[2]
            squared_std_values = gpb_stds ** 2
            squared_std_values *= num_pixels_per_band
            sum_squared_std = np.sum(squared_std_values)
            total_samples = num_pixels_per_band * len(gpb_stds)
            gab_std = np.sqrt(sum_squared_std / total_samples)
            
            normal_img = (img - gab_mean) / gab_std
        
        elif stat_procedure == "lpb":
            img_means = np.nanmean(img_tmp, axis=(1, 2))
            img_stds = np.nanstd(img_tmp, axis=(1, 2))
            normal_img = (img - img_means[:, None, None]) / img_stds[:, None, None]
        
        elif stat_procedure == "lab":
            img_mean = np.nanmean(img_tmp)
            img_std = np.nanstd(img_tmp)
            normal_img = (img - img_mean) / img_std

    # # shift the values to the positive range --> [0,+inf)
    # min_vals = np.nanmin(normal_img, axis=(1, 2))
    # for i in range(normal_img.shape[0]):
    #     if min_vals[i] < 0:
    #         normal_img[i,:,:] -= min_vals[i]

    return normal_img
