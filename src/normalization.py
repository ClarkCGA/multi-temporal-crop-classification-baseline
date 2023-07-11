import numpy as np


def do_normalization(img, normal_strategy, stat_procedure, bounds=(0, 1), 
                     nodata=None, clip_val=None, global_stats=None):
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
        bounds (tuple): Lower and upper bound of rescaled values applied to all
                        the bands in the image. Default is (0, 1).
        nodata (int): Value reserved to show nodata.
        clip_val (float): Defines how much of the distribution tails to be cut off.
                          Default is None.
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
    
    if global_stats is None:
        gpb_mins = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0] * 3)
        gpb_maxs = np.array([271.0, 315.0, 464.0, 786.0, 869.0, 775.0] * 3)
        gpb_means = np.array([377.9830716, 653.4058739, 702.1834053, 2573.849324, 2344.568628, 1432.163695] * 3)
        gpb_stds = np.array([171.5116587, 215.5407551, 364.3545396, 686.5730746, 769.6448444, 675.9192684] * 3)
    else:
        gpb_mins = global_stats['min']
        gpb_maxs = global_stats['max']
        gpb_means = global_stats['mean']
        gpb_stds = global_stats['std']

    
    # create a mask of nodata values and replace it with nan for computation.
    if nodata:
        mask = img == nodata
        img[mask] = np.nan

    if normal_strategy == "min_max":
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError("Normalization bounds should be a tuple or list of length 2.")

        lower_bound, upper_bound = map(float, bounds)

        if stat_procedure == "gpb":
            normal_img = (upper_bound - lower_bound) * (img - gpb_mins[:, None, None]) / (
                gpb_maxs[:, None, None] - gpb_mins[:, None, None])
        
        elif stat_procedure == "gab":
            gab_min = np.mean(gpb_mins)
            gab_max = np.mean(gpb_maxs)
            normal_img = (upper_bound - lower_bound) * (img - gab_min) / (gab_max - gab_min)
        
        elif stat_procedure == "lab":
            if clip_val is not None:
                img = np.clip(img, np.nanpercentile(img, clip_val), 
                              np.nanpercentile(img, 100 - clip_val))
            
            lab_min = np.nanmin(img)
            lab_max = np.nanmax(img)
            normal_img = (upper_bound - lower_bound) * (img - lab_min) / (lab_max - lab_min)
        
        elif stat_procedure == "lpb":
            if clip_val is not None:
                lower_percentiles = np.nanpercentile(img, clip_val, axis=(1, 2))
                upper_percentiles = np.nanpercentile(img, 100 - clip_val, axis=(1, 2))
                for b in range(img.shape[0]):
                    img[b] = np.clip(img[b], lower_percentiles[b], upper_percentiles[b])
            
            lpb_mins = np.nanmin(img, axis=(1, 2))
            lpb_maxs = np.nanmax(img, axis=(1, 2))

            normal_img = (upper_bound - lower_bound) * (img - lpb_mins[:, None, None]) / (
                lpb_maxs[:, None, None] - lpb_mins[:, None, None])


    elif normal_strategy == "z_value":
        
        if stat_procedure == "gpb":
            normal_img = (img - gpb_means[:, None, None]) / gpb_stds[:, None, None]
        
        elif stat_procedure == "gab":
            gpb_mean = np.mean(gpb_means)
            gpb_std = np.mean(gpb_stds)
            normal_img = (img - gpb_mean) / gpb_std
        
        elif stat_procedure == "lpb":
            img_means = np.nanmean(img, axis=(1, 2))
            img_stds = np.nanstd(img, axis=(1, 2))
            normal_img = (img - img_means[:, None, None]) / img_stds[:, None, None]
        
        elif stat_procedure == "lab":
            img_mean = np.nanmean(img)
            img_std = np.nanstd(img)
            normal_img = (img - img_mean) / img_std

    # shift the values to the positive range --> [0,+inf)
    min_vals = np.nanmin(normal_img, axis=(1, 2))
    for i in range(normal_img.shape[0]):
        if min_vals[i] < 0:
            normal_img[i,:,:] -= min_vals[i]
        
    if nodata:
        normal_img[np.isnan(normal_img)] = nodata

    return normal_img
