#@title image_augmentation.py

from skimage import transform as trans
import numpy as np
import numpy.ma as ma
import cv2
import random
from collections.abc import Sequence
from scipy.ndimage import rotate


def center_rotate(img, label, degree):
    r"""
    Synthesize a new pair of image, label chips by rotating the input chip around its center.
    Arguments:
            img (ndarray) -- Stacked image bands with a dimension of (H,W,C).
            label (ndarray) -- Ground truth layer with a dimension of (H,W).
            degree (tuple or list) -- If the  passed argument has exactly two elements then they
                                      act as a bound on the possible range of values to be used for rotation.
                                      If number of elements is more than two then one element is chosen
                                      randomly as the rotation degree.
    Returns:
        img -- A numpy array of rotated variables or brightness value.
        label -- A numpy array of rotated ground truth.
    """

    # Validate input parameters
    if not isinstance(img, np.ndarray) or not isinstance(label, np.ndarray):
        raise ValueError("img and label must be numpy arrays.")
    if img.ndim != 3:
        raise ValueError("img must have dimensions (H, W, C).")
    if label.ndim != 2:
        raise ValueError("label must have dimensions (H, W).")
    if not any(isinstance(degree, t) for t in (tuple, list)):
        raise ValueError("Degree must be a tuple or a list.")

    # And draw a random degree between the bounds from a uniform distribution.
    if len(degree) == 2:
        rotation_degree = random.uniform(degree[0], degree[1])
    elif len(degree) > 2:
        rotation_degree = random.choice(degree)
    else:
        raise ValueError("Parameter degree needs at least two elements.")

    # Get the spatial dimensions of the image.
    h, w = label.shape

    # Determine the image center.
    center = (w // 2, h // 2)

    # Grab the rotation matrix.
    # Third arg --> scale: Isotropic scale factor.
    rot_matrix = cv2.getRotationMatrix2D(center, rotation_degree, 1.0)

    # perform the actual rotation on image and label.
    img = cv2.warpAffine(img, rot_matrix, (w, h))
    label = cv2.warpAffine(label, rot_matrix, (w, h))

    # Round all pixel values greater than 0.5 to 1 and assign zero to the rest.
    label = np.rint(label)

    return img, label


# use scipy package if there is issue installing opencv
def rotate_image_and_label(image, label, angle):
    """
    Applies rotation augmentation to an image patch and label.

    Args:
        image (numpy array) : The input image patch as a numpy array.
        label (numpy array) : The corresponding label as a numpy array.
        angle (list of floats) : If the list has exactly two elements they will
            be considered the lower and upper bounds for the rotation angle
            (in degrees) respectively. If number of elements are bigger than 2,
            then one value is chosen randomly as the rotation angle.

    Returns:
        A tuple containing the rotated image patch and label as numpy arrays.
    """
    if isinstance(angle, tuple) or isinstance(angle, list):
        if len(angle) == 2:
            rotation_degree = random.uniform(angle[0], angle[1])
        elif len(angle) > 2:
            rotation_degree = random.choice(angle)
        else:
            raise ValueError("Parameter angle needs at least two elements.")
    else:
        raise ValueError(
            "Rotation bound param for augmentation must be a tuple or list."
        )

    # Apply rotation augmentation to the image patch
    rotated_image = rotate(input=image, angle=rotation_degree, axes=(1,0),
                           reshape=False, mode='reflect')

    # Apply rotation augmentation to the label
    rotated_label = rotate(input=label, angle=rotation_degree, axes=(1,0),
                           reshape=False, mode='nearest')

    # Return the rotated image patch and label as a tuple
    return rotated_image.copy(), rotated_label.copy()


def flip(img, label, flip_type):
    r"""
    Synthesize a new pair of image, label chips by flipping the input chips around a user defined axis.

    Arguments:
            img (ndarray) -- Concatenated variables or brightness value with a dimension of (H,W,C)
            label (ndarray) -- Ground truth with a dimension of (H,W)
            flip_type (list) -- A flip type based on the choice of axis.
                                Provided transformation are:
                                    1) 'v_flip', vertical flip
                                    2) 'h_flip', horizontal flip
                                    3) 'd_flip', diagonal flip
    Returns:
            img -- A numpy array of flipped variables or brightness value.
            label --A numpy array of flipped labeled reference (ground truth).
    """

    if not isinstance(img, np.ndarray) or not isinstance(label, np.ndarray):
        raise ValueError("img and label must be numpy arrays.")
    if img.ndim != 3:
        raise ValueError("img must have dimensions (H, W, C).")
    if label.ndim != 2:
        raise ValueError("label must have dimensions (H, W).")
    if not isinstance(flip_type, str):
        raise ValueError("Flip type must be a string.")

    # Horizontal flip
    if flip_type == "h_flip":
        img = np.flip(img, 0)
        label = np.flip(label, 0)

    # Vertical flip
    elif flip_type == "v_flip":
        img = np.flip(img, 1)
        label = np.flip(label, 1)

    # Diagonal flip
    elif flip_type == "d_flip":
        img = np.transpose(img, axes=(1, 0))
        label = np.transpose(label, axes=(1, 0))

    else:
        raise ValueError("Unsupported flip type. Valid options are: 'h_flip', 'v_flip', 'd_flip'.")

    return img.copy(), label.copy()


def re_scale(img, label, scale=(0.75, 1.5), crop_strategy="center"):
    r"""
    Synthesize a new pair of image, label chips by rescaling the input chips.

    Arguments:
            img (ndarray) -- Image chip with a dimension of (H,W,C).
            label (ndarray) -- Reference annotation layer with a dimension of (H,W).
            scale (tuple or list) -- A range of scale ratio.
            crop_strategy (str) -- decides whether to crop the rescaled image chip randomly
                                   or at the center.
    Returns:
           Tuple[np.ndarray, np.ndarray] including:
            resampled_img -- A numpy array of rescaled variables or brightness values in the
                             same size as the input chip.
            resampled_label --A numpy array of flipped ground truth in the same size as input.
    """

    if not isinstance(img, np.ndarray) or not isinstance(label, np.ndarray):
        raise ValueError("img and label must be numpy arrays.")
    if img.ndim != 3:
        raise ValueError("img must have dimensions (H, W, C).")
    if label.ndim != 2:
        raise ValueError("label must have dimensions (H, W).")

    h, w, c = img.shape

    if isinstance(scale, Sequence):
        resize_h = round(random.uniform(scale[0], scale[1]) * h)
        resize_w = resize_h
    else:
        raise Exception('Wrong scale type!')

    assert crop_strategy in ["center", "random"], "'crop_strategy' is not recognized."

    # We are using a bi-linear interpolation by default for resampling.
    # When output image size is zero then the output size is calculated based on fx and fy.
    resampled_img = trans.resize(img, (resize_h, resize_w), preserve_range=True)
    resampled_label = trans.resize(label, (resize_h, resize_w), preserve_range=True)

    if crop_strategy == "center":
        x_off = max(0, abs(resize_h - h) // 2)
        y_off = max(0, abs(resize_w - w) // 2)
    elif crop_strategy == "random":
        x_off = random.randint(0, max(0, abs(resize_h - h)))
        y_off = random.randint(0, max(0, abs(resize_w - w)))

    canvas_img = np.zeros((h, w, c), dtype=img.dtype)
    canvas_label = np.zeros((h, w), dtype=label.dtype)

    if resize_h > h and resize_w > w:
        canvas_img = resampled_img[x_off: x_off + min(h, resize_h), y_off: y_off + min(w, resize_w), :]
        canvas_label = resampled_label[x_off: x_off + min(h, resize_h), y_off: y_off + min(w, resize_w)]
        canvas_label = np.rint(canvas_label)

    elif resize_h < h and resize_w < w:
        canvas_img[x_off: x_off + resize_h, y_off: y_off + resize_w] = resampled_img
        canvas_label[x_off: x_off + resize_h, y_off: y_off + resize_w] = resampled_label

    return canvas_img, canvas_label


def shift_brightness(img, gamma_range=(0.2, 2.0), shift_subset=(4, 4, 4), patch_shift=False):
    """
    Shift image brightness through gamma correction

    Params:

        img (ndarray): Concatenated variables or brightness value with a dimension of (H, W, C)
        gamma_range (tuple): Range of gamma values
        shift_subset (tuple): Number of bands or channels for each shift
        patch_shift (bool): Whether apply the shift on small patches

     Returns:

        ndarray, brightness shifted image

    """
    c_start = 0
    for i in shift_subset:
        gamma = random.triangular(gamma_range[0], gamma_range[1], 1)
        if patch_shift:
            # shift on patch
            # generate mask - random rotate or/and rescale

            h, w, _ = img.shape
            rotMtrx = cv2.getRotationMatrix2D(center=(random.randint(0, h), random.randint(0, w)),
                                              angle=random.randint(0, 90),
                                              scale=random.uniform(1, 2))
            mask = cv2.warpAffine(img[:, :, c_start:c_start + i], rotMtrx, (w, h))
            mask = np.where(mask, 0, 1)
            # apply mask
            img_ma = ma.masked_array(img[:, :, c_start:c_start + i], mask=mask)
            img[:, :, c_start:c_start + i] = ma.power(img_ma, gamma)
            # default extra step -- shift on image
            gamma_full = random.triangular(0.5, 1.5, 1)
            img[:, :, c_start:c_start + i] = np.power(img[:, :, c_start:c_start + i], gamma_full)

        else:
            img[:, :, c_start:c_start + i] = np.power(img[:, :, c_start:c_start + i], gamma)
        c_start += i

    return img


def gaussian_blur(img, kernel_size):
    """
        Apply Gaussian blur to the input image.

        Args:
            img (np.ndarray): Input image as a NumPy array.
            kernel_size (int): Size of the Gaussian kernel.

        Returns:
            np.ndarray: Blurred image as a NumPy array.

        Note:
            The sigmaX parameter specifies the standard deviation of the Gaussian
            kernel along the x-axis, and if set to 0, OpenCV automatically computes
            it based on the kernel size using the formula:
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8.
        """
    # When sigma=0, it is computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`
    aug_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=0)

    return aug_img


def adjust_brightness(img, value=-0.2):
    """
    Adjust the brightness of the input image by adding a value to each pixel.

    Args:
        img (np.ndarray): Input image as a NumPy array.
        value (float): Value to be added to each pixel to adjust brightness. Default is -0.2.

    Returns:
        np.ndarray: Image with adjusted brightness as a NumPy array.

    Notes:
        - If the input image has a floating-point "dtype" (np.float, np.float32, np.float64),
            the pixel values are assumed to be in the range [0, 1].
        - If the input image has an integer "dtype", the pixel values are assumed to be in the
            range specified by the "dtype" (e.g., [0, 255] for np.uint8).
        - The adjusted pixel values are clipped to the minimum and maximum values allowed
            by the "dtype" of the input image.
    """
    if img.dtype in [np.float, np.float32, np.float64]:
        dtype_min, dtype_max = 0, 1
        dtype = np.float32
    else:
        dtype_min = np.iinfo(img.dtype).min
        dtype_max = np.iinfo(img.dtype).max
        dtype = np.iinfo(img.dtype)

    aug_img = np.clip(img.astype(np.float) + value, dtype_min, dtype_max).astype(dtype)

    return aug_img


def adjust_contrast(img, factor=1):
    """
    Adjust the contrast of the input image by multiplying it with a contrast factor.

    Args:
        img (np.ndarray): Input image as a NumPy array.
        factor (float): Contrast factor to adjust the contrast. Default is 1.0.

    Returns:
        np.ndarray: Image with adjusted contrast as a NumPy array.
    """
    if img.dtype in [np.float, np.float32, np.float64]:
        dtype_min, dtype_max = 0, 1
        dtype = np.float32
    else:
        dtype_min = np.iinfo(img.dtype).min
        dtype_max = np.iinfo(img.dtype).max
        dtype = np.iinfo(img.dtype)

    aug_img = np.clip(img.astype(np.float) * factor, dtype_min, dtype_max).astype(dtype)

    return aug_img