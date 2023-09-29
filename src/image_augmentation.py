#@title image_augmentation.py

from skimage import transform as trans
import numpy as np
import numpy.ma as ma
import cv2
import random
from collections.abc import Sequence
from scipy.ndimage import rotate



def center_rotate(image, label, degree):
    """
    Applies rotation augmentation to an image patch and label.

    Args:
        image (numpy array) : The input image patch as a numpy array.
        label (numpy array) : The corresponding label as a numpy array.
        degree (list of floats) : If the list has exactly two elements they will
            be considered the lower and upper bounds for the rotation angle 
            (in degrees) respectively. If number of elements are bigger than 2, 
            then one value is chosen randomly as the rotation angle.

    Returns:
        A tuple containing the rotated image patch and label as numpy arrays.
    """
    if isinstance(degree, (tuple, list)):
        if len(degree) == 2:
            rotation_degree = random.uniform(*degree)
        elif len(degree) > 2:
            rotation_degree = random.choice(degree)
        else:
            raise ValueError("Parameter angle needs at least two elements.")
    else:
        raise ValueError(
            "Rotation bound param for augmentation must be a tuple or list."
        )

    # Apply rotation augmentation to the image patch
    rotated_image = rotate(image, rotation_degree, axes=(1, 0), 
                           reshape=False, mode='reflect')

    # Apply rotation augmentation to the label
    rotated_label = rotate(label, rotation_degree, axes=(1, 0), 
                           reshape=False, mode='nearest')

    # Return the rotated image patch and label as a tuple
    return rotated_image, rotated_label


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

    def diagonal_flip(image):
        flipped = np.flip(image, 1)
        flipped = np.flip(flipped, 0)
        return flipped

    if isinstance(flip_type, str):
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
            img = diagonal_flip(img)
            label = diagonal_flip(label)

        else:
            raise ValueError("Flip type must be one of: 'h_flip', 'v_flip' or 'd_flip'.")
    else:
        raise ValueError("Flip type param must be a tuple or list.")

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
    resampled_label = trans.resize(label, (resize_h, resize_w), order=0, preserve_range=True)

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
