import numpy as np
from scipy.signal import fftconvolve

def convolve_images(
        images: np.ndarray,
        gabor_filters: list[np.ndarray],
        save_path: str,
        normalise: bool = True,
) -> np.ndarray:
    """
    Convolve a batch of 2D images with a set of Gabor filters.

    Args:
        images (np.ndarray): shape (num_images, H, W).
        gabor_filters (Sequence[np.ndarray]): F filters, each shape (kH, kW).
        save_path (str): where to np.save the result.

    Returns:
        np.ndarray: shape (num_images, F, H, W) of filtered images.
    """
    num_images, H, W = images.shape

    # stack filters into a single (F, kH, kW) array
    filters = np.stack(gabor_filters, axis=0)
    F, kH, kW = filters.shape

    # prepare output
    convolved = np.zeros((num_images, F, H, W), dtype=np.result_type(images, filters))

    for i in range(num_images):
        # broadcast_to wants a shape tuple
        img_stack = np.broadcast_to(images[i], (F, H, W))
        # FFT‐based convolution over axes 1 & 2 (the spatial dims)
        convolved[i] = fftconvolve(img_stack, filters, mode='same', axes=(1, 2))
        # Step 2: Normalise each convolved image using the euclidian norm
        if normalise:
            norms = np.linalg.norm(convolved[i], axis=(1, 2), keepdims=True)  # shape (F,1,1)
            convolved[i] /= norms


    # save and return
    np.save(save_path, convolved)
    return convolved

