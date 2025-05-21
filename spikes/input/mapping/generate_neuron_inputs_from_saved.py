
import numpy as np

def generate_neuron_inputs_from_saved(
    convolved_path: str,
    mapping_path: str
) -> np.ndarray:
    """
    Load convolved images and CSR mapping, then for each neuron,
    gather its inputs (only positive values) and sum them.

    Returns:
        inputs: np.ndarray of shape (num_images, N_neurons)
    """
    # load data
    print("convolved_path", convolved_path)
    convolved = np.load(convolved_path)  # shape (num_images, F, H, W)
    print(convolved.shape)
    m = np.load(mapping_path)
    indices = m['indices']
    indptr = m['indptr']

    num_images = convolved.shape[0]
    flat = convolved.reshape(num_images, -1)  # (num_images, F*H*W)
    N = indptr.size - 1

    # allocate output
    inputs = np.zeros((num_images, N), dtype=flat.dtype)

    # vectorized per-neuron gather & positive-sum
    for j in range(N):
        start, end = indptr[j], indptr[j+1]
        vals = flat[:, indices[start:end]]  # (num_images, fanin_j)
        # zero out negatives then sum
        inputs[:, j] = np.clip(vals, a_min=0, a_max=None).sum(axis=1)

    return inputs

