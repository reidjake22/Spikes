import numpy as np
from typing import List

def sample_receptive_inputs(
    volume: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray
) -> List[np.ndarray]:
    """
    Given a 3D volume (F, H, W) and CSR mapping (indices, indptr),
    return a list of 1D arrays, where each entry is the gathered inputs
    to the corresponding neuron.
    """
    flat_vol = volume.ravel()
    N = len(indptr) - 1
    outputs: List[np.ndarray] = []
    for n in range(N):
        start, end = indptr[n], indptr[n + 1]
        outputs.append(flat_vol[indices[start:end]])
    return outputs