import numpy as np
from typing import Tuple, Optional
import os
def create_receptive_field_mapping(
    num_filters: int,
    height: int,
    width: int,
    grid_shape: Tuple[int,int],   # (N_x, N_y) instead of explicit centers
    radius: float,
    avg_fanin: int,
    mapping_save_path: Optional[str] = None,
    rng: np.random.Generator = np.random.default_rng(),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build CSR mapping from (F,H,W) volume to N_x*N_y neurons
    laid out on an N_x × N_y grid over the image.
    """
    print(f" mapping_save_path: {mapping_save_path}")
    N_x, N_y = grid_shape
    # 0) Generate neuron centers on a regular grid
    rows = np.linspace(0, height-1, N_x)
    cols = np.linspace(0, width-1,  N_y)
    R, C = np.meshgrid(rows, cols, indexing='ij')
    neuron_centers = np.stack([R, C], axis=-1).reshape(-1, 2)
    N = neuron_centers.shape[0]


    raw_lists = []
    # 1) collect raw receptive-field coords (f,i,j) per neuron
    for (i0, j0) in neuron_centers:
        coords = [
            (f, i, j)
            for i in range(height) for j in range(width)
            if (i - i0)**2 + (j - j0)**2 <= radius**2
            for f in range(num_filters)
        ]
        raw_lists.append(coords)

    # 2) compute global subsampling probability
    sizes = [len(lst) for lst in raw_lists]
    avg_size = np.mean(sizes) if sizes else 0
    p = avg_fanin / avg_size if avg_size > 0 else 0.0

    # 3) subsample each list & convert to flat indices
    dims = (num_filters, height, width)
    flat_lists = []
    for lst in raw_lists:
        k = len(lst)
        if k > 0:
            mask = rng.random(k) < p
            subsampled = [lst[i] for i, m in enumerate(mask) if m]
            # unpack tuples and ravel into 1D indices
            f_arr, i_arr, j_arr = zip(*subsampled) if subsampled else ([], [], [])
            flat = np.ravel_multi_index((f_arr, i_arr, j_arr), dims=dims)
            flat_lists.append(np.array(flat, dtype=int))
        else:
            flat_lists.append(np.zeros(0, dtype=int))

    # 4) build CSR arrays
    lengths = [len(fl) for fl in flat_lists]
    indptr = np.zeros(N+1, dtype=int)
    indptr[1:] = np.cumsum(lengths)
    indices = np.empty(indptr[-1], dtype=int)
    pos = 0
    for fl in flat_lists:
        l = fl.size
        indices[pos:pos+l] = fl
        pos += l

    # 5) optionally save mapping
    if mapping_save_path:
        print(f"Saving mapping to {mapping_save_path} as .npz")
        np.savez(os.path.join(mapping_save_path,"mapping"), indices=indices, indptr=indptr)

    return indices, indptr
