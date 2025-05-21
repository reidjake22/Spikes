from ..gabor_filters import GaborFilters
def generate_gabor_filters(
    save_path,
    lambdas: list,
    betas: list,
    thetas: list,
    psis: list,
    gammas: list,
    size: int,

):
    # for now we use old GaborFilter class cause we know it works
    gabor_filters = GaborFilters(size, lambdas, betas, thetas, psis, gammas)
    gabor_filters.store_all_as_images(save_path)
    gabor_filters.store_all_as_numpy(save_path)
    print(f"Saved all Gabor filters to {save_path}")
    gabor_filter_array = [ gabor_filter.filter_array for gabor_filter in gabor_filters.filters ]
    return gabor_filter_array
