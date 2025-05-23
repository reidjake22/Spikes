
def dot_product(u: list[int], v: list[int]) -> int:
    """
    Computes the dot product of two vectors u and v.
    """
    # First check whether vectors are of same length and not empty
    if len(u) != len(v):
        raise ValueError("Vectors are of different dimensions")
    if len(u) == 0:
        raise ValueError("Your vectors are both empty")
    # Compute the dot product
    doc_product = sum(x * y for x, y in zip(u,v))
    return doc_product


