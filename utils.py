import numpy as np

def supersample(array, factor):
    """
    Super sample an index array by adding values uniformly between points.

    Parameters
    ----------
    array: ndarray
        Index array (such as would be used on an axis.)
        Should be sorted and uniformly distributed.
    """
    Δt = abs(array[1] - array[0])
    np.sort(np.concatenate( [tarr[:-1] + i/factor * Δt for i in range(factor)]
                            + [tarr[-1]] ))
        # We separate out tarr[-1] so that supersampling remains between
        # existing points.
