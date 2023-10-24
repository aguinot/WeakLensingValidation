# utility functions

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


def weighted_average(data, weights, seed, n_realization=100, verbose=True):
    """
    Compute weighted average and average standard deviation using bootstrap.

    Parameters
    ----------
    data: numpy.ndarray
        Data array.
    weights: numpy.ndarray
        Weight array.
    seed: int
        Seed to initialized randoms.
    n_realization: int
        Number of realization for the bootstrap. (Default: 100)
    verbose: bool
        Enable verbose. (Default: True)

    Returns
    -------
    average: float
        Weighted average.
    std: float
        Standard deviation of the average.
    """

    rng = np.random.RandomState(seed)

    samp_size = len(data)

    all_est = []
    for i in tqdm(
        range(n_realization),
        total=n_realization,
        disable=np.invert(verbose)
    ):
        sub_data_ind = rng.choice(samp_size, size=samp_size, replace=True)
        all_est.append(
            np.average(
                data[sub_data_ind],
                weights=weights[sub_data_ind],
            )
        )

    all_est = np.array(all_est)

    return np.mean(all_est), np.std(all_est)


def weighted_average_parallel(
    data,
    weights,
    seed,
    n_realization=100,
    n_jobs=-1,
    verbose=True
):
    """
    Compute weighted average and average standard deviation using bootstrap in
    parallel.

    Parameters
    ----------
    data: numpy.ndarray
        Data array.
    weights: numpy.ndarray
        Weight array.
    seed: int
        Seed to initialized randoms.
    n_realization: int
        Number of realization for the bootstrap. (Default: 100)
    n_jobs: int
        Number of processes to use for parallel computing. Use -1 to use all
        available ressources. (Default: -1)
    verbose: bool
        Enable verbose. (Default: True)

    Returns
    -------
    average: float
        Weighted average.
    std: float
        Standard deviation of the average.
    """

    def runner(data, weights, n_samp, seed_tmp):
        rng_tmp = np.random.RandomState(seed_tmp)
        sub_data_ind = rng_tmp.choice(n_samp, size=n_samp, replace=True)
        res = np.average(data[sub_data_ind], weights=weights[sub_data_ind])
        return res

    rng_master = np.random.RandomState(seed)
    seeds = rng_master.randint(low=0, high=2**30, size=n_realization)

    samp_size = len(data)
    all_est = (
        Parallel(n_jobs=n_jobs, backend='threading')
        (
            delayed(runner)
            (data, weights, samp_size, seeds[i]) for i in tqdm(
                range(n_realization),
                total=n_realization,
                disable=np.invert(verbose)
            )
        )
    )

    all_est = np.array(all_est)

    return np.mean(all_est), np.std(all_est)
