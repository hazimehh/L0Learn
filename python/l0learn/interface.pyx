cimport numpy
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool as cppbool
import numpy as np
from scipy.sparse import csc_matrix

from typing import Union, Optional, List

from l0learn.cyarma cimport dmat, sp_dmat, numpy_to_sp_dmat_d, numpy_to_dmat_d, dvec, numpy_to_dvec_d, \
    sp_dmat_field_to_list, dvec_field_to_list

# def gen_synthetic():
#     raise NotImplementedError
#
#
# def gen_synthetic_high_corr():
#     raise NotImplementedError
#
#
# def gen_synthetic_logistic():
#     raise NotImplementedError


def np_to_arma_check(arr):
    # TODO: Add checks for Behaved and OwnsData
    if isinstance(arr, np.ndarray):
        if not arr.flags['F_CONTIGUOUS']:
            raise ValueError("expected arr to be F_CONTIGUOUS.")
    elif isinstance(arr, csc_matrix):
        if not arr.data.flags['F_CONTIGUOUS']:
            raise ValueError("expected arr.data to be F_CONTIGUOUS.")
        if not arr.indices.flags['F_CONTIGUOUS']:
            raise ValueError("expected arr.indices to be F_CONTIGUOUS.")
        if not arr.indptr.flags['F_CONTIGUOUS']:
            raise ValueError("expected arr.indptr to be F_CONTIGUOUS.")
    else:
        raise NotImplementedError(f"expected arr to be of type {np.ndarray} or {csc_matrix}, but got {type(arr)}.")

    if arr.ndim == 0:
            raise ValueError("expected 'arr.ndim' to be 1 or 2, but got 0. Should be passed as scalar")
    elif arr.ndim > 2:
        raise NotImplementedError(f"expected 'arr.ndim' to be 1 or 2, but got {arr.ndim}. Not supported")

    if np.product(arr.shape) == 0:
        raise ValueError(f"expected non-degenerate dimensions of arr, but got {arr.ndim}")

    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"expected numerical dtype, but got {arr.dtype}")

    if not np.isrealobj(arr):
        raise ValueError(f"expected non-complex dtype, but got {arr.dtype}")


SUPPORTED_LOSS = ("SquaredError", "Logistic", "SquaredHinge")
CLASSIFICATION_LOSS = SUPPORTED_LOSS[1], SUPPORTED_LOSS[2]
SUPPORTED_PENALTY = ("L0", "L0L1", "L0L2")
SUPPORTED_ALGORITHM = ("CD", "CDPSI")


def fit(X: Union[np.ndarray, csc_matrix],
        y: np.ndarray,
        loss: str = "SquaredHinge",
        penalty: str = "L0",
        algorithm: str = "CD",
        max_support_size: int = 100,
        num_lambda: Optional[int] = 100,
        num_gamma: Optional[int] = 1,
        gamma_max: float = 10.,
        gamma_min: float = .0001,
        partial_sort: bool = True,
        max_iter: int = 200,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        active_set: bool = True,
        active_set_num: int = 3,
        max_swaps: int = 100,
        scale_down_factor: float = 0.8,
        screen_size: int = 1000,
        lambda_grid: Optional[List[List[float]]] = None,
        exclude_first_k: int = 0,
        intercept: bool = True,
        lows: Union[np.ndarray, float] = -float('inf'),
        highs: Union[np.ndarray, float] = +float('inf'),):
    """
    Computes the regularization path for the specified loss function and penalty function.

    Parameters
    ----------
    X : np.ndarray or csc_matrix of shape (N, P)
        Data Matrix where rows of X are observations and columns of X are features
    y : np.ndarray of shape (P)
        The response vector where y[i] corresponds to X[i, :]
        For classification, a binary vector (-1, 1) is requried .
    loss : str
        The loss function. Currently supports the choices:
            "SquaredError" (for regression),
            "Logistic" (for logistic regression), and
            "SquaredHinge" (for smooth SVM).
    penalty : str
        The type of regularization.
        This can take either one of the following choices:
            "L0",
            "L0L2", and
            "L0L1"
    algorithm : str
        The type of algorithm used to minimize the objective function. Currently "CD" and "CDPSI" are are supported.
        "CD" is a variant of cyclic coordinate descent and runs very fast. "CDPSI" performs local combinatorial search
        on top of CD and typically achieves higher quality solutions (at the expense of increased running time).
    max_support_size
    num_lambda
    num_gamma
    gamma_max
    gamma_min
    partial_sort
    max_iter
    rtol
    atol
    active_set
    active_set_num
    max_swaps
    scale_down_factor
    screen_size
    lambda_grid
    exclude_first_k
    intercept
    lows
    highs

    Returns
    -------

    """

    if not isinstance(X, (np.ndarray, csc_matrix)) or not np.isrealobj(X) or X.ndim != 2 or not np.product(X.shape):
        raise ValueError(f"expected X to be a 2D non-degenerate real numpy or csc_matrix, but got {X}.")

    n, p = X.shape
    if not isinstance(y, np.ndarray) or not np.isrealobj(y) or y.ndim != 1 or len(y) != n:
        raise ValueError(f"expected y to be a 1D real numpy, but got {y}.")

    if loss not in SUPPORTED_LOSS:
        raise ValueError(f"expected loss parameter to be on of {SUPPORTED_LOSS}, but got {loss}")

    if penalty not in SUPPORTED_PENALTY:
        raise ValueError(f"expected penalty parameter to be on of {SUPPORTED_PENALTY}, but got {penalty}")
    if algorithm not in SUPPORTED_ALGORITHM:
        raise ValueError(f"expected algorithm parameter to be on of {SUPPORTED_ALGORITHM}, but got {algorithm}")
    if not isinstance(max_support_size, int) or not (0 < max_support_size <= p):
        raise ValueError(f"expected max_support_size parameter to be a positive integer less than {p},"
                         f" but got {max_support_size}")
    if gamma_max < 0:
        raise ValueError(f"expected gamma_max parameter to be a positive float, but got {gamma_max}")
    if gamma_min < 0 or gamma_min > gamma_max:
        raise ValueError(f"expected gamma_max parameter to be a positive float less than gamma_max,"
                         f" but got {gamma_min}")
    if not isinstance(partial_sort, bool):
        raise ValueError(f"expected partial_sort parameter to be a bool, but got {partial_sort}")
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError(f"expected max_iter parameter to be a positive integer, but got {max_iter}")
    if rtol < 0 or rtol >= 1:
        raise ValueError(f"expected rtol parameter to exist in [0, 1), but got {rtol}")
    if atol < 0:
        raise ValueError(f"expected atol parameter to exist in [0, INF), but got {atol}")
    if not isinstance(active_set, bool):
        raise ValueError(f"expected active_set parameter to be a bool, but got {active_set}")
    if not isinstance(active_set_num, int) or active_set_num < 1:
        raise ValueError(f"expected active_set_num parameter to be a positive integer, but got {active_set_num}")
    if not isinstance(max_swaps, int) or max_swaps < 1:
        raise ValueError(f"expected max_swaps parameter to be a positive integer, but got {max_swaps}")
    if not (0 < scale_down_factor < 1):
        raise ValueError(f"expected scale_down_factor parameter to exist in (0, 1), but got {scale_down_factor}")
    if not isinstance(screen_size, int) or screen_size < 1:
        raise ValueError(f"expected screen_size parameter to be a positive integer, but got {screen_size}")
    if not isinstance(exclude_first_k, int) or not (0 <= exclude_first_k <= p):
        raise ValueError(f"expected exclude_first_k parameter to be a positive integer less than {p}, "
                         f"but got {exclude_first_k}")
    if not isinstance(intercept, bool):
        raise ValueError(f"expected intercept parameter to be a bool, "
                 f"but got {intercept}")

    if loss in CLASSIFICATION_LOSS:
        if sorted(np.unique(y)) != [-1, 1]:
            raise ValueError(f"expected y vector to only have two unique values (-1 and 1) (Binary Classification), "
                             f"but got {np.unique(y)}")

        if penalty == "L0":
            # TODO: Must be corrected in R code:  https://github.com/hazimehh/L0Learn/blob/7a65474dfdb01489a0c263d7b24fbafad56fba61/R/fit.R#L136
            # Pure L0 is not supported for classification
            # Below we add a small L2 component.

            if len(lambda_grid) != 1:
                # If this error checking was left to the lower section, it would confuse users as
                # we are converting L0 to L0L2 with small L2 penalty.
                # Here we must check if lambdaGrid is supplied (And thus use 'autolambda')
                # If 'lambdaGrid' is supplied, we must only supply 1 list of lambda values
                raise ValueError(f"L0 Penalty requires 'lambda_grid' to be a list of length 1, but got {lambda_grid}.")

        penalty = "L0L2"
        gamma_max = 1e-7
        gamma_min = 1e-7

    if lambda_grid is None:
        lambda_grid = [[0.]]
        auto_lambda = True
        if not isinstance(num_lambda, int) or num_lambda < 1:
            raise ValueError(f"expected num_lambda to a positive integer when lambda_grid is None, but got {num_lambda}.")
        if not isinstance(num_gamma, int) or num_gamma < 1:
            raise ValueError(f"expected num_gamma to a positive integer when lambda_grid is None, but got {num_gamma}.")
        if penalty == "L0" and num_gamma != 1:
            raise ValueError(f"expected num_gamma to 1 when penalty = 'L0', but got {num_gamma}.")
    else: # lambda_grid should be a List[List[float]]
        if num_gamma is not None:
            raise ValueError(f"expected num_gamma to be None if lambda_grid is specified by the user, "
                             f"but got {num_gamma}")
        num_gamma = len(lambda_grid)

        if num_lambda is not None:
            raise ValueError(f"expected num_lambda to be None if lambda_grid is specified by the user, "
                             f"but got {num_lambda}")
        num_lambda = 0 #  This value is ignored.
        auto_lambda = False
        bad_lambda_grid = False

        if penalty == "L0" and num_gamma != 1:
           raise ValueError(f"expected lambda_grid to of length 1 when penalty = 'L0', but got {len(lambda_grid)}")

        for i, sub_lambda_grid in enumerate(lambda_grid):
            current = float("inf")
            if sub_lambda_grid[0] <= 0:
                raise ValueError(f"Expected all values of lambda_grid to be positive, "
                                 f"but got lambda_grid[{i}] containing a negative value")
            if any(np.diff(sub_lambda_grid) >= 0):
                raise ValueError(f"Expected each element of lambda_grid to be a list of decreasing value, "
                                 f"but got lambda_grid[{i}] containing an increasing value.")

    n, p = X.shape
    with_bounds = False

    if isinstance(lows, float):
        if lows > 0:
            raise ValueError(f"expected lows to be a non-positive float, but got {lows}")
        elif lows > -float('inf'):
            with_bounds = True
    elif isinstance(lows, np.ndarray) and lows.ndim == 1 and len(lows) == p and all(lows <= 0):
        with_bounds = True
    else:
        raise ValueError(f"expected lows to be a non-positive float, or a 1D numpy array of length {p} of non-positives "
                         f"floats, but got {lows}")

    if isinstance(highs, float):
        if highs < 0:
            raise ValueError(f"expected highs to be a non-negative float, but got {highs}")
        if highs < float('inf'):
            with_bounds = True
    elif isinstance(highs, np.ndarray) and highs.ndim == 1 and len(highs) == p and (highs >= 0):
        with_bounds = True
    else:
        raise ValueError(f"expected highs to be a non-negative float, or a 1D numpy array of length {p} of "
                         f"non-negative floats, but got {highs}")

    if with_bounds:
        if isinstance(lows, float):
            lows = np.ones(p) * lows
        if isinstance(highs, float):
            highs = np.ones(p) * highs

        if any(lows >= highs):
            bad_bounds = np.argwhere(lows >= highs)
            raise ValueError(f"expected to be high to be elementwise greater than lows, "
                             f"but got indices {bad_bounds[0]} where that is not the case ")
    else:
        lows = np.array([0.])
        highs = np.array(([0.]))

    cdef vector[vector[double]] c_lambda_grid
    try:
        c_lambda_grid = lambda_grid
    except TypeError:
        raise ValueError(f"expected lambda_grid to be a list of list of doubles, but got {lambda_grid}")

    cdef string c_loss = loss.encode('UTF-8')
    cdef string c_penalty = penalty.encode('UTF-8')
    cdef string c_algorithim = algorithm.encode('UTF-8')

    cdef fitmodel results
    if isinstance(X, np.ndarray):
        results = _L0LearnFit_dense(X=numpy_to_dmat_d(X),
                                    y=numpy_to_dvec_d(y),
                                    Loss=c_loss,
                                    Penalty=c_penalty,
                                    Algorithm=c_algorithim,
                                    NnzStopNum=max_support_size,
                                    G_ncols=num_lambda,
                                    G_nrows=num_gamma,
                                    Lambda2Max=gamma_max,
                                    Lambda2Min=gamma_min,
                                    PartialSort=partial_sort,
                                    MaxIters=max_iter,
                                    rtol=rtol,
                                    atol=atol,
                                    ActiveSet=active_set,
                                    ActiveSetNum=active_set_num,
                                    MaxNumSwaps=max_swaps,
                                    ScaleDownFactor=scale_down_factor,
                                    ScreenSize=screen_size,
                                    LambdaU=not auto_lambda,
                                    Lambdas=c_lambda_grid,
                                    ExcludeFirstK=exclude_first_k,
                                    Intercept=intercept,
                                    withBounds=with_bounds,
                                    Lows=numpy_to_dvec_d(lows),
                                    Highs=numpy_to_dvec_d(highs))
    else: # isinstance(X, csc_matrix)
        results = _L0LearnFit_sparse(X=numpy_to_sp_dmat_d(X),
                                     y=numpy_to_dvec_d(y),
                                     Loss=c_loss,
                                     Penalty=c_penalty,
                                     Algorithm=c_algorithim,
                                     NnzStopNum=max_support_size,
                                     G_ncols=num_lambda,
                                     G_nrows=num_gamma,
                                     Lambda2Max=gamma_max,
                                     Lambda2Min=gamma_min,
                                     PartialSort=partial_sort,
                                     MaxIters=max_iter,
                                     rtol=rtol,
                                     atol=atol,
                                     ActiveSet=active_set,
                                     ActiveSetNum=active_set_num,
                                     MaxNumSwaps=max_swaps,
                                     ScaleDownFactor=scale_down_factor,
                                     ScreenSize=screen_size,
                                     LambdaU=not auto_lambda,
                                     Lambdas=c_lambda_grid,
                                     ExcludeFirstK=exclude_first_k,
                                     Intercept=intercept,
                                     withBounds=with_bounds,
                                     Lows=numpy_to_dvec_d(lows),
                                     Highs=numpy_to_dvec_d(highs))

    return {"NnzCount": results.NnzCount,
            "Lambda0": results.Lambda0,
            "Lambda12": results.Lambda12,
            "Beta": sp_dmat_field_to_list(results.Beta),
            "Intercept": results.Intercept,
            "Converged": results.Converged}

def cvfit(X: Union[np.ndarray, csc_matrix],
          y: np.ndarray,
          loss: str = "SquaredHinge",
          penalty: str = "L0",
          algorithm: str = "CD",
          num_folds: int = 10,
          seed: int = 1,
          max_support_size: int = 100,
          num_lambda: Optional[int] = 100,
          num_gamma: Optional[int] = 1,
          gamma_max: float = 10.,
          gamma_min: float = .0001,
          partial_sort: bool = True,
          max_iter: int = 200,
          rtol: float = 1e-6,
          atol: float = 1e-9,
          active_set: bool = True,
          active_set_num: int = 3,
          max_swaps: int = 100,
          scale_down_factor: float = 0.8,
          screen_size: int = 1000,
          lambda_grid: Optional[List[List[float]]] = None,
          exclude_first_k: int = 0,
          intercept: bool = True,
          lows: Union[np.ndarray, float] = -float('inf'),
          highs: Union[np.ndarray, float] = +float('inf'),):
    if not isinstance(X, (np.ndarray, csc_matrix)) or not np.isrealobj(X) or X.ndim != 2 or not np.product(X.shape):
        raise ValueError(f"expected X to be a 2D non-degenerate real numpy or csc_matrix, but got {X}.")

    n, p = X.shape
    if not isinstance(y, np.ndarray) or not np.isrealobj(y) or y.ndim != 1 or len(y) != n:
        raise ValueError(f"expected y to be a 1D real numpy, but got {y}.")

    if loss not in SUPPORTED_LOSS:
        raise ValueError(f"expected loss parameter to be on of {SUPPORTED_LOSS}, but got {loss}")

    if penalty not in SUPPORTED_PENALTY:
        raise ValueError(f"expected penalty parameter to be on of {SUPPORTED_PENALTY}, but got {penalty}")
    if algorithm not in SUPPORTED_ALGORITHM:
        raise ValueError(f"expected algorithm parameter to be on of {SUPPORTED_ALGORITHM}, but got {algorithm}")
    if not isinstance(num_folds, int) or num_folds < 2:
        raise ValueError(f"expected num_folds parameter to be a integer greater than 2, but got {num_folds}")
    if not isinstance(seed, int):
        raise ValueError(f"expected seed parameter to be an integer, but got {seed}")
    if not isinstance(max_support_size, int) or not (0 < max_support_size <= p):
        raise ValueError(f"expected max_support_size parameter to be a positive integer less than {p},"
                         f" but got {max_support_size}")
    if gamma_max < 0:
        raise ValueError(f"expected gamma_max parameter to be a positive float, but got {gamma_max}")
    if gamma_min < 0 or gamma_min > gamma_max:
        raise ValueError(f"expected gamma_max parameter to be a positive float less than gamma_max,"
                         f" but got {gamma_min}")
    if not isinstance(partial_sort, bool):
        raise ValueError(f"expected partial_sort parameter to be a bool, but got {partial_sort}")
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError(f"expected max_iter parameter to be a positive integer, but got {max_iter}")
    if rtol < 0 or rtol >= 1:
        raise ValueError(f"expected rtol parameter to exist in [0, 1), but got {rtol}")
    if atol < 0:
        raise ValueError(f"expected atol parameter to exist in [0, INF), but got {atol}")
    if not isinstance(active_set, bool):
        raise ValueError(f"expected active_set parameter to be a bool, but got {active_set}")
    if not isinstance(active_set_num, int) or active_set_num < 1:
        raise ValueError(f"expected active_set_num parameter to be a positive integer, but got {active_set_num}")
    if not isinstance(max_swaps, int) or max_swaps < 1:
        raise ValueError(f"expected max_swaps parameter to be a positive integer, but got {max_swaps}")
    if not (0 < scale_down_factor < 1):
        raise ValueError(f"expected scale_down_factor parameter to exist in (0, 1), but got {scale_down_factor}")
    if not isinstance(screen_size, int) or screen_size < 1:
        raise ValueError(f"expected screen_size parameter to be a positive integer, but got {screen_size}")
    if not isinstance(exclude_first_k, int) or not (0 <= exclude_first_k <= p):
        raise ValueError(f"expected exclude_first_k parameter to be a positive integer less than {p}, "
                         f"but got {exclude_first_k}")
    if not isinstance(intercept, bool):
        raise ValueError(f"expected intercept parameter to be a bool, "
                 f"but got {intercept}")

    if loss in CLASSIFICATION_LOSS:
        if sorted(np.unique(y)) != [-1, 1]:
            raise ValueError(f"expected y vector to only have two unique values (-1 and 1) (Binary Classification), "
                             f"but got {np.unique(y)}")

        if penalty == "L0":
            # TODO: Must be corrected in R code:  https://github.com/hazimehh/L0Learn/blob/7a65474dfdb01489a0c263d7b24fbafad56fba61/R/fit.R#L136
            # Pure L0 is not supported for classification
            # Below we add a small L2 component.

            if len(lambda_grid) != 1:
                # If this error checking was left to the lower section, it would confuse users as
                # we are converting L0 to L0L2 with small L2 penalty.
                # Here we must check if lambdaGrid is supplied (And thus use 'autolambda')
                # If 'lambdaGrid' is supplied, we must only supply 1 list of lambda values
                raise ValueError(f"L0 Penalty requires 'lambda_grid' to be a list of length 1, but got {lambda_grid}.")

        penalty = "L0L2"
        gamma_max = 1e-7
        gamma_min = 1e-7

    if lambda_grid is None:
        lambda_grid = [[0.]]
        auto_lambda = True
        if not isinstance(num_lambda, int) or num_lambda < 1:
            raise ValueError(f"expected num_lambda to a positive integer when lambda_grid is None, but got {num_lambda}.")
        if not isinstance(num_gamma, int) or num_gamma < 1:
            raise ValueError(f"expected num_gamma to a positive integer when lambda_grid is None, but got {num_gamma}.")
        if penalty == "L0" and num_gamma != 1:
            raise ValueError(f"expected num_gamma to 1 when penalty = 'L0', but got {num_gamma}.")
    else: # lambda_grid should be a List[List[float]]
        if num_gamma is not None:
            raise ValueError(f"expected num_gamma to be None if lambda_grid is specified by the user, "
                             f"but got {num_gamma}")
        num_gamma = len(lambda_grid)

        if num_lambda is not None:
            raise ValueError(f"expected num_lambda to be None if lambda_grid is specified by the user, "
                             f"but got {num_lambda}")
        num_lambda = 0 #  This value is ignored.
        auto_lambda = False
        bad_lambda_grid = False

        if penalty == "L0" and num_gamma != 1:
           raise ValueError(f"expected lambda_grid to of length 1 when penalty = 'L0', but got {len(lambda_grid)}")

        for i, sub_lambda_grid in enumerate(lambda_grid):
            current = float("inf")
            if sub_lambda_grid[0] <= 0:
                raise ValueError(f"Expected all values of lambda_grid to be positive, "
                                 f"but got lambda_grid[{i}] containing a negative value")
            if any(np.diff(sub_lambda_grid) >= 0):
                raise ValueError(f"Expected each element of lambda_grid to be a list of decreasing value, "
                                 f"but got lambda_grid[{i}] containing an increasing value.")

    n, p = X.shape
    with_bounds = False

    if isinstance(lows, float):
        if lows > 0:
            raise ValueError(f"expected lows to be a non-positive float, but got {lows}")
        elif lows > -float('inf'):
            with_bounds = True
    elif isinstance(lows, np.ndarray) and lows.ndim == 1 and len(lows) == p and all(lows <= 0):
        with_bounds = True
    else:
        raise ValueError(f"expected lows to be a non-positive float, or a 1D numpy array of length {p} of non-positives "
                         f"floats, but got {lows}")

    if isinstance(highs, float):
        if highs < 0:
            raise ValueError(f"expected highs to be a non-negative float, but got {highs}")
        if highs < float('inf'):
            with_bounds = True
    elif isinstance(highs, np.ndarray) and highs.ndim == 1 and len(highs) == p and (highs >= 0):
        with_bounds = True
    else:
        raise ValueError(f"expected highs to be a non-negative float, or a 1D numpy array of length {p} of "
                         f"non-negative floats, but got {highs}")

    if with_bounds:
        if isinstance(lows, float):
            lows = np.ones(p) * lows
        if isinstance(highs, float):
            highs = np.ones(p) * highs

        if any(lows >= highs):
            bad_bounds = np.argwhere(lows >= highs)
            raise ValueError(f"expected to be high to be elementwise greater than lows, "
                             f"but got indices {bad_bounds[0]} where that is not the case ")
    else:
        lows = np.array([0.])
        highs = np.array(([0.]))

    cdef vector[vector[double]] c_lambda_grid
    try:
        c_lambda_grid = lambda_grid
    except TypeError:
        raise ValueError(f"expected lambda_grid to be a list of list of doubles, but got {lambda_grid}")

    cdef string c_loss = loss.encode('UTF-8')
    cdef string c_penalty = penalty.encode('UTF-8')
    cdef string c_algorithim = algorithm.encode('UTF-8')

    cdef cvfitmodel results
    if isinstance(X, np.ndarray):
        results = _L0LearnCV_dense(X=numpy_to_dmat_d(X),
                                    y=numpy_to_dvec_d(y),
                                    Loss=c_loss,
                                    Penalty=c_penalty,
                                    Algorithm=c_algorithim,
                                    NnzStopNum=max_support_size,
                                    G_ncols=num_lambda,
                                    G_nrows=num_gamma,
                                    Lambda2Max=gamma_max,
                                    Lambda2Min=gamma_min,
                                    PartialSort=partial_sort,
                                    MaxIters=max_iter,
                                    rtol=rtol,
                                    atol=atol,
                                    ActiveSet=active_set,
                                    ActiveSetNum=active_set_num,
                                    MaxNumSwaps=max_swaps,
                                    ScaleDownFactor=scale_down_factor,
                                    ScreenSize=screen_size,
                                    LambdaU=not auto_lambda,
                                    Lambdas=c_lambda_grid,
                                    nfolds=num_folds,
                                    seed=seed,
                                    ExcludeFirstK=exclude_first_k,
                                    Intercept=intercept,
                                    withBounds=with_bounds,
                                    Lows=numpy_to_dvec_d(lows),
                                    Highs=numpy_to_dvec_d(highs))
    else: # isinstance(X, csc_matrix)
        results = _L0LearnCV_sparse(X=numpy_to_sp_dmat_d(X),
                                     y=numpy_to_dvec_d(y),
                                     Loss=c_loss,
                                     Penalty=c_penalty,
                                     Algorithm=c_algorithim,
                                     NnzStopNum=max_support_size,
                                     G_ncols=num_lambda,
                                     G_nrows=num_gamma,
                                     Lambda2Max=gamma_max,
                                     Lambda2Min=gamma_min,
                                     PartialSort=partial_sort,
                                     MaxIters=max_iter,
                                     rtol=rtol,
                                     atol=atol,
                                     ActiveSet=active_set,
                                     ActiveSetNum=active_set_num,
                                     MaxNumSwaps=max_swaps,
                                     ScaleDownFactor=scale_down_factor,
                                     ScreenSize=screen_size,
                                     LambdaU=not auto_lambda,
                                     Lambdas=c_lambda_grid,
                                     nfolds=num_folds,
                                     seed=seed,
                                     ExcludeFirstK=exclude_first_k,
                                     Intercept=intercept,
                                     withBounds=with_bounds,
                                     Lows=numpy_to_dvec_d(lows),
                                     Highs=numpy_to_dvec_d(highs))

    return {"NnzCount": results.NnzCount,
            "Lambda0": results.Lambda0,
            "Lambda12": results.Lambda12,
            "Beta": sp_dmat_field_to_list(results.Beta),
            "Intercept": results.Intercept,
            "Converged": results.Converged,
            "CVMeans": dvec_field_to_list(results.CVMeans)[0],
            "CVSDs": dvec_field_to_list(results.CVSDs)[0]}


cdef fitmodel _L0LearnFit_dense(const dmat& X,
                                const dvec& y,
                                const string Loss,
                                const string Penalty,
                                const string Algorithm,
                                const size_t NnzStopNum,
                                const size_t G_ncols,
                                const size_t G_nrows,
                                const double Lambda2Max,
                                const double Lambda2Min,
                                const cppbool PartialSort,
                                const size_t MaxIters,
                                const double rtol,
                                const double atol,
                                const cppbool ActiveSet,
                                const size_t ActiveSetNum,
                                const size_t MaxNumSwaps,
                                const double ScaleDownFactor,
                                const size_t ScreenSize,
                                const cppbool LambdaU,
                                const vector[vector[double]]& Lambdas,
                                const size_t ExcludeFirstK,
                                const cppbool Intercept,
                                const cppbool withBounds,
                                const dvec &Lows,
                                const dvec &Highs):
    return L0LearnFit[dmat](X, y, Loss, Penalty, Algorithm, NnzStopNum, G_ncols, G_nrows, Lambda2Max, Lambda2Min, PartialSort,
                      MaxIters, rtol, atol, ActiveSet, ActiveSetNum, MaxNumSwaps, ScaleDownFactor, ScreenSize, LambdaU,
                      Lambdas, ExcludeFirstK, Intercept, withBounds, Lows, Highs)


cdef fitmodel _L0LearnFit_sparse(const sp_dmat& X,
                                 const dvec& y,
                                 const string Loss,
                                 const string Penalty,
                                 const string Algorithm,
                                 const size_t NnzStopNum,
                                 const size_t G_ncols,
                                 const size_t G_nrows,
                                 const double Lambda2Max,
                                 const double Lambda2Min,
                                 const cppbool PartialSort,
                                 const size_t MaxIters,
                                 const double rtol,
                                 const double atol,
                                 const cppbool ActiveSet,
                                 const size_t ActiveSetNum,
                                 const size_t MaxNumSwaps,
                                 const double ScaleDownFactor,
                                 const size_t ScreenSize,
                                 const cppbool LambdaU,
                                 const vector[vector[double]]& Lambdas,
                                 const size_t ExcludeFirstK,
                                 const cppbool Intercept,
                                 const cppbool withBounds,
                                 const dvec &Lows,
                                 const dvec &Highs):
    return L0LearnFit[sp_dmat](X, y, Loss, Penalty, Algorithm, NnzStopNum, G_ncols, G_nrows, Lambda2Max, Lambda2Min, PartialSort,
                      MaxIters, rtol, atol, ActiveSet, ActiveSetNum, MaxNumSwaps, ScaleDownFactor, ScreenSize, LambdaU,
                      Lambdas, ExcludeFirstK, Intercept, withBounds, Lows, Highs)


cdef cvfitmodel _L0LearnCV_dense(const dmat& X,
                                 const dvec& y,
                                 const string Loss,
                                 const string Penalty,
                                 const string Algorithm,
                                 const size_t NnzStopNum,
                                 const size_t G_ncols,
                                 const size_t G_nrows,
                                 const double Lambda2Max,
                                 const double Lambda2Min,
                                 const cppbool PartialSort,
                                 const size_t MaxIters,
                                 const double rtol,
                                 const double atol,
                                 const cppbool ActiveSet,
                                 const size_t ActiveSetNum,
                                 const size_t MaxNumSwaps,
                                 const double ScaleDownFactor,
                                 const size_t ScreenSize,
                                 const cppbool LambdaU,
                                 const vector[vector[double]]& Lambdas,
                                 const unsigned int nfolds,
                                 const double seed,
                                 const size_t ExcludeFirstK,
                                 const cppbool Intercept,
                                 const cppbool withBounds,
                                 const dvec &Lows,
                                 const dvec &Highs):
    return L0LearnCV[dmat](X, y, Loss, Penalty, Algorithm, NnzStopNum, G_ncols, G_nrows, Lambda2Max, Lambda2Min, PartialSort,
                      MaxIters, rtol, atol, ActiveSet, ActiveSetNum, MaxNumSwaps, ScaleDownFactor, ScreenSize, LambdaU,
                      Lambdas, nfolds, seed, ExcludeFirstK, Intercept, withBounds, Lows, Highs)


cdef cvfitmodel _L0LearnCV_sparse(const sp_dmat& X,
                                 const dvec& y,
                                 const string Loss,
                                 const string Penalty,
                                 const string Algorithm,
                                 const size_t NnzStopNum,
                                 const size_t G_ncols,
                                 const size_t G_nrows,
                                 const double Lambda2Max,
                                 const double Lambda2Min,
                                 const cppbool PartialSort,
                                 const size_t MaxIters,
                                 const double rtol,
                                 const double atol,
                                 const cppbool ActiveSet,
                                 const size_t ActiveSetNum,
                                 const size_t MaxNumSwaps,
                                 const double ScaleDownFactor,
                                 const size_t ScreenSize,
                                 const cppbool LambdaU,
                                 const vector[vector[double]]& Lambdas,
                                 const unsigned int nfolds,
                                 const double seed,
                                 const size_t ExcludeFirstK,
                                 const cppbool Intercept,
                                 const cppbool withBounds,
                                 const dvec &Lows,
                                 const dvec &Highs):
    return L0LearnCV[sp_dmat](X, y, Loss, Penalty, Algorithm, NnzStopNum, G_ncols, G_nrows, Lambda2Max, Lambda2Min, PartialSort,
                      MaxIters, rtol, atol, ActiveSet, ActiveSetNum, MaxNumSwaps, ScaleDownFactor, ScreenSize, LambdaU,
                      Lambdas, nfolds, seed, ExcludeFirstK, Intercept, withBounds, Lows, Highs)
