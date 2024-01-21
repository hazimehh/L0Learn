from typing import Union, List, Sequence, Dict, Any, Optional

from .models import FitModel, CVFitModel
from .l0learn import (
    _L0LearnFit_sparse,
    _L0LearnFit_dense,
    _L0LearnCV_dense,
    _L0LearnCV_sparse,
)
from scipy.sparse import csc_matrix
import numpy as np
from warnings import warn

SUPPORTED_LOSS = ("SquaredError", "Logistic", "SquaredHinge")
CLASSIFICATION_LOSS = SUPPORTED_LOSS[1], SUPPORTED_LOSS[2]
SUPPORTED_PENALTY = ("L0", "L0L1", "L0L2")
SUPPORTED_ALGORITHM = ("CD", "CDPSI")


def _fit_check(
    X: Union[np.ndarray, csc_matrix],
    y: np.ndarray,
    loss: str,
    penalty: str,
    algorithm: str,
    max_support_size: int,
    num_lambda: Union[int, None],
    num_gamma: Union[int, None],
    gamma_max: float,
    gamma_min: float,
    partial_sort: bool,
    max_iter: int,
    rtol: float,
    atol: float,
    active_set: bool,
    active_set_num: int,
    max_swaps: int,
    scale_down_factor: float,
    screen_size: int,
    lambda_grid: Union[List[Sequence[float]], None],
    exclude_first_k: int,
    intercept: bool,
    lows: Union[np.ndarray, float],
    highs: Union[np.ndarray, float],
) -> Dict[str, Any]:
    if isinstance(X, (np.ndarray, csc_matrix)):
        if X.dtype != np.float64:
            raise ValueError(
                f"expected X to have dtype {np.float64}, but got {X.dtype}"
            )
        if X.ndim != 2:
            raise ValueError(f"expected X to be 2D, but got {X.ndim}D")
        if not np.prod(X.shape):
            raise ValueError(
                f"expected X to have non-degenerate axis, but got {X.shape}"
            )
        # if isinstance(X, np.ndarray):
        #     if not X.flags.contiguous:
        #         raise ValueError(f"expected X to be contiguous, but is not.")
        # else: # isinstance(X, csc_matrix):
        #     if not (X.indptr.flags.contiguous and X.indices.flags.contiguous and X.data.flags.continuous):
        #         raise ValueError(f"expected X to have contiguous `indptr`, `indices`, and `data`, but is not.")
    else:
        raise ValueError(
            f"expected X to be a {np.ndarray} or a {csc_matrix}, but got {type(X)}."
        )

    n, p = X.shape
    if (
        not isinstance(y, np.ndarray)
        or not np.isrealobj(y)
        or y.ndim != 1
        or len(y) != n
    ):
        raise ValueError(f"expected y to be a 1D real numpy, but got {y}.")
    if loss not in SUPPORTED_LOSS:
        raise ValueError(
            f"expected loss parameter to be on of {SUPPORTED_LOSS}, but got {loss}"
        )
    if penalty not in SUPPORTED_PENALTY:
        raise ValueError(
            f"expected penalty parameter to be on of {SUPPORTED_PENALTY}, but got {penalty}"
        )
    if algorithm not in SUPPORTED_ALGORITHM:
        raise ValueError(
            f"expected algorithm parameter to be on of {SUPPORTED_ALGORITHM}, but got {algorithm}"
        )
    if not isinstance(max_support_size, int) or 1 > max_support_size:
        raise ValueError(
            f"expected max_support_size parameter to be a positive integer, but got {max_support_size}"
        )
    max_support_size = min(p, max_support_size)

    if gamma_max < 0:
        raise ValueError(
            f"expected gamma_max parameter to be a positive float, but got {gamma_max}"
        )
    if gamma_min < 0 or gamma_min > gamma_max:
        raise ValueError(
            f"expected gamma_max parameter to be a positive float less than gamma_max,"
            f" but got {gamma_min}"
        )
    if not isinstance(partial_sort, bool):
        raise ValueError(
            f"expected partial_sort parameter to be a bool, but got {partial_sort}"
        )
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError(
            f"expected max_iter parameter to be a positive integer, but got {max_iter}"
        )
    if rtol < 0 or rtol >= 1:
        raise ValueError(f"expected rtol parameter to exist in [0, 1), but got {rtol}")
    if atol < 0:
        raise ValueError(
            f"expected atol parameter to exist in [0, INF), but got {atol}"
        )
    if not isinstance(active_set, bool):
        raise ValueError(
            f"expected active_set parameter to be a bool, but got {active_set}"
        )
    if not isinstance(active_set_num, int) or active_set_num < 1:
        raise ValueError(
            f"expected active_set_num parameter to be a positive integer, but got {active_set_num}"
        )
    if not isinstance(max_swaps, int) or max_swaps < 1:
        raise ValueError(
            f"expected max_swaps parameter to be a positive integer, but got {max_swaps}"
        )
    if not (0 < scale_down_factor < 1):
        raise ValueError(
            f"expected scale_down_factor parameter to exist in (0, 1), but got {scale_down_factor}"
        )
    if not isinstance(screen_size, int) or screen_size < 1:
        raise ValueError(
            f"expected screen_size parameter to be a positive integer, but got {screen_size}"
        )
    screen_size = min(screen_size, p)

    if not isinstance(exclude_first_k, int) or not (0 <= exclude_first_k <= p):
        raise ValueError(
            f"expected exclude_first_k parameter to be a positive integer less than {p}, "
            f"but got {exclude_first_k}"
        )
    if not isinstance(intercept, bool):
        raise ValueError(
            f"expected intercept parameter to be a bool, " f"but got {intercept}"
        )

    if loss in CLASSIFICATION_LOSS:
        unique_items = sorted(np.unique(y))
        if len(unique_items) != 2:
            raise ValueError(
                f"expected y vector to only have two unique values (Binary Classification), "
                f"but got {unique_items}"
            )
        else:
            a, *_ = unique_items  # a is the lower value
            y = np.copy(y)
            first_value = y == a
            second_value = y != a
            y[first_value] = -1.0
            y[second_value] = 1.0
            if y.dtype != np.float64:
                y = y.astype(float)

        if penalty == "L0":
            # Pure L0 is not supported for classification
            # Below we add a small L2 component.

            if lambda_grid is not None and len(lambda_grid) != 1:
                # If this error checking was left to the lower section, it would confuse users as
                # we are converting L0 to L0L2 with small L2 penalty.
                # Here we must check if lambdaGrid is supplied (And thus use 'autolambda')
                # If 'lambdaGrid' is supplied, we must only supply 1 list of lambda values
                raise ValueError(
                    f"L0 Penalty requires 'lambda_grid' to be a list of length 1, but got {lambda_grid}."
                )

            penalty = "L0L2"
            gamma_max = 1e-7
            gamma_min = 1e-7
        elif penalty != "L0" and num_gamma == 1:
            warn(
                f"num_gamma set to 1 with {penalty} penalty. Only one {penalty[2:]} penalty value will be fit."
            )

    if y.dtype != np.float64:
        raise ValueError(
            f"expected y vector to have type {np.float64}, but got {y.dtype}"
        )

    if lambda_grid is None:
        lambda_grid = [[0.0]]
        auto_lambda = True
        if not isinstance(num_lambda, int) or num_lambda < 1:
            raise ValueError(
                f"expected num_lambda to a positive integer when lambda_grid is None, but got {num_lambda}."
            )
        if not isinstance(num_gamma, int) or num_gamma < 1:
            raise ValueError(
                f"expected num_gamma to a positive integer when lambda_grid is None, but got {num_gamma}."
            )
        if penalty == "L0" and num_gamma != 1:
            raise ValueError(
                f"expected num_gamma to 1 when penalty = 'L0', but got {num_gamma}."
            )
    else:  # lambda_grid should be a List[List[float]]
        if num_gamma is not None:
            raise ValueError(
                f"expected num_gamma to be None if lambda_grid is specified by the user, "
                f"but got {num_gamma}"
            )
        num_gamma = len(lambda_grid)

        if num_lambda is not None:
            raise ValueError(
                f"expected num_lambda to be None if lambda_grid is specified by the user, "
                f"but got {num_lambda}"
            )
        num_lambda = 0  # This value is ignored.
        auto_lambda = False

        if penalty == "L0" and num_gamma != 1:
            raise ValueError(
                f"expected lambda_grid to of length 1 when penalty = 'L0', but got {len(lambda_grid)}"
            )

        for i, sub_lambda_grid in enumerate(lambda_grid):
            if sub_lambda_grid[0] <= 0:
                raise ValueError(
                    f"Expected all values of lambda_grid to be positive, "
                    f"but got lambda_grid[{i}] containing a negative value"
                )
            if any(np.diff(sub_lambda_grid) >= 0):
                raise ValueError(
                    f"Expected each element of lambda_grid to be a list of decreasing value, "
                    f"but got lambda_grid[{i}] containing an increasing value."
                )

    n, p = X.shape
    with_bounds = False

    if isinstance(lows, float):
        if lows > 0:
            raise ValueError(
                f"expected lows to be a non-positive float, but got {lows}"
            )
        elif lows > -float("inf"):
            with_bounds = True
    elif (
        isinstance(lows, np.ndarray)
        and lows.ndim == 1
        and len(lows) == p
        and all(lows <= 0)
    ):
        with_bounds = True
    else:
        raise ValueError(
            f"expected lows to be a non-positive float, or a 1D numpy array of length {p} of non-positives "
            f"floats, but got {lows}"
        )

    if isinstance(highs, float):
        if highs < 0:
            raise ValueError(
                f"expected highs to be a non-negative float, but got {highs}"
            )
        if highs < float("inf"):
            with_bounds = True
    elif (
        isinstance(highs, np.ndarray)
        and highs.ndim == 1
        and len(highs) == p
        and all(highs >= 0)
    ):
        with_bounds = True
    else:
        raise ValueError(
            f"expected highs to be a non-negative float, or a 1D numpy array of length {p} of "
            f"non-negative floats, but got {highs}"
        )

    if with_bounds:
        if isinstance(lows, float):
            lows = np.ones(p) * lows
        if isinstance(highs, float):
            highs = np.ones(p) * highs

        if any(lows >= highs):
            bad_bounds = np.argwhere(lows >= highs)
            raise ValueError(
                f"expected to be high to be elementwise greater than lows, "
                f"but got indices {bad_bounds[0]} where that is not the case "
            )
    else:
        lows = np.array([0.0])
        highs = np.array(([0.0]))

    return {
        "max_support_size": max_support_size,
        "screen_size": screen_size,
        "y": y,
        "penalty": penalty,
        "gamma_max": float(gamma_max),
        "gamma_min": float(gamma_min),
        "lambda_grid": [[float(i) for i in lst] for lst in lambda_grid],
        "num_gamma": num_gamma,
        "num_lambda": num_lambda,
        "auto_lambda": auto_lambda,
        "with_bounds": with_bounds,
        "lows": lows.astype("float"),
        "highs": highs.astype("float"),
    }


def fit(
    X: Union[np.ndarray, csc_matrix],
    y: np.ndarray,
    loss: str = "SquaredError",
    penalty: str = "L0",
    algorithm: str = "CD",
    max_support_size: int = 100,
    num_lambda: Optional[int] = 100,
    num_gamma: Optional[int] = 1,
    gamma_max: float = 10.0,
    gamma_min: float = 0.0001,
    partial_sort: bool = True,
    max_iter: int = 200,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    active_set: bool = True,
    active_set_num: int = 3,
    max_swaps: int = 100,
    scale_down_factor: float = 0.8,
    screen_size: int = 1000,
    lambda_grid: Optional[List[Sequence[float]]] = None,
    exclude_first_k: int = 0,
    intercept: bool = True,
    lows: Union[np.ndarray, float] = -float("inf"),
    highs: Union[np.ndarray, float] = +float("inf"),
) -> FitModel:
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

    max_support_size : int
        Must be greater than 0.
        The maximum support size at which to terminate the regularization path. We recommend setting this to a small
        fraction of min(n,p) (e.g. 0.05 * min(n,p)) as L0 regularization typically selects a small portion of non-zeros.

    num_lambda : int, optional
        The number of lambda values to select in the regularization path.
        This value must be None if lambda_grid is supplied.When supplied, must be greater than 0.
        Note: lambda is the regularization parameter corresponding to the L0 norm.

    num_gamma: int, optional
        The number of gamma values to select in the regularization path.
        This value must be None if lambda_grid is supplied. When supplied, must be greater than 0.
        Note:  gamma is the regularization parameter corresponding to L1 or L2, depending on the chosen penalty).

    gamma_max : float
        The maximum value of gamma when using the L0L2 penalty.
        This value must be greater than 0.

        Note: For the L0L1 penalty this is automatically selected.

    gamma_min : float
        The minimum value of Gamma when using the L0L2 penalty.
        This value must be greater than 0 but less than gamma_max.
        Note: For the L0L1 penalty, the minimum value of gamma in the grid is set to gammaMin * gammaMax.

    partial_sort : bool
        If TRUE partial sorting will be used for sorting the coordinates to do greedy cycling (see our paper for
        for details). Otherwise, full sorting is used. #TODO: Add link for paper

    max_iter : int
        The maximum number of iterations (full cycles) for CD per grid point. The algorithm may not use the full number
        of iteration per grid point if convergence is found (defined by rtol and atol parameter)
        Must be greater than 0

    rtol : float
        The relative tolerance which decides when to terminate optimization as based on the relative change in the
        objective between iterations.
        Must be greater than 0 and less than 1.

    atol : float
        The absolute tolerance which decides when to terminate optimization as based on the absolute L2 norm of the
        residuals
        Must be greater than 0

    active_set : bool
        If TRUE, performs active set updates. (see our paper for for details). #TODO: Add link for paper

    active_set_num : int
        The number of consecutive times a support should appear before declaring support stabilization.
        (see our paper for for details). #TODO: Add link for paper

        Must be greater than 0.

    max_swaps : int
        The maximum number of swaps used by CDPSI for each grid point.
        Must be greater than 0. Ignored by CD algorithims.

    scale_down_factor : float
        Roughly amount each lambda value is scaled by between grid points. Larger values lead to closer lambdas and
        typically to smaller gaps between the support sizes.

        For details, see our paper - Section 5 on Adaptive Selection of Tuning Parameters). #TODO: Add link for paper

        Must be greater than 0 and less than 1 (strictly for both.)

    screen_size : int
        The number of coordinates to cycle over when performing initial correlation screening. #TODO: Add link for paper

        Must be greater than 0 and less than number of columns of X.

    lambda_grid : list of list of floats
        A grid of lambda values to use in computing the regularization path. This is by default an empty list
        and is ignored. When specified, lambda_grid should be a list of list of floats, where the ith element
         (corresponding to the ith gamma) should be a decreasing sequence of lambda values. The length of this sequence
         is directly the number of lambdas to be tried for that gamma.

        In the the "L0" penalty case, lambda_grid should be a list of 1.
        In the "L0LX" penalty cases, lambda_grid can be a list of any length. The length of lambda_grid will be the
        number of gamma values tried.

        See the example notebook for more details.

        Note: When lambda_grid is supplied, num_gamma and num_lambda must be None.

    exclude_first_k : int
        The first exclude_first_k features in X will be excluded from variable selection. In other words, the first
        exclude_first_k variables will not be included in the L0-norm penalty however they will  be included in the
        L1 or L2 norm penalties, if they are specified.

        Must be a positive integer less than the columns of X.

    intercept : bool
        If False, no intercept term is included or fit in the regularization path
        Intercept terms are not regularized by L0 or L1/L2.

    lows : np array or float
        Lower bounds for coefficients. Either a scalar for all coefficients to have the same bound or a vector of
        size p (number of columns of X) where lows[i] is the lower bound for coefficient i.

        Lower bounds can not be above 0 (i.e. we can not specify that all coefficients must be larger than a > 0).
        Lower bounds can be set to 0 iff the corresponding upper bound for that coefficient is also not 0.

    highs : np array or float
        Upper bounds for coefficients. Either a scalar for all coefficients to have the same bound or a vector of
        size p (number of columns of X) where highs[i] is the upper bound for coefficient i.

        Upper bounds can not be below 0 (i.e. we can not specify that all coefficients must be smaller than a < 0).
        Upper bounds can be set to 0 iff the corresponding lower bound for that coefficient is also not 0.

    Returns
    -------
    fit_model : l0learn.models.FitModel
        FitModel instance containing all relevant information from the solution path.

    See Also
    -------
    l0learn.cvfit
    l0learn.models.FitModel

    Examples
    --------
    >>>fit_model = l0learn.fit(X, y, penalty="L0", max_support_size=20)
    """
    check = _fit_check(
        X=X,
        y=y,
        loss=loss,
        penalty=penalty,
        algorithm=algorithm,
        max_support_size=max_support_size,
        num_lambda=num_lambda,
        num_gamma=num_gamma,
        gamma_max=gamma_max,
        gamma_min=gamma_min,
        partial_sort=partial_sort,
        max_iter=max_iter,
        rtol=rtol,
        atol=atol,
        active_set=active_set,
        active_set_num=active_set_num,
        max_swaps=max_swaps,
        scale_down_factor=scale_down_factor,
        screen_size=screen_size,
        lambda_grid=lambda_grid,
        exclude_first_k=exclude_first_k,
        intercept=intercept,
        lows=lows,
        highs=highs,
    )

    max_support_size = check["max_support_size"]
    screen_size = check["screen_size"]
    y = check["y"]
    penalty = check["penalty"]
    gamma_max = check["gamma_max"]
    gamma_min = check["gamma_min"]
    lambda_grid = check["lambda_grid"]
    num_gamma = check["num_gamma"]
    num_lambda = check["num_lambda"]
    auto_lambda = check["auto_lambda"]
    with_bounds = check["with_bounds"]
    lows = check["lows"]
    highs = check["highs"]

    if isinstance(X, np.ndarray):
        c_results = _L0LearnFit_dense(
            X,  # const T &X,
            y,  # const arma::vec &y,
            loss,  # const std::string Loss,
            penalty,  # const std::string Penalty,
            algorithm,  # const std::string Algorithm,
            max_support_size,  # const std::size_t NnzStopNum,
            num_lambda,  # const std::size_t G_ncols,
            num_gamma,  # const std::size_t G_nrows,
            gamma_max,  # const double Lambda2Max,
            gamma_min,  # const double Lambda2Min,
            partial_sort,  # const bool PartialSort,
            max_iter,  # const std::size_t MaxIters,
            rtol,  # const double rtol,
            atol,  # const double atol,
            active_set,  # const bool ActiveSet,
            active_set_num,  # const std::size_t ActiveSetNum,
            max_swaps,  # const std::size_t MaxNumSwaps,
            scale_down_factor,  # const double ScaleDownFactor,
            screen_size,  # const std::size_t ScreenSize,
            not auto_lambda,  # const bool LambdaU,
            lambda_grid,  # const std::vector<std::vector<double>> &Lambdas,
            exclude_first_k,  # const std::size_t ExcludeFirstK,
            intercept,  # const bool Intercept,
            with_bounds,  # const bool withBounds,
            lows,  # const arma::vec &Lows,
            highs,
        )  # const arma::vec &Highs
    elif isinstance(X, csc_matrix):
        c_results = _L0LearnFit_sparse(
            X,  # const T &X,
            y,  # const arma::vec &y,
            loss,  # const std::string Loss,
            penalty,  # const std::string Penalty,
            algorithm,  # const std::string Algorithm,
            max_support_size,  # const std::size_t NnzStopNum,
            num_lambda,  # const std::size_t G_ncols,
            num_gamma,  # const std::size_t G_nrows,
            gamma_max,  # const double Lambda2Max,
            gamma_min,  # const double Lambda2Min,
            partial_sort,  # const bool PartialSort,
            max_iter,  # const std::size_t MaxIters,
            rtol,  # const double rtol,
            atol,  # const double atol,
            active_set,  # const bool ActiveSet,
            active_set_num,  # const std::size_t ActiveSetNum,
            max_swaps,  # const std::size_t MaxNumSwaps,
            scale_down_factor,  # const double ScaleDownFactor,
            screen_size,  # const std::size_t ScreenSize,
            not auto_lambda,  # const bool LambdaU,
            lambda_grid,  # const std::vector<std::vector<double>> &Lambdas,
            exclude_first_k,  # const std::size_t ExcludeFirstK,
            intercept,  # const bool Intercept,
            with_bounds,  # const bool withBounds,
            lows,  # const arma::vec &Lows,
            highs,
        )  # const arma::vec &Highs

    results = FitModel(
        settings={"loss": loss, "intercept": intercept, "penalty": penalty},
        lambda_0=c_results.Lambda0,
        gamma=c_results.Lambda12,
        support_size=c_results.NnzCount,
        coeffs=c_results.Beta,
        intercepts=c_results.Intercept,
        converged=c_results.Converged,
    )
    return results


def cvfit(
    X: Union[np.ndarray, csc_matrix],
    y: np.ndarray,
    loss: str = "SquaredError",
    penalty: str = "L0",
    algorithm: str = "CD",
    num_folds: int = 10,
    seed: int = 1,
    max_support_size: int = 100,
    num_lambda: Optional[int] = 100,
    num_gamma: Optional[int] = 1,
    gamma_max: float = 10.0,
    gamma_min: float = 0.0001,
    partial_sort: bool = True,
    max_iter: int = 200,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    active_set: bool = True,
    active_set_num: int = 3,
    max_swaps: int = 100,
    scale_down_factor: float = 0.8,
    screen_size: int = 1000,
    lambda_grid: Optional[List[Sequence[float]]] = None,
    exclude_first_k: int = 0,
    intercept: bool = True,
    lows: Union[np.ndarray, float] = -float("inf"),
    highs: Union[np.ndarray, float] = +float("inf"),
) -> CVFitModel:
    """Computes the regularization path for the specified loss function and penalty function and performs K-fold
    cross-validation.

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

    num_folds : int
        Must be greater than 1 and less than N (number of .
        The number of folds for cross-validation.

    max_support_size : int
        Must be greater than 0.
        The maximum support size at which to terminate the regularization path. We recommend setting this to a small
        fraction of min(n,p) (e.g. 0.05 * min(n,p)) as L0 regularization typically selects a small portion of non-zeros.

    num_lambda : int, optional
        The number of lambda values to select in the regularization path.
        This value must be None if lambda_grid is supplied.When supplied, must be greater than 0.
        Note: lambda is the regularization parameter corresponding to the L0 norm.

    num_gamma: int, optional
        The number of gamma values to select in the regularization path.
        This value must be None if lambda_grid is supplied. When supplied, must be greater than 0.
        Note:  gamma is the regularization parameter corresponding to L1 or L2, depending on the chosen penalty).

    gamma_max : float
        The maximum value of gamma when using the L0L2 penalty.
        This value must be greater than 0.

        Note: For the L0L1 penalty this is automatically selected.

    gamma_min : float
        The minimum value of Gamma when using the L0L2 penalty.
        This value must be greater than 0 but less than gamma_max.
        Note: For the L0L1 penalty, the minimum value of gamma in the grid is set to gammaMin * gammaMax.

    partial_sort : bool
        If TRUE partial sorting will be used for sorting the coordinates to do greedy cycling (see our paper for
        for details). Otherwise, full sorting is used. #TODO: Add link for paper

    max_iter : int
        The maximum number of iterations (full cycles) for CD per grid point. The algorithm may not use the full number
        of iteration per grid point if convergence is found (defined by rtol and atol parameter)
        Must be greater than 0

    rtol : float
        The relative tolerance which decides when to terminate optimization as based on the relative change in the
        objective between iterations.
        Must be greater than 0 and less than 1.

    atol : float
        The absolute tolerance which decides when to terminate optimization as based on the absolute L2 norm of the
        residuals
        Must be greater than 0

    active_set : bool
        If TRUE, performs active set updates. (see our paper for for details). #TODO: Add link for paper

    active_set_num : int
        The number of consecutive times a support should appear before declaring support stabilization.
        (see our paper for for details). #TODO: Add link for paper

        Must be greater than 0.

    max_swaps : int
        The maximum number of swaps used by CDPSI for each grid point.
        Must be greater than 0. Ignored by CD algorithims.

    scale_down_factor : float
        Roughly amount each lambda value is scaled by between grid points. Larger values lead to closer lambdas and
        typically to smaller gaps between the support sizes.

        For details, see our paper - Section 5 on Adaptive Selection of Tuning Parameters). #TODO: Add link for paper

        Must be greater than 0 and less than 1 (strictly for both.)

    screen_size : int
        The number of coordinates to cycle over when performing initial correlation screening. #TODO: Add link for paper

        Must be greater than 0 and less than number of columns of X.

    lambda_grid : list of list of floats
        A grid of lambda values to use in computing the regularization path. This is by default an empty list
        and is ignored. When specified, lambda_grid should be a list of list of floats, where the ith element
         (corresponding to the ith gamma) should be a decreasing sequence of lambda values. The length of this sequence
         is directly the number of lambdas to be tried for that gamma.

        In the the "L0" penalty case, lambda_grid should be a list of 1.
        In the "L0LX" penalty cases, lambda_grid can be a list of any length. The length of lambda_grid will be the
        number of gamma values tried.

        See the example notebook for more details.

        Note: When lambda_grid is supplied, num_gamma and num_lambda must be None.

    exclude_first_k : int
        The first exclude_first_k features in X will be excluded from variable selection. In other words, the first
        exclude_first_k variables will not be included in the L0-norm penalty however they will  be included in the
        L1 or L2 norm penalties, if they are specified.

        Must be a positive integer less than the columns of X.

    intercept : bool
        If False, no intercept term is included or fit in the regularization path
        Intercept terms are not regularized by L0 or L1/L2.

    lows : np array or float
        Lower bounds for coefficients. Either a scalar for all coefficients to have the same bound or a vector of
        size p (number of columns of X) where lows[i] is the lower bound for coefficient i.

        Lower bounds can not be above 0 (i.e. we can not specify that all coefficients must be larger than a > 0).
        Lower bounds can be set to 0 iff the corresponding upper bound for that coefficient is also not 0.

    highs : np array or float
        Upper bounds for coefficients. Either a scalar for all coefficients to have the same bound or a vector of
        size p (number of columns of X) where highs[i] is the upper bound for coefficient i.

        Upper bounds can not be below 0 (i.e. we can not specify that all coefficients must be smaller than a < 0).
        Upper bounds can be set to 0 iff the corresponding lower bound for that coefficient is also not 0.

    Returns
    -------
    fit_model : l0learn.models.FitModel
        FitModel instance containing all relevant information from the solution path.

    See Also
    -------
    l0learn.cvfit
    l0learn.models.FitModel

    Examples
    --------
    >>>fit_model = l0learn.fit(X, y, penalty="L0", max_support_size=20)
    """

    check = _fit_check(
        X=X,
        y=y,
        loss=loss,
        penalty=penalty,
        algorithm=algorithm,
        max_support_size=max_support_size,
        num_lambda=num_lambda,
        num_gamma=num_gamma,
        gamma_max=gamma_max,
        gamma_min=gamma_min,
        partial_sort=partial_sort,
        max_iter=max_iter,
        rtol=rtol,
        atol=atol,
        active_set=active_set,
        active_set_num=active_set_num,
        max_swaps=max_swaps,
        scale_down_factor=scale_down_factor,
        screen_size=screen_size,
        lambda_grid=lambda_grid,
        exclude_first_k=exclude_first_k,
        intercept=intercept,
        lows=lows,
        highs=highs,
    )

    max_support_size = check["max_support_size"]
    screen_size = check["screen_size"]
    y = check["y"]
    penalty = check["penalty"]
    gamma_max = check["gamma_max"]
    gamma_min = check["gamma_min"]
    lambda_grid = check["lambda_grid"]
    num_gamma = check["num_gamma"]
    num_lambda = check["num_lambda"]
    auto_lambda = check["auto_lambda"]
    with_bounds = check["with_bounds"]
    lows = check["lows"]
    highs = check["highs"]

    n, p = X.shape

    if not isinstance(num_folds, int) or num_folds < 2 or num_folds > n:
        raise ValueError(
            f"expected num_folds parameter to be a positive integer less than {n}, but got {num_folds}"
        )

    if isinstance(X, np.ndarray):
        c_results = _L0LearnCV_dense(
            X,
            y,
            loss,
            penalty,
            algorithm,
            max_support_size,
            num_lambda,
            num_gamma,
            gamma_max,
            gamma_min,
            partial_sort,
            max_iter,
            rtol,
            atol,
            active_set,
            active_set_num,
            max_swaps,
            scale_down_factor,
            screen_size,
            not auto_lambda,
            lambda_grid,
            num_folds,
            seed,
            exclude_first_k,
            intercept,
            with_bounds,
            lows,
            highs,
        )
    elif isinstance(X, csc_matrix):
        c_results = _L0LearnCV_sparse(
            X,
            y,
            loss,
            penalty,
            algorithm,
            max_support_size,
            num_lambda,
            num_gamma,
            gamma_max,
            gamma_min,
            partial_sort,
            max_iter,
            rtol,
            atol,
            active_set,
            active_set_num,
            max_swaps,
            scale_down_factor,
            screen_size,
            auto_lambda,
            lambda_grid,
            num_folds,
            seed,
            exclude_first_k,
            intercept,
            with_bounds,
            lows,
            highs,
        )

    results = CVFitModel(
        settings={"loss": loss, "intercept": intercept, "penalty": penalty},
        lambda_0=c_results.Lambda0,
        gamma=c_results.Lambda12,
        support_size=c_results.NnzCount,
        coeffs=c_results.Beta,
        intercepts=c_results.Intercept,
        converged=c_results.Converged,
        cv_means=c_results.CVMeans,
        cv_sds=c_results.CVSDs,
    )
    return results
