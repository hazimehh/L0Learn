import pytest
import numpy as np
from scipy.sparse import csc_matrix

import l0learn

N = 50


@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_X_sparse_support(f):
    x = np.random.random(size=(N, N))
    x_sparse = csc_matrix(x)
    y = np.random.random(size=(N,))
    model_fit = f(x_sparse, y, intercept=False)
    assert max(model_fit.support_size[0]) == N


@pytest.mark.parametrize(
    "x",
    [
        np.random.random(size=(N, N, N)),  # Wrong Size
        "A String",  # Wrong Type
        np.random.random(size=(N, N)).astype(complex),  # Wrong dtype
        np.random.random(size=(N, N)).astype(int),  # Wrong dtype
        np.random.random(size=(0, N)),
    ],
)  # degenerate 2D array
@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_X_dense_bad_checks(f, x):
    # Check size of matrix X
    y = np.random.random(size=(N,))
    with pytest.raises(ValueError):
        f(x, y)


@pytest.mark.parametrize(
    "y",
    [
        np.random.random(size=(N, N, N)),  # wrong dimensions
        "A String",  # wrong type
        np.random.random(size=(N,)).astype(complex),  # wrong dtype
        np.random.random(size=(N + 1)),
    ],
)  # wrong size
@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_y_bad_checks(f, y):
    # Check size of matrix X
    x = np.random.random(size=(N, N))
    with pytest.raises(ValueError):
        f(x, y)


@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_loss_bad_checks(f):
    # Check size of matrix X
    x = np.random.random(size=(N, N))
    y = np.random.random(size=(N,))
    with pytest.raises(ValueError):
        f(x, y, loss="NOT A LOSS")


@pytest.mark.parametrize("loss", l0learn.interface.SUPPORTED_LOSS)
@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_loss_good_checks(f, loss):
    # Check size of matrix X
    x = np.random.random(size=(N, N))
    y = np.random.randint(low=0, high=2, size=(N,)).astype(float)
    _ = f(x, y, loss=loss)


@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_penalty_bad_checks(f):
    # Check size of matrix X
    x = np.random.random(size=(N, N))
    y = np.random.random(size=(N,))
    with pytest.raises(ValueError):
        f(x, y, penalty="L0LX")


@pytest.mark.parametrize("penalty", l0learn.interface.SUPPORTED_PENALTY)
@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_penalty_good_checks(f, penalty):
    # Check size of matrix X
    x = np.random.random(size=(N, N))
    y = np.random.random(size=(N,))
    _ = f(x, y, penalty=penalty)


@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_algorithm_bad_checks(f):
    # Check size of matrix X
    x = np.random.random(size=(N, N))
    y = np.random.random(size=(N,))
    with pytest.raises(ValueError):
        f(x, y, algorithm="NOT CD or CDPSI")


@pytest.mark.parametrize("algorithm", l0learn.interface.SUPPORTED_ALGORITHM)
@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_algorithm_good_checks(f, algorithm):
    # Check size of matrix X
    x = np.random.random(size=(N, N))
    y = np.random.random(size=(N,))
    _ = f(x, y, algorithm=algorithm)


@pytest.mark.parametrize("max_support_size", [-1, 2.0])
@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_max_support_size_bad_checks(f, max_support_size):
    # Check size of matrix X
    x = np.random.random(size=(N, N))
    y = np.random.random(size=(N,))

    with pytest.raises(ValueError):
        _ = f(x, y, max_support_size=max_support_size)


@pytest.mark.parametrize("max_support_size", [N, N - 1, N + 1])
@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_max_support_size_good_checks(f, max_support_size):
    # Check size of matrix X
    x = np.random.random(size=(N, N))
    y = np.random.random(size=(N,))
    _ = f(x, y, max_support_size=max_support_size)


@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_gamma_max_bad_checks(f):
    # Check size of matrix X
    x = np.random.random(size=(N, N))
    y = np.random.random(size=(N,))

    with pytest.raises(ValueError):
        _ = f(x, y, gamma_max=-1)


@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_gamma_min_bad_checks(f):
    # Check size of matrix X
    x = np.random.random(size=(N, N))
    y = np.random.random(size=(N,))

    with pytest.raises(ValueError):
        _ = f(x, y, gamma_min=1, gamma_max=0.5)

    with pytest.raises(ValueError):
        _ = f(x, y, gamma_min=-1)


@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_paritial_sort_bad_checks(f):
    # Check size of matrix X
    x = np.random.random(size=(N, N))
    y = np.random.random(size=(N,))

    with pytest.raises(ValueError):
        _ = f(x, y, partial_sort="NOT A BOOL")


@pytest.mark.parametrize("max_iter", [1.0, 0])
@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_max_iter_bad_checks(f, max_iter):
    # Check size of matrix X
    x = np.random.random(size=(N, N))
    y = np.random.random(size=(N,))

    with pytest.raises(ValueError):
        _ = f(x, y, max_iter=max_iter)


@pytest.mark.parametrize("rtol", [1.0, -0.1])
@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_rtol_bad_checks(f, rtol):
    # Check size of matrix X
    x = np.random.random(size=(N, N))
    y = np.random.random(size=(N,))

    with pytest.raises(ValueError):
        _ = f(x, y, rtol=rtol)


@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_atol_bad_checks(f):
    # Check size of matrix X
    x = np.random.random(size=(N, N))
    y = np.random.random(size=(N,))

    with pytest.raises(ValueError):
        _ = f(x, y, atol=-1)


@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_active_set_sort_bad_checks(f):
    # Check size of matrix X
    x = np.random.random(size=(N, N))
    y = np.random.random(size=(N,))

    with pytest.raises(ValueError):
        _ = f(x, y, active_set="NOT A BOOL")


@pytest.mark.parametrize("active_set_num", [1.3, 0])
@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_active_set_num_bad_checks(f, active_set_num):
    # Check size of matrix X
    x = np.random.random(size=(N, N))
    y = np.random.random(size=(N,))

    with pytest.raises(ValueError):
        _ = f(x, y, active_set_num=active_set_num)


@pytest.mark.parametrize("max_swaps", [0, 4.5])
@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_max_swaps_bad_checks(f, max_swaps):
    # Check size of matrix X
    x = np.random.random(size=(N, N))
    y = np.random.random(size=(N,))

    with pytest.raises(ValueError):
        _ = f(x, y, max_swaps=max_swaps)


@pytest.mark.parametrize("scale_down_factor", [-1, 2])
@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_scale_down_factor_bad_checks(f, scale_down_factor):
    # Check size of matrix X
    x = np.random.random(size=(N, N))
    y = np.random.random(size=(N,))

    with pytest.raises(ValueError):
        _ = f(x, y, scale_down_factor=scale_down_factor)


@pytest.mark.parametrize("screen_size", [-1, 2.0])
@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_screen_size_bad_checks(f, screen_size):
    # Check size of matrix X
    x = np.random.random(size=(N, N))
    y = np.random.random(size=(N,))

    with pytest.raises(ValueError):
        _ = f(x, y, screen_size=screen_size)


@pytest.mark.parametrize("exclude_first_k", [-1, 2.0, N + 1])
@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_exclude_first_k_bad_checks(f, exclude_first_k):
    # Check size of matrix X
    x = np.random.random(size=(N, N))
    y = np.random.random(size=(N,))

    with pytest.raises(ValueError):
        _ = f(x, y, exclude_first_k=exclude_first_k)


@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_intercept_bad_checks(f):
    # Check size of matrix X
    x = np.random.random(size=(N, N))
    y = np.random.random(size=(N,))

    with pytest.raises(ValueError):
        _ = f(x, y, intercept="NOT A BOOL")


@pytest.mark.parametrize("loss", l0learn.interface.CLASSIFICATION_LOSS)
@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_classification_loss_bad_y_checks(f, loss):
    # Check size of matrix X
    x = np.random.random(size=(N, N))
    y = np.zeros(N)
    y[0] = 1
    y[1] = 2

    with pytest.raises(ValueError):
        _ = f(x, y, loss=loss)


@pytest.mark.parametrize("loss", l0learn.interface.CLASSIFICATION_LOSS)
@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_classification_loss_bad_lambda_grid_L0_checks(f, loss):
    # Check size of matrix X
    x = np.random.random(size=(N, N))
    y = np.random.randint(0, 2, size=N)
    lambda_grid = [[10], [10]]

    with pytest.raises(ValueError):
        _ = f(
            x,
            y,
            loss=loss,
            penalty="L0",
            lambda_grid=lambda_grid,
            num_gamma=None,
            num_lambda=None,
        )

    _ = f(
        x,
        y,
        loss=loss,
        penalty="L0",
        lambda_grid=[[10]],
        num_gamma=None,
        num_lambda=None,
    )


@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_bad_lambda_grid_L0_checks(f):
    # Check size of matrix X
    x = np.random.random(size=(N, N))
    y = np.random.random(size=N)
    lambda_grid = [[10], [10]]

    with pytest.raises(ValueError):
        _ = f(
            x, y, penalty="L0", lambda_grid=lambda_grid, num_gamma=None, num_lambda=None
        )

    _ = f(x, y, penalty="L0", lambda_grid=[[10]], num_gamma=None, num_lambda=None)


@pytest.mark.parametrize("penalty", ["L0L1", "L0L2"])
@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_regression_loss_bad_num_gamma_L0_checks(f, penalty):
    x = np.random.random(size=(N, N))
    y = np.random.random(size=N)

    with pytest.warns(None) as wrn:
        _ = f(x, y, loss="SquaredError", penalty=penalty, num_gamma=1, num_lambda=10)

    assert len(wrn) == 1

    with pytest.warns(None) as wrn:
        _ = f(x, y, loss="SquaredError", penalty=penalty, num_gamma=2, num_lambda=10)

    assert len(wrn) == 0


@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_auto_lambda_bad_checks(f):
    x = np.random.random(size=(N, N))
    y = np.random.random(size=N)

    with pytest.raises(ValueError):
        _ = f(x, y, penalty="L0L1", num_gamma=3.0, num_lambda=5)

    with pytest.raises(ValueError):
        _ = f(x, y, penalty="L0L1", num_gamma=-2, num_lambda=5)

    with pytest.raises(ValueError):
        _ = f(x, y, penalty="L0L1", num_gamma=5, num_lambda=5.0)

    with pytest.raises(ValueError):
        _ = f(x, y, penalty="L0L1", num_gamma=5, num_lambda=-2)

    with pytest.raises(ValueError):
        _ = f(x, y, penalty="L0", num_gamma=2, num_lambda=10)

    _ = f(x, y, penalty="L0", num_gamma=1, num_lambda=10)


@pytest.mark.parametrize("penalty", ["L0L1", "L0L2"])
@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_lambda_grid_bad_over_defined_checks(f, penalty):
    x = np.random.random(size=(N, N))
    y = np.random.random(size=N)

    with pytest.raises(ValueError):
        _ = f(
            x,
            y,
            penalty=penalty,
            lambda_grid=[[10], [10]],
            num_lambda=1,
            num_gamma=None,
        )

    with pytest.raises(ValueError):
        _ = f(
            x,
            y,
            penalty=penalty,
            lambda_grid=[[10], [10]],
            num_lambda=None,
            num_gamma=2,
        )

    _ = f(
        x, y, penalty=penalty, lambda_grid=[[10], [10]], num_lambda=None, num_gamma=None
    )


@pytest.mark.parametrize(
    "penalty_lambda_grid",
    [
        ("L0L1", [[-1]]),
        ("L0", [[10, 11]]),
    ],
)
@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_lambda_grid_bad_checks(f, penalty_lambda_grid):
    penalty, lambda_grid = penalty_lambda_grid
    x = np.random.random(size=(N, N))
    y = np.random.random(size=N)

    with pytest.raises(ValueError):
        _ = f(
            x,
            y,
            penalty=penalty,
            lambda_grid=lambda_grid,
            num_gamma=None,
            num_lambda=None,
        )


@pytest.mark.parametrize(
    "bounds",
    [
        ("NOT A FLOAT", 1.0),
        (1.0, "NOT A FLOAT"),
        (1.0, 1.0),
        (-np.ones((N, 2)), 1.0),
        (-np.ones(N + 1), 1.0),
        (np.ones(N), 1.0),
        (-1.0, -1.0),
        (-1.0, np.ones((N, 2))),
        (-1.0, np.ones(N + 1)),
        (-1.0, -1.0 * np.ones(N)),
        (0.0, 0.0),
        (np.zeros(N), np.zeros(N)),
    ],
)
@pytest.mark.parametrize("f", [l0learn.fit, l0learn.cvfit])
def test_with_bounds_bad_checks(f, bounds):
    lows, highs = bounds

    x = np.random.random(size=(N, N))
    y = np.random.random(size=N)

    with pytest.raises(ValueError):
        _ = f(x, y, lows=lows, highs=highs)


@pytest.mark.parametrize("num_folds", [-1, 0, 1, N + 1, 2.0])
def test_cvfit_num_folds_bad_check(num_folds):
    x = np.random.random(size=(N, N))
    y = np.random.random(size=N)

    with pytest.raises(ValueError):
        _ = l0learn.cvfit(x, y, num_folds=num_folds)
