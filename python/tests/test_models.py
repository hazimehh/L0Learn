import numpy as np
import pandas as pd
from hypothesis.strategies import floats
from scipy.sparse import csc_matrix

from l0learn.models import FitModel, regularization_loss

from pytest import fixture
from hypothesis import given
from hypothesis.extra import numpy as npst


@fixture
def sample_FitModel(loss: str = 'SquaredError', intercept: bool = True, penalty: str = 'L0L1'):
    i = intercept
    beta1 = csc_matrix([[0, 0],
                        [1, 2]])
    beta2 = csc_matrix([[0],
                        [3]])
    beta3 = csc_matrix([[0, 0, 0],
                        [4, 5, 6]])

    return FitModel(settings={'loss': loss, 'intercept': intercept, 'penalty': penalty},
                    lambda_0=[[10, 5], [10], [10, 5, 0]],
                    gamma=[2, 1, 0],
                    support_size=[[0, 1], [3], [4, 5, 6]],
                    coeffs=[beta1, beta2, beta3],
                    intercepts=[[-1*i, -2*i], [-3*i], [-4*i, -5*i, -6*i]],
                    converged=[[True, True], [False], [True, False, True]])


def test_FitModel_coeff(sample_FitModel):
    np.testing.assert_array_equal(sample_FitModel.coeff().toarray(),
                                  np.array([[-1, -2, -3, -4, -5, -6],
                                            [0,  0,  0,  0,  0,  0],
                                            [1,  2,  3,  4,  5,  6]]))
    np.testing.assert_array_equal(sample_FitModel.coeff(include_intercept=False).toarray(),
                                  np.array([[0, 0, 0, 0, 0, 0],
                                            [1, 2, 3, 4, 5, 6]]))

    np.testing.assert_array_equal(sample_FitModel.coeff(gamma=1).toarray(), np.array([[-3, ],
                                                                                      [0, ],
                                                                                      [3, ]]))
    np.testing.assert_array_equal(sample_FitModel.coeff(gamma=1, include_intercept=False).toarray(),
                                  np.array([[0, ],
                                            [3, ]]))

    np.testing.assert_array_equal(sample_FitModel.coeff(lambda_0=6).toarray(), np.array([[-2, ],
                                                                                         [0, ],
                                                                                         [2, ]]))

    np.testing.assert_array_equal(sample_FitModel.coeff(lambda_0=6, include_intercept=False).toarray(),
                                  np.array([[0, ],
                                            [2, ]]))

    np.testing.assert_array_equal(sample_FitModel.coeff(gamma=0, lambda_0=6).toarray(), np.array([[-5, ],
                                                                                                  [0, ],
                                                                                                  [5, ]]))

    np.testing.assert_array_equal(sample_FitModel.coeff(gamma=0, lambda_0=6, include_intercept=False).toarray(),
                                  np.array([[0, ],
                                            [5, ]]))


def test_characteristics(sample_FitModel):
    pd.testing.assert_frame_equal(sample_FitModel.characteristics(),
                                  pd.DataFrame({'l0':           [10,  5, 10, 10,  5,  0],
                                                'support_size':  [0,  1,  3,  4,  5,  6],
                                                'intercept':    [-1, -2, -3, -4, -5, -6],
                                                'converged':    [True, True, False, True, False, True],
                                                'l1':            [2,  2,  1,  0,  0,  0]}))

    pd.testing.assert_frame_equal(sample_FitModel.characteristics(gamma=1),
                                  pd.DataFrame({'l0':           [10],
                                                'support_size':  [3],
                                                'intercept':    [-3],
                                                'converged':    [False],
                                                'l1':            [1]}))

    pd.testing.assert_frame_equal(sample_FitModel.characteristics(lambda_0=6),
                                  pd.DataFrame({'l0':           [5],
                                                'support_size':  [1],
                                                'intercept':    [-2],
                                                'converged':    [True],
                                                'l1':            [2]}))

    pd.testing.assert_frame_equal(sample_FitModel.characteristics(gamma=0, lambda_0=5),
                                  pd.DataFrame({'l0':           [5],
                                                'support_size':  [5],
                                                'intercept':    [-5],
                                                'converged':    [False],
                                                'l1':            [0]}))


@given(coeffs=npst.arrays(dtype=np.float,
                          elements=floats(allow_nan=False, allow_infinity=False, max_value=1e100, min_value=-1e100),
                          shape=npst.array_shapes(min_dims=2, max_dims=2)))
def test_regularization_loss(coeffs):
    coeffs_csc = csc_matrix(coeffs)
    if coeffs.shape[1] == 1:
        np.testing.assert_equal((coeffs != 0).sum(), regularization_loss(coeffs_csc, l0=1))
        np.testing.assert_equal((coeffs != 0).sum() + sum(abs(coeffs)) + sum(coeffs**2),
                                regularization_loss(coeffs_csc, l0=1, l1=1, l2=1))
    else:
        num_solutions = coeffs.shape[1]
        np.testing.assert_equal((coeffs != 0).sum(axis=0), regularization_loss(coeffs_csc, l0=[1]*num_solutions))
        np.testing.assert_equal((coeffs != 0).sum(axis=0) + abs(coeffs).sum(axis=0) + (coeffs ** 2).sum(axis=0),
                                regularization_loss(coeffs_csc, l0=[1]*num_solutions, l1=1, l2=1))
        np.testing.assert_equal(np.arange(num_solutions)*((coeffs != 0).sum(axis=0))
                                + abs(coeffs).sum(axis=0)
                                + (coeffs ** 2).sum(axis=0),
                                regularization_loss(coeffs_csc, l0=range(num_solutions), l1=1, l2=1))


def test_squared_error():
    pass


def test_logistic_loss():
    pass


def test_squared_hinge_loss():
    pass


def test_score():
    pass


def test_predict(sample_FitModel):
    ones_1D = np.ones([1, 2])
    ones_2D = np.ones([2, 2])

    np.testing.assert_array_equal(sample_FitModel.predict(ones_1D), np.zeros([1, 6]))
    np.testing.assert_array_equal(sample_FitModel.predict(ones_2D), np.zeros([2, 6]))

    np.testing.assert_array_equal(sample_FitModel.predict(ones_1D, gamma=1), np.zeros([1, 1]))
    np.testing.assert_array_equal(sample_FitModel.predict(ones_2D, gamma=1), np.zeros([2, 1]))

    np.testing.assert_array_equal(sample_FitModel.predict(ones_1D, lambda_0=1), np.zeros([1, 1]))
    np.testing.assert_array_equal(sample_FitModel.predict(ones_2D, lambda_0=1), np.zeros([2, 1]))

    np.testing.assert_array_equal(sample_FitModel.predict(ones_1D, lambda_0=1, gamma=1), np.zeros([1, 1]))
    np.testing.assert_array_equal(sample_FitModel.predict(ones_2D, lambda_0=1, gamma=1), np.zeros([2, 1]))

    x_1D = np.array([[1, 2]])
    x_2D = np.array([[1, 2], [1, 2]])

    np.testing.assert_array_equal(sample_FitModel.predict(x_1D), np.arange(1, 7)[np.newaxis, :])
    np.testing.assert_array_equal(sample_FitModel.predict(x_2D), np.tile(np.arange(1, 7), (2, 1)))

    np.testing.assert_array_equal(sample_FitModel.predict(x_1D, gamma=1), np.array([[3]]))
    np.testing.assert_array_equal(sample_FitModel.predict(x_2D, gamma=1), np.array([[3], [3]]))

    np.testing.assert_array_equal(sample_FitModel.predict(x_1D, lambda_0=1), np.array([[2]]))
    np.testing.assert_array_equal(sample_FitModel.predict(x_2D, lambda_0=1), np.array([[2], [2]]))

    np.testing.assert_array_equal(sample_FitModel.predict(x_1D, lambda_0=1, gamma=1), np.array([[3]]))
    np.testing.assert_array_equal(sample_FitModel.predict(x_2D, lambda_0=1, gamma=1), np.array([[3], [3]]))
