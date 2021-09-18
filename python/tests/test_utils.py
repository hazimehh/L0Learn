import numpy as np
from scipy.sparse import csc_matrix

from l0learn.utils import FitModel

from pytest import fixture


@fixture
def sample_FitModel():
    gamma = [2, 1, 0]
    lambda_0 = [[10, 5], [10], [10, 5, 0]]
    intercepts = [[-1, -2], [-3], [-4, -5, -6]]
    beta1 = csc_matrix([[0, 0],
                        [1, 2]])
    beta2 = csc_matrix([[0],
                        [3]])
    beta3 = csc_matrix([[0, 0, 0],
                        [4, 5, 6]])

    return FitModel(lambda_0=lambda_0,
                    gamma=gamma,
                    support_size=[[0]],
                    coeffs=[beta1, beta2, beta3],
                    intercepts=intercepts,
                    converged=[[True]])


def test_FitModel_coeff(sample_FitModel):
    np.testing.assert_array_equal(sample_FitModel.coeff().toarray(), np.array([[-1, -2, -3, -4, -5, -6],
                                                                               [0, 0, 0, 0, 0, 0],
                                                                               [1, 2, 3, 4, 5, 6]]))
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
