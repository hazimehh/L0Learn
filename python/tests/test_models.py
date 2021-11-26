import numpy as np
import pandas as pd
import pytest
from hypothesis.strategies import floats
from scipy.sparse import csc_matrix, rand

from l0learn.models import FitModel, CVFitModel, regularization_loss, squared_error, logistic_loss, squared_hinge_loss

from pytest import fixture
from hypothesis import given
from hypothesis.extra import numpy as npst


def _sample_FitModel(loss: str = 'SquaredError', intercept: bool = True, penalty: str = 'L0L1'):
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
                    intercepts=[[-1 * i, -2 * i], [-3 * i], [-4 * i, -5 * i, -6 * i]],
                    converged=[[True, True], [False], [True, False, True]])


@fixture
def sample_FitModel():
    return _sample_FitModel()


@fixture
def sample_CVFitModel(num_folds: int = 2):
    fit_model = _sample_FitModel()
    cvMeans = np.random.random(num_folds) + 2  # Ensure positive
    cvSTDs = np.random.random(num_folds)

    settings = fit_model.settings
    settings['num_folds'] = num_folds

    return CVFitModel(settings=fit_model.settings,
                      lambda_0=fit_model.lambda_0,
                      gamma=fit_model.gamma,
                      support_size=fit_model.support_size,
                      coeffs=fit_model.coeffs,
                      intercepts=fit_model.intercepts,
                      converged=fit_model.converged,
                      cv_means=cvMeans,
                      cv_sds=cvSTDs)


def test_CVFitModel_plot(sample_CVFitModel):
    sample_CVFitModel.cv_plot(gamma=1)


def test_FitModel_plot(sample_FitModel):
    sample_FitModel.plot(gamma=1)
    sample_FitModel.plot(gamma=1, show_lines=True)


def test_FitModel_coeff(sample_FitModel):
    np.testing.assert_array_equal(sample_FitModel.coeff().toarray(),
                                  np.array([[-1, -2, -3, -4, -5, -6],
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


def test_characteristics(sample_FitModel):
    pd.testing.assert_frame_equal(sample_FitModel.characteristics(),
                                  pd.DataFrame({'l0': [10, 5, 10, 10, 5, 0],
                                                'support_size': [0, 1, 3, 4, 5, 6],
                                                'intercept': [-1, -2, -3, -4, -5, -6],
                                                'converged': [True, True, False, True, False, True],
                                                'l1': [2, 2, 1, 0, 0, 0]}))

    pd.testing.assert_frame_equal(sample_FitModel.characteristics(gamma=1),
                                  pd.DataFrame({'l0': [10],
                                                'support_size': [3],
                                                'intercept': [-3],
                                                'converged': [False],
                                                'l1': [1]}))

    pd.testing.assert_frame_equal(sample_FitModel.characteristics(lambda_0=6),
                                  pd.DataFrame({'l0': [5],
                                                'support_size': [1],
                                                'intercept': [-2],
                                                'converged': [True],
                                                'l1': [2]}))

    pd.testing.assert_frame_equal(sample_FitModel.characteristics(gamma=0, lambda_0=5),
                                  pd.DataFrame({'l0': [5],
                                                'support_size': [5],
                                                'intercept': [-5],
                                                'converged': [False],
                                                'l1': [0]}))


@given(coeffs=npst.arrays(dtype=np.float,
                          elements=floats(allow_nan=False, allow_infinity=False, max_value=1e100, min_value=-1e100),
                          shape=npst.array_shapes(min_dims=2, max_dims=2)))
def test_regularization_loss(coeffs):
    coeffs_csc = csc_matrix(coeffs)
    if coeffs.shape[1] == 1:
        np.testing.assert_equal((coeffs != 0).sum(), regularization_loss(coeffs_csc, l0=1))
        np.testing.assert_equal((coeffs != 0).sum() + sum(abs(coeffs)) + sum(coeffs ** 2),
                                regularization_loss(coeffs_csc, l0=1, l1=1, l2=1))
    else:
        num_solutions = coeffs.shape[1]
        np.testing.assert_equal((coeffs != 0).sum(axis=0), regularization_loss(coeffs_csc, l0=[1] * num_solutions))
        np.testing.assert_equal((coeffs != 0).sum(axis=0) + abs(coeffs).sum(axis=0) + (coeffs ** 2).sum(axis=0),
                                regularization_loss(coeffs_csc, l0=[1] * num_solutions, l1=1, l2=1))
        np.testing.assert_equal(np.arange(num_solutions) * ((coeffs != 0).sum(axis=0))
                                + abs(coeffs).sum(axis=0)
                                + (coeffs ** 2).sum(axis=0),
                                regularization_loss(coeffs_csc, l0=range(num_solutions), l1=1, l2=1))


def test_regularization_loss_bad_inputs():
    N = 10
    a = rand(N, N, .2, format='csc')
    l0 = np.ones(N)
    l1 = np.ones((1, N))
    l2 = 5
    with pytest.raises(ValueError):
        regularization_loss(a, l0, l1, l2)

    l0 = np.ones(N + 1)
    l1 = np.ones(N + 1)
    l2 = 5
    with pytest.raises(ValueError):
        regularization_loss(a, l0, l1, l2)


@pytest.mark.parametrize("y_y_hat_error",
                         [(np.arange(10), np.arange(10), 0,),
                          (np.arange(10).reshape(10, 1),
                           np.arange(10),
                           np.array([142.5, 102.5, 72.5, 52.5, 42.5, 42.5, 52.5, 72.5, 102.5, 142.5])),
                          ])
def test_squared_error_testing(y_y_hat_error):
    y, y_hat, error = y_y_hat_error
    np.testing.assert_equal(squared_error(y, y_hat), error)


@pytest.mark.parametrize("y_y_hat_error",
                         [(np.arange(10), np.arange(10), 0),
                          (np.arange(10).reshape(10, 1),
                           np.arange(10),
                           np.array([142.5, 102.5, 72.5, 52.5, 42.5, 42.5, 52.5, 72.5, 102.5, 142.5])),
                          ])
@given(coeffs=npst.arrays(dtype=np.float,
                          elements=floats(allow_nan=False, allow_infinity=False, max_value=100, min_value=-100),
                          shape=(10, 10)))
def test_squared_error_testing_reg(y_y_hat_error, coeffs):
    coeffs_csc = csc_matrix(coeffs)
    y, y_hat, error = y_y_hat_error
    np.testing.assert_array_almost_equal(squared_error(y, y_hat, coeffs_csc, l0=0, l1=0, l2=0), error)

    reg_error = (coeffs != 0).sum(axis=0) + abs(coeffs).sum(axis=0) + np.square(coeffs).sum(axis=0)
    np.testing.assert_array_almost_equal(squared_error(y, y_hat, coeffs_csc, l0=np.ones(10), l1=1, l2=1),
                                         error + reg_error)


@pytest.mark.parametrize("y_y_hat_error",
                         [(np.ones(10), np.ones(10), 0,),
                          (np.ones(10).reshape(10, 1), np.ones(10), np.zeros(10)),
                          (np.ones(10), 0.5 * np.ones(10), np.log(2) * 10)
                          ])
def test_logistic_loss(y_y_hat_error):
    y, y_hat, error = y_y_hat_error
    np.testing.assert_almost_equal(logistic_loss(y, y_hat), error)


@pytest.mark.parametrize("y_y_hat_error",
                         [(np.ones(10), np.ones(10), 0,),
                          (np.ones(10).reshape(10, 1), np.ones(10), np.zeros(10)),
                          (np.ones(10), 0.5 * np.ones(10), np.log(2) * 10)
                          ])
@given(coeffs=npst.arrays(dtype=np.float,
                          elements=floats(allow_nan=False, allow_infinity=False, max_value=100, min_value=-100),
                          shape=(10, 10)))
def test_logistic_loss_testing_reg(y_y_hat_error, coeffs):
    coeffs_csc = csc_matrix(coeffs)
    y, y_hat, error = y_y_hat_error
    np.testing.assert_almost_equal(logistic_loss(y, y_hat, coeffs_csc, l0=0, l1=0, l2=0), error)

    reg_error = (coeffs != 0).sum(axis=0) + abs(coeffs).sum(axis=0) + np.square(coeffs).sum(axis=0)
    np.testing.assert_array_almost_equal(logistic_loss(y, y_hat, coeffs_csc, l0=np.ones(10), l1=1, l2=1),
                                         error + reg_error)


@pytest.mark.parametrize("y_y_hat_error",
                         [(np.ones(10), np.ones(10), 0,),
                          (np.ones(10).reshape(10, 1), np.ones(10), np.zeros(10)),
                          (np.ones(10), 0.5 * np.ones(10), 0.25),
                          (np.ones(10), np.zeros(10), 1)
                          ])
def test_squared_hinge_loss(y_y_hat_error):
    y, y_hat, error = y_y_hat_error
    np.testing.assert_almost_equal(squared_hinge_loss(y, y_hat), error)


@pytest.mark.parametrize("y_y_hat_error",
                         [(np.ones(10), np.ones(10), 0,),
                          (np.ones(10).reshape(10, 1), np.ones(10), np.zeros(10)),
                          (np.ones(10), 0.5 * np.ones(10), 0.25),
                          (np.ones(10), np.zeros(10), 1)
                          ])
@given(coeffs=npst.arrays(dtype=np.float,
                          elements=floats(allow_nan=False, allow_infinity=False, max_value=100, min_value=-100),
                          shape=(10, 10)))
def test_squared_hinge_loss_reg(y_y_hat_error, coeffs):
    coeffs_csc = csc_matrix(coeffs)
    y, y_hat, error = y_y_hat_error
    np.testing.assert_almost_equal(squared_hinge_loss(y, y_hat, coeffs_csc, l0=0, l1=0, l2=0), error)

    reg_error = (coeffs != 0).sum(axis=0) + abs(coeffs).sum(axis=0) + np.square(coeffs).sum(axis=0)
    np.testing.assert_array_almost_equal(squared_hinge_loss(y, y_hat, coeffs_csc, l0=np.ones(10), l1=1, l2=1),
                                         error + reg_error)


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("loss", ["SquaredHinge", "Logistic", "SquaredError"])
def test_score(training, loss):
    fit_model = _sample_FitModel(loss=loss)

    x_training = np.identity(2)
    y_training = np.array([1, 1])

    scored = fit_model.score(x_training, y_training, lambda_0=10, gamma=2, training=training)

    if training:
        exra_args = {"coeffs": fit_model.coeff(lambda_0=10, gamma=2, include_intercept=False),
                     "l0": fit_model.characteristics(lambda_0=10, gamma=2)["l0"],
                     "l1": fit_model.characteristics(lambda_0=10, gamma=2)["l1"]}
    else:
        exra_args = {}

    print(exra_args)

    if loss == "Logistic":
        expected_value = logistic_loss(y_true=y_training,
                                       y_pred=fit_model.predict(x_training, lambda_0=10, gamma=2),
                                       **exra_args)
    elif loss == "SquaredHinge":
        expected_value = squared_hinge_loss(y_true=y_training,
                                            y_pred=fit_model.predict(x_training, lambda_0=10, gamma=2),
                                            **exra_args)

    else:
        expected_value = squared_error(y_true=y_training,
                                       y_pred=fit_model.predict(x_training, lambda_0=10, gamma=2),
                                       **exra_args)

    np.testing.assert_array_almost_equal(scored, expected_value)


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("loss", ["SquaredHinge", "Logistic", "SquaredError"])
def test_score_characteristics(training, loss, sample_FitModel):
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
