import warnings
from dataclasses import dataclass, field
from functools import wraps
from typing import List, Dict, Any, Optional, Tuple, Union, Sequence, Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import csc_matrix, vstack, hstack, find
from scipy.stats import multivariate_normal, binom


def regularization_loss(coeffs: csc_matrix,
                        l0: Union[float, Sequence[float]] = 0,
                        l1: Union[float, Sequence[float]] = 0,
                        l2: Union[float, Sequence[float]] = 0, ) -> Union[float, np.ndarray]:
    """
    Calculates the regularization loss for a path of (or individual) solution(s).

    Parameters
    ----------
    coeffs
    l0:
    l1
    l2

    Returns
    -------
    loss : float or np.ndarray of shape (K,) where K is the number of solutions in coeffs
        Let the shape of coeffs be (P, K) and l0, l1, and l2 by either a float or a sequence of length K
            Then loss will be of shape: (K, ) with the following layout:
                loss[i] = l0[i]*||coeffs[:, i||_0 + l1[i]*||coeffs[:, i||_1 + l2[i]*||coeffs[:, i||_2

    Notes
    -----

    """
    if coeffs.ndim != 2:
        raise NotImplementedError("expected coeffs to be a 2D array")

    l0 = np.asarray(l0)
    l1 = np.asarray(l1)
    l2 = np.asarray(l2)

    _, num_solutions = coeffs.shape

    infer_penalties = l0 + l1 + l2

    if infer_penalties.ndim == 0:
        num_penalties = num_solutions
    elif infer_penalties.ndim == 1:
        num_penalties = len(infer_penalties)
    else:
        raise ValueError(f"expected penalties l0, l1, and l2 mutually broadcast to a float or a 1D array,"
                         f" but got {infer_penalties.ndim}D array.")

    if num_penalties != num_solutions:
        raise ValueError(f"expected number of penalties to be equal to the number of solutions if multiple penalties "
                         f"are provided, but got {num_penalties} penalties and {num_solutions} solutions.")

    l0 = np.broadcast_to(l0, [num_penalties])
    l1 = np.broadcast_to(l1, [num_penalties])
    l2 = np.broadcast_to(l2, [num_penalties])

    if num_solutions == 1:
        _, _, values = find(coeffs)

        return float(l0 * len(values) + l1 * sum(abs(values)) + sum((np.sqrt(l2) * values) ** 2))
    else:  # TODO: Implement this regularization loss better than a for loop over num_solutions!
        _, num_solutions = coeffs.shape
        loss = np.zeros(num_solutions)

        for solution_index in range(num_solutions):
            loss[solution_index] = regularization_loss(coeffs[:, solution_index],
                                                       l0=l0[solution_index],
                                                       l1=l1[solution_index],
                                                       l2=l2[solution_index])

    return loss


# def broadcast_with_regularization_loss(
#         metric_func: Callable[[Any], Tuple[np.ndarray, np.ndarray]]) -> Callable[[Any], np.ndarray]:
#     @wraps(metric_func)
#     def reshape_metric_loss(*args, **kwargs) -> np.ndarray:
#         metric_loss, reg_loss = metric_func(*args, **kwargs)
#
#         if reg_loss.ndim == 0:
#             return metric_loss + reg_loss
#         elif reg_loss.ndim == 2 and metric_loss.ndim == 2:
#             # reg_loss of shape (l, k)
#             # metric_loss of shape (m, k)
#             return metric_loss[:, np.newaxis, :] + reg_loss  # result of shape (m, l, k)
#         elif reg_loss.ndim == 1 and metric_loss.ndim == 1:
#             # reg_loss of shape (l,)
#             # squared_residuals of shape (m, )
#             return metric_loss[:, np.newaxis] + reg_loss
#         else:
#             raise ValueError("This should not happen")
#
#     return reshape_metric_loss


def squared_error(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  coeffs: Optional[csc_matrix] = None,
                  l0: Union[float, Sequence[float]] = 0,
                  l1: Union[float, Sequence[float]] = 0,
                  l2: Union[float, Sequence[float]] = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates Squared Error loss of solution with optional regularization

    Parameters
    ----------
    y_true : np.ndarray of shape (m, )
    y_pred : np.ndarray of shape (m, ) or (m, k)
    coeffs : np.ndarray of shape (p, k), optional
    l0 : float or sequence of floats of shape (l)
    l1 : float or sequence of floats of shape (l)
    l2 : float or sequence of floats of shape (l)

    Returns
    -------
    squared_error = (m, k)

    """
    reg_loss = 0
    if coeffs is not None:
        reg_loss = regularization_loss(coeffs=coeffs, l0=l0, l1=l1, l2=l2)

    if y_pred.ndim == 2:
        y_true = y_true[:, np.newaxis]

    squared_residuals = 0.5 * np.square(y_true - y_pred)

    return squared_residuals + reg_loss


def logistic_loss(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  coeffs: Optional[csc_matrix] = None,
                  l0: float = 0,
                  l1: float = 0,
                  l2: float = 0, ) -> Union[float, np.ndarray]:
    # TODO: Check this formula. If there is an error here, there might be an error in the C++ code for Logistic.

    reg_loss = 0
    if coeffs is not None:
        reg_loss = regularization_loss(coeffs=coeffs, l0=l0, l1=l1, l2=l2)

    # C++ Src Code:
    # ExpyXB = arma::exp(this->y % (*(this->X) * this->B + this->b0));
    # return arma::sum(arma::log(1 + 1 / expyXB)) + regularization

    exp_y_XB = np.exp(y_true * np.log(y_pred))
    log_loss = np.log(1 + 1 / exp_y_XB)

    return log_loss + reg_loss


def squared_hinge_loss(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       coeffs: Optional[csc_matrix] = None,
                       l0: float = 0,
                       l1: float = 0,
                       l2: float = 0, ) -> Union[float, np.ndarray]:
    # TODO: Check this formula. If there is an error here, there might be an error in the C++ code for Logistic.

    reg_loss = 0
    if coeffs is not None:
        reg_loss = regularization_loss(coeffs=coeffs, l0=l0, l1=l1, l2=l2)

    # C++ Src Code:
    # onemyxb = 1 - this->y % (*(this->X) * this->B + this->b0);
    # arma::uvec indices = arma::find(onemyxb > 0);
    # return arma::sum(onemyxb.elem(indices) % onemyxb.elem(indices)) + this->lambda0 * n_nonzero(
    #     B) + this->lambda1 * arma::norm(B, 1) + this->lambda2 * l2norm * l2norm;

    square_one_minus_y_XB = np.square(np.max(1 - y_true * y_pred, 0))
    return square_one_minus_y_XB + reg_loss


@dataclass(frozen=True, repr=False, eq=False)
class FitModel:
    settings: Dict[str, Any]
    lambda_0: List[List[float]] = field(repr=False)
    gamma: List[float] = field(repr=False)
    support_size: List[List[int]] = field(repr=False)
    coeffs: List[csc_matrix] = field(repr=False)
    intercepts: List[List[float]] = field(repr=False)
    converged: List[List[bool]] = field(repr=False)

    def characteristics_as_pandas_table(self, new_data: Optional[Tuple[List[float],
                                                                       List[List[float]],
                                                                       List[List[int]],
                                                                       List[List[float]],
                                                                       List[List[bool]]]] = None) -> pd.DataFrame:
        if new_data is not None:
            gamma, lambda_0, support_size, intercepts, converged = new_data
        else:
            gamma, lambda_0, support_size, intercepts, converged = (self.gamma,
                                                                    self.lambda_0,
                                                                    self.support_size,
                                                                    self.intercepts,
                                                                    self.converged)
        tables = []
        for gamma, lambda_sequence, support_sequence, intercept_sequence, converged_sequence in zip(gamma,
                                                                                                    lambda_0,
                                                                                                    support_size,
                                                                                                    intercepts,
                                                                                                    converged):

            data = {'l0': lambda_sequence,
                    'support_size': support_sequence,
                    'intercept': intercept_sequence,
                    'converged': converged_sequence}

            if len(self.settings['penalty']) > 2:
                data[self.settings['penalty'][2:].lower()] = [gamma] * len(lambda_sequence)

            tables.append(pd.DataFrame(data))

        return pd.concat(tables).reset_index(drop=True)

    def _repr_html_(self):
        return self.characteristics_as_pandas_table()._repr_html_()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.settings})"

    def coeff(self,
              lambda_0: Optional[float] = None,
              gamma: Optional[float] = None,
              include_intercept: bool = True) -> csc_matrix:
        """ Extracts the coefficient according to lambda_0 and gamma

        If both lambda_0 and gamma are not supplied, then a matrix of coefficients for all the solutions in the
        regularization path is returned.

        If lambda_0 is supplied but gamma is not, the smallest value of gamma is used.

        If gamma is supplied but not lambda_0, then a matrix of coefficients for all the solutions in the
        regularization path for gamma is returned.

        Parameters
        ----------
        lambda_0 : float, optional
        gamma : float, optional
        include_intercept : bool, default True

        Returns
        -------
        coeff: csc_matrix

        """
        if gamma is None:
            if lambda_0 is None:
                # Return all solutions
                solutions = hstack(self.coeffs)
                intercepts = [[intercept for intercept_list in self.intercepts for intercept in intercept_list]]
            else:
                # Return solution with closest lambda in first gamma
                return self.coeff(gamma=self.gamma[0],
                                  lambda_0=lambda_0,
                                  include_intercept=include_intercept)
        else:
            gamma_index = int(np.argmin(np.abs(np.asarray(self.gamma) - gamma)))
            if lambda_0 is None:
                # Return all solutions for specific gamma
                solutions = self.coeffs[gamma_index]
                intercepts = [self.intercepts[gamma_index]]
            else:
                # Return solution with closest lambda in gamma_index
                lambda_index = int(np.argmin(np.abs(np.asarray(self.lambda_0[gamma_index]) - lambda_0)))
                solutions = self.coeffs[gamma_index][:, lambda_index]
                intercepts = [self.intercepts[gamma_index][lambda_index]]

        if include_intercept:
            solutions = csc_matrix(vstack([intercepts[0], solutions]))

        return solutions

    def characteristics(self,
                        lambda_0: Optional[float] = None,
                        gamma: Optional[float] = None) -> pd.DataFrame:

        if gamma is None:
            if lambda_0 is None:
                # Return all solutions
                intercepts = self.intercepts
                lambda_0 = self.lambda_0
                gamma = self.gamma
                support_size = self.support_size
                converged = self.converged
            else:
                # Return solution with closest lambda in first gamma
                return self.characteristics(gamma=self.gamma[0],
                                            lambda_0=lambda_0)
        else:
            gamma_index = int(np.argmin(np.abs(np.asarray(self.gamma) - gamma)))
            if lambda_0 is None:
                # Return all solutions for specific gamma
                intercepts = [self.intercepts[gamma_index]]
                lambda_0 = [self.lambda_0[gamma_index]]
                gamma = [self.gamma[gamma_index]]
                support_size = [self.support_size[gamma_index]]
                converged = [self.converged[gamma_index]]
            else:
                # Return solution with closest lambda in gamma_index
                lambda_index = int(np.argmin(np.abs(np.asarray(self.lambda_0[gamma_index]) - lambda_0)))
                intercepts = [[self.intercepts[gamma_index][lambda_index]]]
                lambda_0 = [[self.lambda_0[gamma_index][lambda_index]]]
                gamma = [self.gamma[gamma_index]]
                support_size = [[self.support_size[gamma_index][lambda_index]]]
                converged = [[self.converged[gamma_index][lambda_index]]]

        return self.characteristics_as_pandas_table(new_data=(gamma, lambda_0, support_size, intercepts, converged))

    def plot(self, gamma: float = 0, show_lines: bool = False, **kwargs):
        """ Plots the regularization path for a given gamma.

        Parameters
        ----------
        gamma : float
            The value of gamma at which to plot
        show_lines : bool
            If True, the lines connecting the points in the plot are shown.
        kwargs : dict of str to any
            Key Word arguments passed to matplotlib.pyplot

        Notes
        -----
        If the solutions with the same support size exist in a regularization path, the first support size is plotted.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
        """

        gamma_to_plot = self.coeff(gamma=gamma, include_intercept=False)
        p = gamma_to_plot.shape[1]

        # Find what coeffs are seen in regularization path. Skip later solutions
        seen_supports = set()
        seen_coeffs = set()
        for col in range(p):
            rows, cols, values = find(gamma_to_plot[:, col])
            support_size = len(rows)
            if support_size in seen_supports or not support_size:
                warnings.warn(f"Duplicate solution seen at support size {support_size}. Plotting only first solution")
                continue
            seen_coeffs.update(rows)

        # For each coefficient seen in regularization path, record value of each coefficient over
        seen_coeffs = list(sorted(seen_coeffs))
        coef_value_over_path = gamma_to_plot[seen_coeffs, :].toarray()
        path_support_size = (gamma_to_plot != 0).sum(axis=0).T

        f, ax = plt.subplots(**kwargs)

        if show_lines:
            linestyle = kwargs.pop("linestyle", "_.")
        else:
            linestyle = kwargs.pop("linestyle", "None")

        marker = kwargs.pop("marker", "o")

        for i, coeff in enumerate(seen_coeffs):
            ax.plot(path_support_size, coef_value_over_path[i, :], label=coeff, linestyle=linestyle, marker=marker)

        plt.ylabel("Coefficient Value")
        plt.xlabel("Support Size")

        return ax

    def score(self,
              x: np.ndarray,
              y: np.ndarray,
              lambda_0: Optional[float] = None,
              gamma: Optional[float] = None,
              training: bool = False,
              include_characteristics: bool = False) -> Union[pd.DataFrame, np.ndarray]:
        """

        Parameters
        ----------
        x
        y
        lambda_0
        gamma

        Returns
        -------

        """
        predictions = self.predict(x=x, lambda_0=lambda_0, gamma=gamma)
        characteristics = self.characteristics(lambda_0=lambda_0, gamma=gamma)

        if training:
            coeffs = self.coeff(lambda_0=lambda_0, gamma=gamma)
            l0 = characteristics.get('l0', 0)
            l1 = characteristics.get('l1', 0)
            l2 = characteristics.get('l2', 0)
        else:
            coeffs = None
            l0 = 0
            l1 = 0
            l2 = 0

        if self.settings['loss'] == "SquaredError":
            score = squared_error(y_true=y, y_pred=predictions, coeffs=coeffs, l0=l0, l1=l1, l2=l2, )
        elif self.settings['loss'] == "Logistic":
            score = logistic_loss(y_true=y, y_pred=predictions, coeffs=coeffs, l0=l0, l1=l1, l2=l2, )
        else:
            score = squared_hinge_loss(y_true=y, y_pred=predictions, coeffs=coeffs, l0=l0, l1=l1, l2=l2, )

        if include_characteristics:
            characteristics[self.settings['loss']] = score
            return characteristics
        else:
            return score

    def predict(self,
                x: np.ndarray,
                lambda_0: Optional[float] = None,
                gamma: Optional[float] = None) -> np.ndarray:
        """ Predicts the response for a given sample.

        Parameters
        ----------
        x: array-like
            An array of feature observations on which predictions are made.
            X should have shape (N, P). Intercept terms will be added by the predict function is specified during
             original training.
        lambda_0 : float, optional
            Which lambda_0 value to use for predictions

            See FitModel.coeff for details on lambda_0 specifications
        gamma : float, optional
            Which gamma value to use for predictions

            See FitModel.coeff for details on gamma specifications
        Returns
        -------
        predictions : np.ndarray
            Predictions of the response vector given x.

            For logistic regression (loss specified as "Logistic"), values are non-threshold

            When multiple coeffs are specified due to the settings of lambda_0 and gamma the prediction array is still
            as expected with:

            predictions[i, j] (row i, column j) refers to predicted response for observation i using coefficients from
            solution j.
        """
        coeffs = self.coeff(lambda_0=lambda_0,
                            gamma=gamma,
                            include_intercept=self.settings['intercept'])

        n = x.shape[0]
        if self.settings['intercept']:
            x = np.hstack([np.ones((n, 1)), x])

        activations = x @ coeffs

        if self.settings['loss'] == "Logistic":
            return 1 / (1 + np.exp(-activations))
        else:
            return activations


@dataclass(frozen=True, repr=False, eq=False)
class CVFitModel(FitModel):
    cv_means: List[np.ndarray] = field(repr=False)
    cv_sds: List[np.ndarray] = field(repr=False)

    def cv_plot(self, gamma: float = 0, **kwargs):
        """
        Plot the cross-validation errors for a given gamma.

        Parameters
        ----------
        gamma : float
            The value of gamma at which to plot
        kwargs
            Key Word arguments passed to matplotlib.pyplot
        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
        """
        gamma_index = int(np.argmin(np.abs(np.asarray(self.gamma) - gamma)))
        f, ax = plt.subplots(**kwargs)

        plt.errorbar(x=self.support_size[gamma_index],
                     y=self.cv_means[gamma_index],
                     yerr=self.cv_sds[gamma_index], )

        plt.ylabel("Cross-Validation Error")
        plt.xlabel("Support Size")

        return ax


def gen_synthetic(n: int,
                  p: int,
                  k: int,
                  seed: Optional[int] = None,
                  rho: float = 0,
                  b0: float = 0,
                  snr: float = 1
                  ) -> Dict[str, Union[float, np.ndarray]]:
    """
    Generates a synthetic dataset as follows:
        1) Sample every element in data matrix X from N(0,1).
        2) Generate a vector B with the first k entries set to 1 and the rest are zeros.
        3) Sample every element in the noise vector e from N(0,A) where A is selected so y, X, B have snr as specified.
        4) Set y = XB + b0 + e.

    Parameters
    ----------
    n : int
        Number of samples
    p : int
        Number of features
    k : int
        Number of non-zeros in true vector of coefficients
    seed : int, optional
        The seed used for randomly generating the data
        If None, numbers will be random
    rho : float, default 0.
        The threshold for setting X values to 0. if |X[i, j]| < rho => X[i, j] <- 0
    b0 : float, default 0
        The intercept value to translate y by.
    snr : float, default 1
        Desired Signal-to-Noise ratio. This sets the magnitude of the error term 'e'.

        Note: SNR is defined as  SNR = Var(XB)/Var(e)

    Returns
    -------
    Data: Dict
        A dict containing:
            the data matrix X,
            the response vector y,
            the coefficients B,
            the error vector e,
            the intercept term b0.

    Examples
    --------
    >>>data = gen_synthetic(n=500,p=1000,k=10,seed=1)
    """

    # TODO: Don't re-seed generator. https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html
    np.random.seed(seed)

    X = np.random.normal(size=(n, p))
    X[abs(X) < rho] = 0
    B = np.zeros(p)
    B[0:k] = 1

    if snr == float("Inf"):
        sd_e = 0
    else:
        sd_e = np.sqrt(np.var(X @ B) / snr)

    e = np.random.normal(n, scale=sd_e)
    y = X @ B + e + b0
    return {"X": X, "y": y, "B": B, "e": e, "b0": b0}


def cor_matrix(p: int, base_cor: float):
    if not (0 < base_cor < 1):
        raise ValueError(f"Expected base_cor to be a float between 0 and 1 exclusively, but got {base_cor}")

    cor_mat = base_cor * np.ones((p, p))
    pow_mat = abs(np.arange(p) - np.arange(p).reshape(-1, 1))

    return np.power(cor_mat, pow_mat)


def gen_synthetic_high_corr(n: int,
                            p: int,
                            k: int,
                            seed: Optional[int] = None,
                            rho: float = 0,
                            b0: float = 0,
                            snr: float = 1,
                            mu: float = 0,
                            base_cor: float = 0.8,
                            ) -> Dict[str, Union[float, np.ndarray]]:
    """
    Generates a synthetic dataset as follows:
        1) Generate a correlation matrix, SIG,  where item [i, j] = base_core^|i-j|.
        2) Draw from a Multivariate Normal Distribution using (mu and SIG) to generate X.
        3) Generate a vector B with every ~p/k entry set to 1 and the rest are zeros.
        4) Sample every element in the noise vector e from N(0,A) where A is selected so y, X, B have snr as specified.
        5) Set y = XB + b0 + e.

    Parameters
    ----------
    n : int
        Number of samples
    p : int
        Number of features
    k : int
        Number of non-zeros in true vector of coefficients
    seed : int
        The seed used for randomly generating the data
    rho : float
        The threshold for setting values to 0.  if |X(i, j)| < rho => X(i, j) <- 0
    b0 : float
        intercept value to scale y by.
    snr : float
        desired Signal-to-Noise ratio. This sets the magnitude of the error term 'e'.
        SNR is defined as  SNR = Var(XB)/Var(e)
    mu : float
        The mean for drawing from the Multivariate Normal Distribution. A scalar of vector of length p.
    base_cor : float
        The base correlation, A in [i, j] = A^|i-j|.

    Returns
    -------
    data : dict
        A dict containing:
            the data matrix X,
            the response vector y,
            the coefficients B,
            the error vector e,
            the intercept term b0.

    Examples
    --------
    >>>gen_synthetic_high_corr(n=500,p=1000,k=10,seed=1)

    """
    # TODO: Don't re-seed generator. https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html
    np.random.seed(seed)

    cor = cor_matrix(p, base_cor)
    X = multivariate_normal(n, mu, cor)

    X[abs(X) < rho] = 0.

    B = np.zeros(p)
    for i in np.round(np.linspace(start=0, stop=p, num=k)).astype(int):
        B[i] = 1

    if snr == float("inf"):
        sd_e = 0
    else:
        sd_e = np.sqrt(np.var(X @ B) / snr)

    e = np.random.normal(n, scale=sd_e)
    y = X @ B + e + b0

    return {"X": X, "y": y, "B": B, "e": e, "b0": b0}


def gen_synthetic_logistic(n: int,
                           p: int,
                           k: int,
                           seed: Optional[int] = None,
                           rho: float = 0,
                           b0: float = 0,
                           s: float = 1,
                           mu: Optional[float] = None,
                           base_cor: Optional[float] = None,
                           ) -> Dict[str, Union[float, np.ndarray]]:
    """
    Generates a synthetic dataset as follows:
        1) Generate a data matrix, X, drawn from either N(0, 1) (see gen_synthetic) or a multivariate_normal(mu, sigma)
            (See gen_synthetic_high_corr)
            gen_synthetic_logistic delegates these caluclations to the respective functions.
        2) Generate a vector B with k entries set to 1 and the rest are zeros.
        3) Every coordinate yi of the outcome vector y exists in {0, 1}^n is sampled independently from a Bernoulli
            distribution with success probability:
                P(yi = 1|xi) = 1/(1 + exp(-s<xi, B>))
            Source https://arxiv.org/pdf/2001.06471.pdf Section 5.1 Data Generation

    Parameters
    ----------
    n : int
        Number of samples
    p : int
        Number of features
    k : int
        Number of non-zeros in true vector of coefficients
    seed : int
        The seed used for randomly generating the data
    rho : float
        The threshold for setting values to 0.  if |X(i, j)| > rho => X(i, j) <- 0
    b0 : float
        The intercept value to scale the log odds of y by.
        As b0 -> +inf, y will contain more 1s
        As b0 -> -inf, y will contain more 0s
    s : float
        Signal-to-noise parameter. As s -> +Inf, the data generated becomes linearly separable.
    mu : float, optional
        The mean for drawing from the Multivariate Normal Distribution. A scalar of vector of length p.
        If mu and base_cor are not specified, will be drawn from N(0, 1) using gen_synthetic.
        If mu and base_cor are specified will be drawn from multivariate_normal(mu, sigma) see gen_synthetic_high_corr
    base_cor
        The base correlation, A in [i, j] = A^|i-j|.
        If mu and base_cor are not specified, will be drawn from N(0, 1)
        If mu and base_cor are specified will be drawn from multivariate_normal(mu, sigma) see gen_synthetic_high_corr
    Returns
    -------
    data : dict
         A dict containing:
            the data matrix X
            the response vector y,
            the coefficients B,
            the intercept term b0.

    """
    if mu is None and base_cor is None:
        data = gen_synthetic(n=n, p=p, k=k, seed=seed, rho=rho, b0=b0, snr=float("Inf"))
    else:
        data = gen_synthetic_high_corr(n=n, p=p, k=k, seed=seed, rho=rho, b0=b0, snr=float("Inf"), mu=mu,
                                       base_cor=base_cor)

    y = binom.rvs(1, 1 / (1 + np.exp(-s * data["X"] @ data['B'])), size=n)

    data['y'] = y
    del data['e']

    return data
