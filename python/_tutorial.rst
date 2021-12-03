l0learn starter guide
=====================

Introduction
============
`l0learn` is a fast toolkit for L0-regularized learning. L0 regularization selects the best subset of features and can outperform commonly used feature selection methods (e.g., L1 and MCP) under many sparse learning regimes. The toolkit can (approximately) solve the following three problems.

.. math::
    \\min_{\beta_0, \beta} \sum_{i=1}^{n} \ell(y_i, \beta_0+ \langle x_i, \beta \rangle) + \lambda ||\beta||_0 \quad \quad (L0)

.. math::
    \\min_{\beta_0, \beta} \sum_{i=1}^{n} \ell(y_i, \beta_0+ \langle x_i, \beta \rangle) + \lambda ||\beta||_0 + \gamma||\beta||_1 \quad (L0L1)

.. math::
    \\min_{\beta_0, \beta} \sum_{i=1}^{n} \ell(y_i, \beta_0+ \langle x_i, \beta \rangle) + \lambda ||\beta||_0 + \gamma||\beta||_2^2  \quad (L0L2)

where :math:`\ell` is the loss function, :math:`\beta_0` is the intercept, :math:`\beta` is the vector of coefficients, and :math:`||\beta||_0` denotes the L0 norm of :math:`\beta`, i.e., the number of non-zeros in :math:`\beta`. We support both regression and classification using either one of the following loss functions:

* Squared error loss
* Logistic loss (logistic regression)
* Squared hinge loss (smoothed version of SVM).

The parameter :math:`\lambda` controls the strength of the L0 regularization (larger :math:`\lambda` leads to less non-zeros). The parameter :math:`\gamma` controls the strength of the shrinkage component (which is the L1 norm in case of L0L1 or squared L2 norm in case of L0L2); adding a shrinkage term to L0 can be very effective in avoiding overfitting and typically leads to better predictive models. The fitting is done over a grid of :math:`\lambda` and :math:`\gamma` values to generate a regularization path.

The algorithms provided in `l0learn` are based on cyclic coordinate descent and local combinatorial search. Many computational tricks and heuristics are used to speed up the algorithms and improve the solution quality. These heuristics include warm starts, active set convergence, correlation screening, greedy cycling order, and efficient methods for updating the residuals through exploiting sparsity and problem dimensions. Moreover, we employed a new computationally efficient method for dynamically selecting the regularization parameter :math:`\lambda` in the path. For more details on the algorithms used, please refer to our paper [Fast Best Subset Selection: Coordinate Descent and Local Combinatorial Optimization Algorithms](https://pubsonline.informs.org/doi/10.1287/opre.2019.1919).

The toolkit is implemented in C++ along with an easy-to-use Python interface. In this tutorial, we provide a tutorial on using the Python interface. Particularly, we will demonstrate how use L0Learn's main functions for fitting models, cross-validation, and visualization.

Installation
============
L0Learn can be installed directly from pip by running the following command in terminal:

.. code-block:: console

    pip install l0learn

If you face installation issues, please refer to the [Installation Troubleshooting Wiki](https://github.com/hazimehh/L0Learn/wiki/Installation-Troubleshooting). If the issue is not resolved, you can submit an issue on [L0Learn's Github Repo](https://github.com/hazimehh/L0Learn).

Tutorial
========
To demonstrate how `l0learn` works, we will first generate a synthetic dataset and then proceed to fitting L0-regularized models. The synthetic dataset (y,X) will be generated from a sparse linear model as follows:

* X is a 500x1000 design matrix with iid standard normal entries
* B is a 1000x1 vector with the first 10 entries set to 1 and the rest are zeros.
* e is a 500x1 vector with iid standard normal entries
* y is a 500x1 response vector such that y  = XB + e

This dataset can be generated in Python as follows:

.. code-block:: python

    import numpy as np
    np.random.seed(4) # fix the seed to get a reproducible result
    n, p, k = 500, 1000, 100
    X = np.random.normal(size=(n, p))
    B = np.zeros(p)
    B[:k] = 1
    e = np.random.normal(n)
    y = X@B + e

More expressive and complete functions for generating datasets can be found are available in :py:mod:`l0learn.models`. The available functions are:

* :py:meth:`l0learn.models.gen_synthetic`
* :py:meth:`l0learn.models.gen_synthetic_high_corr`
* :py:meth:`l0learn.models.gen_synthetic_logistic`

We will use L0Learn to estimate B from the data (y,X). First we load L0Learn:

.. code-block:: python

    from l0learn import fit


We will start by fitting a simple L0 model and then proceed to the case of L0L2 and L0L1.

Fitting L0 Regression Models
============================

To fit a path of solutions for the L0-regularized model with at most 20 non-zeros using coordinate descent (CD), we use the :py:meth:`l0learn.models.fit` function as follows:

.. code-block:: python

    fit = fit(X, y, penalty="L0", max_support_size=20)


This will generate solutions for a sequence of :math:`\lambda` values (chosen automatically by the algorithm). To view the sequence of :math:`\lambda` along with the associated support sizes (i.e., the number of non-zeros), we use the built in rich display from `ipython Rich Display <https://ipython.readthedocs.io/en/stable/config/integrating.html+>`_ in iPython Notebooks. When running this tutorial in a more standard python environment, use the function :py:meth:`l0learn.models.FitModel.characteristics` to display the sequence of solutions.

.. code-block:: python

    fit # will render as an pandas DataFrame.


To extract the estimated B for particular values of :math:`\lambda` and :math:`\gamma`, we use the function :py:meth:`l0learn.models.FitModel.coeff`. For example, the solution at :math:`\lambda = 0.0325142` (which corresponds to a support size of 10) can be extracted using:

.. code-block:: python

    fit.coeff(lambda_0=0.0325)

The output is a sparse matrix of type `scipy.sparse.csc_matrix <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html>`. Depending on the `include_intercept` parameter of :py:meth:`l0learn.models.FitModel.coeff`, The first element in the vector is the intercept and the rest are the B coefficients. Aside from the intercept, the only non-zeros in the above solution are coordinates 0, 1, 2, 3, ..., 9, which are the non-zero coordinates in the true support (used to generated the data). Thus, this solution successfully recovers the true support. Note that on some BLAS implementations, the `lambda` value we used above (i.e., `0.0325142`) might be slightly different due to the limitations of numerical precision. Moreover, all the solutions in the regularization path can be extracted at once by calling :code:`fit.coeff()`.

The sequence of :math:`\lambda` generated by `l0learn` is stored in the object :code:`fit`. Specifically, :code:`fit.lambda_0` is a list, where each element of the list is a sequence of :math:`\lambda` values corresponding to a single value of :math:`\gamma`. When using an L0 penalty , which has only one value of :math:`\gamma` (i.e., 0), we can access the sequence of :math:`\lambda` values using :code:`fit.lambda_0[0]`. Thus, :math:`\lambda=0.0325142` we used previously can be accessed using :code:`fit.lambda_0[1][6]` (since it is the 7th value in the output of :code:`fit.characteristics()`). So the previous solution can also be extracted using :code:`fit.coeff(lambda_0=fit.lambda_0[1][6], gamma=0)`.

We can make predictions using a specific solution in the grid using the function :code:`fit.predict(newx, lambda, gamma)` where :code:`newx` is a testing sample (vector or matrix). For example, to predict the response for the samples in the data matrix X using the solution with :math:`\lambda=0.0325142`, we call the prediction function as follows:

.. code-block:: python

    fit.predict(x=X, lambda_0=0.0325142, gamma=0)

We can also visualize the regularization path by plotting the coefficients of the estimated B versus the support size (i.e., the number of non-zeros) using the :py:meth:`l0learn.models.FitModel.plot` method as follows:

.. code-block:: python

    fit.plot(fit, gamma=0)

The legend of the plot presents the variables in the order they entered the regularization path. For example, variable 7 is the first variable to enter the path, and variable 6 is the second to enter. Thus, roughly speaking, we can view the first $k$ variables in the legend as the best subset of size $k$. To show the lines connecting the points in the plot, we can set the parameter :code:`show_lines=True` in the `plot` function, i.e., call :code:`fit.plot(fit, gamma=0, show_lines=True)`. Moreover, we note that the plot function returns a :code:`matplotlib.axes._subplots.AxesSubplot` object, which can be further customized using the :code:`matplotlib` package. In addition, both the :py:meth:`l0learn.models.FitModel.plot` and :py:meth:`l0learn.models.CVFitModel.cv_plot` accept :code:`**kwargs` parameter to allow for customization of the plotting behavior.

Fitting L0L2 and L0L1 Regression Models
=======================================
We have demonstrated the simple case of using an L0 penalty. We can also fit more elaborate models that combine L0 regularization with shrinkage-inducing penalties like the L1 norm or squared L2 norm. Adding shrinkage helps in avoiding overfitting and typically improves the predictive performance of the models. Next, we will discuss how to fit a model using the L0L2 penalty for a two-dimensional grid of :math:`\lambda` and :math:`\gamma` values. Recall that by default, `l0learn` automatically selects the :math:`\lambda` sequence, so we only need to specify the :math:`\gamma` sequence. Suppose we want to fit an L0L2 model with a maximum of 20 non-zeros and a sequence of 5 :math:`\gamma` values ranging between 0.0001 and 10. We can do so by calling :py:meth:`l0learn.fit` with :code:`penalty="L0L2"`, :code:`num_gamma=5`, :code:`gamma_min=0.0001`, and :code:`gamma_max=10` as follows:

.. code-block:: python

    fit = fit(X, y, penalty="L0L2", num_gamma = 5, gamma_min = 0.0001, gamma_max = 10, max_support_size=20)

`l0learn` will generate a grid of 5 :math:`\gamma` values equi-spaced on the logarithmic scale between 0.0001 and 10. Similar to the case for L0, we can display a summary of the regularization path using the :code:`fit.characteristics()` function as follows:

.. code-block:: python

    fit  # Using ipython Rich Display
    # fit.characteristics()  # For non Rich Display


The sequence of :math:`\gamma` values can be accessed using :code:`fit.gamma`. To extract a solution we use the :py:meth:`l0learn.models.FitModel.coeff` method. For example, extracting the solution at `:math:`\lambda=0.0011539` and :math:`\gamma=10` can be done using

.. code-block:: python

    fit.coeff(lambda_0=0.0011539, gamma=10)  # Using ipython Rich Display


Similarly, we can predict the response at this pair of :math:`\lambda` and :math:`\gamma` for the matrix X using

.. code-block:: python

    fit.predict(x=X, lambda_0=0.0011539, gamma=10)

The regularization path can also be plot at a specific :math:`\gamma` using :code:`fit.plot(gamma)`. Finally, we note that fitting an L0L1 model can be done by just changing the `penalty` to "L0L1" in the above (in this case `gamma_max` will be ignored since it is automatically selected by the toolkit; see the reference manual for more details.)

Higher-quality Solutions using Local Search
===========================================
By default, `l0learn` uses coordinate descent (CD) to fit models. Since the objective function is non-convex, the choice of the optimization algorithm can have a significant effect on the solution quality (different algorithms can lead to solutions with very different objective values). A more elaborate algorithm based on combinatorial search can be used by setting the parameter `algorithm="CDPSI"` in the call to :py:meth:`l0learn.fit`. `CDPSI` typically leads to higher-quality solutions compared to CD, especially when the features are highly correlated. `CDPSI` is slower than `CD`, however, for typical applications it terminates in the order of seconds.

Cross-validation
================

We will demonstrate how to use K-fold cross-validation (CV) to select the optimal values of the tuning parameters :math:`\lambda` and math:`\gamma`. To perform CV, we use the :py:meth:`l0learn.cvfit` function, which takes the same parameters as :code:`l0learn.fit`, in addition to the number of folds using the :code:`num_folds` parameter and a seed value using the :code:`seed` parameter (this is used when randomly shuffling the data before performing CV).

For example, to perform 5-fold CV using the `L0L2` penalty (over a range of 5 `gamma` values between 0.0001 and 0.1) with a maximum of 50 non-zeros, we run:

.. code-block:: python

    cv_fit_result = cvfit(X, y, num_folds=5, seed=1, penalty="L0L2", num_gamma=5, gamma_min=0.0001, gamma_max=0.1, max_support_size=50)

Note that the object :py:class:`l0learn.models.CVFitModel` subclasses :py:class:`l0learn.models.FitModel` and thus has the same methods and underlinying structure which is output of running :py:meth:`l0learn.cvfit` on (y,X). The cross-validation errors can be accessed using the `cv_means` attribute of `cvfit`: `cvfit.cv_means` is a list where the ith element, :code:`cvfit.cv_means[i]`, stores the cross-validation errors for the ith value of gamma (:code:`cvfit.gamma[i]`). To find the minimum cross-validation error for every `gamma`, we apply the :code:`np.argmin` function for every element in the list :code:`cvfit.cv_means`, as follows:

.. code-block:: python

    gamma_mins = [(np.argmin(cv_mean), np.min(cv_mean)) for cv_mean in cv_fit_result.cv_means]
    gamma_mins

The above output indicates that the 3rd value of gamma achieves the lowest CV error (`=0.9899542`). We can plot the CV errors against the support size for the 4th value of gamma, i.e., `gamma = cvfit$fit$gamma[4]`, using:


```{r, fig.height = 4.7, fig.width = 7, out.width="90%", dpi=300}
plot(cvfit, gamma=cvfit$fit$gamma[4])
```

The above plot is produced using the `ggplot2` package and can be further customized by the user. To extract the optimal $\lambda$ (i.e., the one with minimum CV error) in this plot, we execute the following:
```{r}
optimalGammaIndex = 4 # index of the optimal gamma identified previously
optimalLambdaIndex = which.min(cvfit$cvMeans[[optimalGammaIndex]])
optimalLambda = cvfit$fit$lambda[[optimalGammaIndex]][optimalLambdaIndex]
optimalLambda
```
To print the solution corresponding to the optimal gamma/lambda pair:
```{r output.lines=15}
coef(cvfit, lambda=optimalLambda, gamma=cvfit$fit$gamma[4])
```
The optimal solution (above) selected by cross-validation correctly recovers the support of the true vector of coefficients used to generated the model.

## Fitting Classification Models
All the commands and plots we have seen in the case of regression extend to classification. We currently support logistic regression (using the parameter `loss = "Logistic"`) and a smoothed version of SVM (using the parameter `loss="SquaredHinge"`). To give some examples, we first generate a synthetic classification dataset (similar to the one we generated in the case of regression):
```{r}
set.seed(1) # fix the seed to get a reproducible result
X = matrix(rnorm(500*1000),nrow=500,ncol=1000)
B = c(rep(1,10),rep(0,990))
e = rnorm(500)
y = sign(X%*%B + e)
```
An L0-regularized logistic regression model can be fit by specificying `loss = "Logistic"` as follows:
```{r output.lines=15}
fit = L0Learn.fit(X,y,loss="Logistic")
print(fit)
```
The output above indicates that $\gamma=10^{-7}$---by default we use a small ridge regularization (with $\gamma=10^{-7}$) to ensure the existence of a solution. To extract the coefficients of the solution with $\lambda = 8.69435$:
```{r output.lines=15}
coef(fit, lambda=8.69435, gamma=1e-7)
```
The above indicates that the 10 non-zeros in the estimated model match those we used in generating the data (i.e, L0 regularization correctly recovered the true support). We can also make predictions at the latter $\lambda$ using:
```{r output.lines=15}
predict(fit, newx=X, lambda=8.69435, gamma=1e-7)
```
Each row in the above is the probability that the corresponding sample belongs to class $1$. Other models (i.e., L0L2 and L0L1) can be similarly fit by specifying `loss = "Logistic"`.

Finally, we note that L0Learn also supports a smoothed version of SVM by using squared hinge loss (`loss = "SquaredHinge"`). The only difference from logistic regression is that the `predict` function returns $\beta_0 + \langle x, \beta \rangle$ (where $x$ is the testing sample), instead of returning probabilities. The latter predictions can be assigned to the appropriate classes by using a thresholding function (e.g., the sign function).


## Advanced Options

### Sparse Matrix Support
Starting in version 2.0.0, L0Learn supports sparse matrices of type dgCMatrix. If your sparse matrix uses a different storage format, please convert it to dgCMatrix before using it in L0Learn. L0Learn keeps the matrix sparse internally and thus is highly efficient if the matrix is sufficiently sparse. The API for sparse matrices is the same as that of dense matrices, so all the demonstrations in this vignette also apply for sparse matrices. For example, we can fit an L0-regularized model on a sparse matrix as follows:
```{r}

# As an example, we generate a random, sparse matrix with
# 500 samples, 1000 features, and 10% nonzero entries.
X_sparse <- Matrix::rsparsematrix(nrow=500, ncol=1000, density=0.1, rand.x = rnorm)
# Below we generate the response using the same linear model as before,
# but with the sparse data matrix X_sparse.
y_sparseX <- as.vector(X_sparse%*%B + e)

# Call L0Learn.
fit_sparse <- L0Learn.fit(X_sparse, y_sparseX, penalty="L0")

# Note: In the setup above, X_sparse is of type dgCMatrix.
# If your sparse matrix is of a different type, convert it
# to dgCMatrix before calling L0Learn, e.g., using: X_sparse <- as(X_sparse, "dgCMatrix").
```

### Selection on Subset of Variables
In certain applications, it is desirable to always include some of the variables in the model and perform variable selection on others. `L0Learn` supports this option through the `excludeFirstK` parameter. Specifically, setting `excludeFirstK = K` (where K is a non-negative integer) instructs `L0Learn` to exclude the first K variables in the data matrix `X` from the L0-norm penalty (those K variables will still be penalized using the L2 or L1 norm penalties.). For example, below we fit an `L0` model and exclude the first 3 variables from selection by setting `excludeFirstK = 3`:
```{r}
fit <- L0Learn.fit(X, y, penalty="L0", maxSuppSize=10, excludeFirstK=3)
```
Plotting the regularization path:
```{r, fig.height = 4.7, fig.width = 7, out.width="90%", dpi=300}
plot(fit, gamma=0)
```

We can see in the plot above that first 3 variables are included in all the solutions of the path.

### Coefficient Bounds
Starting in version 2.0.0, L0Learn supports bounds for CD algorithms for all losses and penalties. (We plan to support bound constraints for the CDPSI algorithm in the future). By default, L0Learn does not apply bounds, i.e., it assumes $-\infty <= \beta_i <= \infty$ for all i. Users can supply the same bounds for all coefficients by setting the parameters `lows` and `highs` to scalar values (these should satisfy: `lows <= 0`, `lows != highs`, and `highs >= 0`). To use different bounds for the coefficients, `lows` and `highs` can be both set to vectors of length `p` (where the i-th entry corresponds to the bound on coefficient i).

All of the following examples are valid.
```{r}
L0Learn.fit(X, y, penalty="L0", lows=-0.5)
L0Learn.fit(X, y, penalty="L0", highs=0.5)
L0Learn.fit(X, y, penalty="L0", lows=-0.5, highs=0.5)

low_vector <- c(rep(-0.1, 500), rep(-0.125, 500))
fit <- L0Learn.fit(X, y, penalty="L0", lows=low_vector, highs=0.25)
```

We can see the coefficients are subject to the bounds.
```{r}
print(max(fit$beta[[1]]))
print(min(fit$beta[[1]][1:500, ]))
print(min(fit$beta[[1]][501:1000, ]))
```

### User-specified Lambda Grids
By default, `L0Learn` selects the sequence of lambda values in an efficient manner to avoid wasted computation (since close $\lambda$ values can typically lead to the same solution). Advanced users of the toolkit can change this default behavior and supply their own sequence of $\lambda$ values. This can be done supplying the $\lambda$ values through the parameter `lambdaGrid`. L0Learn versions before 2.0.0 would also require setting the `autoLambda` parameter to `FALSE`. This parameter remains in version 2.0.0 for backwards compatibility, but is no longer needed or used.

Specifically, the value assigned to `lambdaGrid` should be a list of lists of decreasing positive values (doubles). The length of `lambdaGrid` (the number of lists stored) specifies the number of gamma parameters that will fill between `gammaMin`, and `gammaMax`. In the case of L0 penalty, `lambdaGrid` must be a list of length 1. In case of L0L2/L0L1 `lambdaGrid` can have any number of sub-lists stored. The length of `lambdaGrid` (the number of lists stored) specifies the number of gamma parameters that will fill between `gammaMin`, and `gammaMax`. The ith element in `LambdaGrid` should be a **decreasing** sequence of positive lambda values which are used by the algorithm for the ith value of gamma. For example, to fit an L0 model with the sequence of user-specified lambda values: 1, 1e-1, 1e-2, 1e-3, 1e-4, we run the following:
```{r}
userLambda <- list()
userLambda[[1]] <- c(1, 1e-1, 1e-2, 1e-3, 1e-4)
fit <- L0Learn.fit(X, y, penalty="L0", lambdaGrid=userLambda, maxSuppSize=1000)
```
To verify the results we print the fit object:
```{r}
print(fit)
```
Note that the $\lambda$ values above are the desired values. For L0L2 and L0L1 penalties, the same can be done where the `lambdaGrid` parameter.
```{r}
userLambda <- list()
userLambda[[1]] <- c(1, 1e-1, 1e-2, 1e-3, 1e-4)
userLambda[[2]] <- c(10, 2, 1, 0.01, 0.002, 0.001, 1e-5)
userLambda[[3]] <- c(1e-4, 1e-5)
# userLambda[[i]] must be a vector of positive decreasing reals.
fit <- L0Learn.fit(X, y, penalty="L0L2", lambdaGrid=userLambda, maxSuppSize=1000)
```

```{r}
print(fit)
```

# References
Hussein Hazimeh and Rahul Mazumder. [Fast Best Subset Selection: Coordinate Descent and Local Combinatorial Optimization Algorithms](https://pubsonline.informs.org/doi/10.1287/opre.2019.1919). Operations Research (2020).

Antoine Dedieu, Hussein Hazimeh, and Rahul Mazumder. [Learning Sparse Classifiers: Continuous and Mixed Integer Optimization Perspectives](https://arxiv.org/abs/2001.06471). JMLR (to appear).
