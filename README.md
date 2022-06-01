# L0Learn: Fast Best Subset Selection 
[![CRAN](https://www.r-pkg.org/badges/version/L0Learn)](https://cran.r-project.org/package=L0Learn) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/7fd68c533fd1493288e7986df3cc6f6d)](https://www.codacy.com/gh/hazimehh/L0Learn/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=hazimehh/L0Learn&amp;utm_campaign=Badge_Grade) [![codecov](https://codecov.io/gh/hazimehh/L0Learn/branch/master/graph/badge.svg?token=QYDNA400OI)](https://codecov.io/gh/hazimehh/L0Learn)

### Hussein Hazimeh, Rahul Mazumder, and Tim Nonet
### Massachusetts Institute of Technology

Downloads from Rstudio: [![](https://cranlogs.r-pkg.org/badges/grand-total/L0Learn)](https://cran.rstudio.com/web/packages/L0Learn/index.html)

## Introduction
L0Learn is a highly efficient framework for solving L0-regularized learning problems. It can (approximately) solve the following three problems, where the empirical loss is penalized by combinations of the L0, L1, and L2 norms:

<img src="https://github.com/hazimehh/L0Learn/blob/master/misc/eqs.png" width = 450>

We support both regression (using squared error loss) and classification (using logistic or squared hinge loss). Optimization is done using coordinate descent and local combinatorial search over a grid of regularization parameter(s) values. Several  computational tricks and heuristics are used to speed up the algorithms and improve the solution quality. These heuristics include warm starts, active set convergence, correlation screening, greedy cycling order, and efficient methods for updating the residuals through exploiting sparsity and problem dimensions. Moreover, we employed a new computationally efficient method for dynamically selecting the regularization parameter λ in the path. We describe the details of the algorithms in our paper: *Fast Best Subset Selection: Coordinate Descent and Local Combinatorial Optimization Algorithms* ([link](https://pubsonline.informs.org/doi/10.1287/opre.2019.1919)). 

The toolkit is implemented in C++11 and can often run faster than popular sparse learning toolkits (see our experiments in the paper above). We also provide an easy-to-use R and Pythoninterface; see the section below for installation and usage of the package.

**NEW: Version 2 (03/2021) adds support for sparse matrices and box constraints on the coefficients.**

## R Package Installation
The latest version (v2.0.3) can be installed from CRAN as follows:
```{R}
install.packages("L0Learn", repos = "http://cran.rstudio.com")
```
Alternatively, L0Learn can also be installed from Github as follows:
```{R}
library(devtools)
install_github("hazimehh/L0Learn")
```
L0Learn's changelog can be accessed from [here](https://github.com/hazimehh/L0Learn/blob/master/ChangeLog).

## Python Package Installation
The latest version #### can be installed from PIP as follows:
```{python}
pip install l0learn
```
Alternative, L0learn can also be built from source as follows:
```{bash}
git clone https://github.com/hazimehh/L0Learn.git
cd L0Learn
pip install .
```


##  Usage
### R Usage
For a tutorial, please refer to [L0Learn's Vignette](https://cran.r-project.org/web/packages/L0Learn/vignettes/L0Learn-vignette.html). For a detailed description of the API, check the [Reference Manual](https://cran.r-project.org/web/packages/L0Learn/L0Learn.pdf).
### Python Usage
For a tutorial and detailed documentation, please refer to [l0learn's documetnation](https://tnonet.github.io/L0Learn/). 

## FAQ
#### Which penalty to use?
Pure L0 regularization can overfit when the signal strength in the data is relatively low. Adding L2 regularization can alleviate this problem and lead to competitive models (see the experiments in our paper). Thus, in practice, **we strongly  recommend using the L0L2 penalty**. Ideally, the parameter gamma (for L2 regularization) should be tuned over a sufficiently large interval, and this can be performed using L0Learn's built-in [R cross-validation method)](https://cran.r-project.org/web/packages/L0Learn/vignettes/L0Learn-vignette.html#cross-validation) or [Python cross-validation method](MISSING).

#### Which algorithm to use?
By default, L0Learn uses a coordinate descent-based algorithm, which achieves competitive run times compared to popular sparse learning toolkits. This can work well for many applications. We also offer a local search algorithm which is guarantteed to return higher quality solutions, at the expense of an increase in the run time. We recommend using the local search algorithm if your problem has highly correlated features or the number of samples is much smaller than the number of features---see the [R local search section of the Vignette](https://cran.r-project.org/web/packages/L0Learn/vignettes/L0Learn-vignette.html#higher-quality_solutions_using_local_search) or [Python local search section of the Vignette](https://cran.r-project.org/web/packages/L0Learn/vignettes/L0Learn-vignette.html#higher-quality_solutions_using_local_search)for how to use this algorithm.


#### How to certify optimality?
While for many challenging statistical instances L0Learn leads to optimal solutions, it cannot provide certificates of optimality. Such certificates can be provided via Integer Programming. Our toolkit [L0BnB](https://github.com/alisaab/l0bnb) is a scalable integer programming framework for L0-regularized regression, which can provide such certificates and potentially improve upon the solutions of L0Learn (if they are sub-optimal). We recommend using L0Learn first to obtain a candidtate solution (or a pool of solutions) and then checking optimality using L0BnB.


## Citing L0Learn
If you find L0Learn useful in your research, please consider citing the following papers.

**Paper 1 (Regression):**
```
@article{doi:10.1287/opre.2019.1919,
author = {Hazimeh, Hussein and Mazumder, Rahul},
title = {Fast Best Subset Selection: Coordinate Descent and Local Combinatorial Optimization Algorithms},
journal = {Operations Research},
volume = {68},
number = {5},
pages = {1517-1537},
year = {2020},
doi = {10.1287/opre.2019.1919},
URL = {https://doi.org/10.1287/opre.2019.1919},
eprint = {https://doi.org/10.1287/opre.2019.1919}
}
```

**Paper 2 (Classification):**
```
@article{JMLR:v22:19-1049,
  author  = {Antoine Dedieu and Hussein Hazimeh and Rahul Mazumder},
  title   = {Learning Sparse Classifiers: Continuous and Mixed Integer Optimization Perspectives},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {135},
  pages   = {1-47},
  url     = {http://jmlr.org/papers/v22/19-1049.html}
}
```

**Paper 3 (Toolkit):**
```
@article{hazimeh2022l0learn,
      title={L0Learn: A Scalable Package for Sparse Learning using L0 Regularization}, 
      author={Hussein Hazimeh and Rahul Mazumder and Tim Nonet},
      year={2022},
      eprint={2202.04820},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
