# L0Learn: Fast Best Subset Selection [![Build Status](https://travis-ci.org/hazimehh/L0Learn.svg?branch=master)](https://travis-ci.org/hazimehh/L0Learn) 

### Hussein Hazimeh and Rahul Mazumder 
### Massachusetts Institute of Technology

Downloads from Rstudio: [![](https://cranlogs.r-pkg.org/badges/grand-total/L0Learn)](https://cran.rstudio.com/web/packages/L0Learn/index.html)



## Introduction
L0Learn is a highly efficient framework for solving L0-regularized learning problems. It can (approximately) solve the following three problems, where the empirical loss is penalized by combinations of the L0, L1, and L2 norms:

<img src="https://github.com/hazimehh/L0Learn/blob/master/misc/eqs.png" width = 450>

We support both regression (using squared error loss) and classification (using logistic or squared hinge loss). Optimization is done using coordinate descent and local combinatorial search over a grid of regularization parameter(s) values. Many computational tricks and heuristics are used to speed up the algorithms and improve the solution quality. These heuristics include warm starts, active set convergence, correlation screening, greedy cycling order, and efficient methods for updating the residuals through exploiting sparsity and problem dimensions. Moreover, we employed a new computationally efficient method for dynamically selecting the regularization parameter λ in the path. We describe the details of the algorithms in our paper: *Fast Best Subset Selection: Coordinate Descent and Local Combinatorial Optimization Algorithms* ([arXiv link](https://arxiv.org/abs/1803.01454)). 

The toolkit is implemented in C++11 and can often run faster than popular sparse learning toolkits (see our experiments in the paper above). We also provide an easy-to-use R interface; see the section below for installation and usage of the R package.

## R Package Installation and Usage
The latest version of L0Learn (v1.2.0) can be installed from Github:
```{R}
library(devtools)
install_github("hazimehh/L0Learn",ref="v1.2")
```

The previous version (v1.1.0) can be installed from CRAN:
```{R}
install.packages("L0Learn", repos = "http://cran.rstudio.com")
```

For a tutorial, please refer to [L0Learn's Vignette](https://cran.r-project.org/web/packages/L0Learn/vignettes/L0Learn-vignette.html). For a detailed description of the API, check the [Reference Manual](https://cran.r-project.org/web/packages/L0Learn/L0Learn.pdf).

L0Learn's changelog can be accessed from [here](https://github.com/hazimehh/L0Learn/blob/master/chagelog). For users who have been using L0Learn before July 1, 2018, please check this [Wiki page](https://github.com/hazimehh/L0Learn/wiki/Note-for-previous-users) for more information on the changes introduced in the new CRAN version.

## Citing L0Learn
If you find L0Learn useful in your research, please consider citing the following paper:
```
@article{hazimeh_mazumder_2019,
   title={Fast Best Subset Selection: Coordinate Descent and Local Combinatorial Optimization Algorithms},
   author={Hazimeh, Hussein and Mazumder, Rahul},
   journal={Operations Research},
   url={https://arxiv.org/abs/1803.01454},
   keywords = {Statistics - Computation, Mathematics - Optimization and Control, Statistics - Machine Learning},
   year={2019}
}
```
