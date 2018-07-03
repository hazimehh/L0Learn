# L0Learn: Fast Best Subset Selection [![Build Status](https://travis-ci.org/hazimehh/L0Learn.svg?branch=master)](https://travis-ci.org/hazimehh/L0Learn)

### Hussein Hazimeh and Rahul Mazumder 


## Introduction
L0Learn is a highly efficient framework for solving L0-regularized regression (and soon classification) problems. It can (approximately) solve the following three problems, where the squared error loss is penalized by combinations of the L0, L1, and L2 norms:

<img src="https://github.com/hazimehh/L0Learn/blob/master/misc/regeqs.png" width = 450>

The optimization is done using coordinate descent and local combinatorial search over a grid of regularization parameter(s) values. We describe the details of the algorithms in our paper: *Fast Best Subset Selection: Coordinate Descent and Local Combinatorial Optimization Algorithms* ([arXiv link](https://arxiv.org/abs/1803.01454)). 

The toolkit is implemented in C++11 and can often run faster than popular sparse learning toolkits (see our experiments in the paper above). We also provide an easy-to-use R interface.
## R Package Installation and Usage
L0Learn 1.0.2 is now available on CRAN. For installation instructions and a tutorial, please refer to [L0Learn's Vignette](http://www.mit.edu/~hazimeh/L0Learn-vignette.html). For a detailed description of the API, check the [Reference Manual](https://cran.r-project.org/web/packages/L0Learn/L0Learn.pdf).

Note for users who have been using L0Learn before July 2018: The new CRAN version introduces many new major features, which unfortunately required doing some changes to the API. The old non-CRAN version (i.e., the one installed directly using devtools from Github) is now archived and its API will not be maintained anymore. As a courtesy, you can still access the old version and its documentation at this [link](https://github.com/hazimehh/L0Learn/tree/Beta). The API of the CRAN version is now stable, and we will ensure backward compatibility in the future versions of L0Learn.

## Citing L0Learn
If you find L0Learn useful in your research, please consider citing the following paper:
```
@ARTICLE{2018arXiv180301454H,
   author = {{Hazimeh}, H. and {Mazumder}, R.},
   title = "{Fast Best Subset Selection: Coordinate Descent and Local Combinatorial Optimization Algorithms}",
   journal = {ArXiv e-prints},
   archivePrefix = "arXiv",
   eprint = {1803.01454},
   primaryClass = "stat.CO",
   keywords = {Statistics - Computation, Mathematics - Optimization and Control, Statistics - Machine Learning},
   year = 2018,
   month = mar,
   adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180301454H},
}
```
