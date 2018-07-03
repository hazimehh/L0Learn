# L0Learn: Fast Best Subset Selection [![Build Status](https://travis-ci.org/hazimehh/L0Learn.svg?branch=master)](https://travis-ci.org/hazimehh/L0Learn)

### Hussein Hazimeh and Rahul Mazumder 

### This branch is for the old non-CRAN version of L0Learn. For the latest version, please [click here](https://github.com/hazimehh/L0Learn).


## Introduction
L0Learn is a highly efficient framework for solving L0-regularized regression (and soon classification) problems. It can (approximately) solve the following three problems, where the squared error loss is penalized by combinations of the L0, L1, and L2 norms:

<img src="https://github.com/hazimehh/L0Learn/blob/master/misc/regeqs.png" width = 450>

The optimization is done using coordinate descent and local combinatorial search over a grid of regularization parameter(s) values. We describe the details of the algorithms in our paper: *Fast Best Subset Selection: Coordinate Descent and Local Combinatorial Optimization Algorithms* ([arXiv link](https://arxiv.org/abs/1803.01454)). 

The toolkit is implemented in C++11 and can often run faster than popular sparse learning toolkits (see our experiments in the paper above). We also provide an easy-to-use R interface.

## R Package Installation
Most of the toolkit is built using C++11, which might not be compatible with older versions of R. Before proceeding with the installation we recommend installing the latest version of R (currently 3.4). In R, you will need first to install and load the "devtools" package:
```
install.packages("devtools")
library(devtools)
```
Now you can install and load the "L0Learn" package as follows:
```
install_github("hazimehh/L0Learn",ref="beta")
library(L0Learn)
```
If you experience problems during the installation of devtools or L0Learn, please refer to the [Troubleshooting Wiki](https://github.com/hazimehh/L0Learn/wiki/Installation-Troubleshooting).

## Usage
Please refer to the [Usage Wiki](https://github.com/hazimehh/L0Learn/wiki/Usage) for a description of the API and a demonstration. 

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
