# L0Learn: Fast Best Subset Selection

### Introduction
L0Learn is a highly optimized framework for solving L0-regularized regression and classification problems. It can solve problems where the empirical risk is penalized by combinations of the L0, L1, and L2 norms; specifically, it approximates the solutions of the following three problems

<img src="https://github.com/hazimehh/L0Learn/blob/NewInterface/misc/l0problems.png" width = 400>

L0Learn currently supports the following loss functions: Sqaured Error Loss, Logistic Loss, and Squared Hinge Loss. The optimization is done using coordinate descent and local combinatorial optimization algorithms over a grid of  regularization parameter(s) values. We describe the details of the algorithms in our paper: Fast Best Subset Selection: Coordinate Descent and Local Combinatorial Optimization Algorithms ([arXiv link](https://arxiv.org/abs/1803.01454)). 

The toolkit is implemented in C++ along with an easy-to-use R interface. Below we provide the installation instructions for the R package.

### R Package Installation
Most of the toolkit is built using C++11, which might not be compatible with older versions of R. Before proceeding with the installation we recommend installing the latest version of R (currently 3.4). In R, you will need first to install and load the "devtools" package:
```
install.packages("devtools")
library(devtools)
```
Now you can install and load the "L0Learn" package as follows:
```
install_github("hazimehh/L0Learn")
library(L0Learn)
```
If you experience problems during the installation of devtools or L0Learn, please refer to the [troubleshooting wiki](https://github.com/hazimehh/L0Learn/wiki/Installation-Troubleshooting).

### Usage
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
