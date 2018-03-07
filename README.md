# L0Learn
---
## Fast Best Subset Selection: Coordinate Descent and Local Combinatorial Optimization Algorithms 

### Introduction
L0Learn is a highly optimized framework for solving L0-regularized regression (and soon classification) problems. We consider sparse linear regression problems where the loss function is penalized by combinations of the L0, L1, and L2 norms. Specifically, the framework can approximate the solutions of the following three problems

<img src="https://user-images.githubusercontent.com/11324150/31854350-b33425ca-b665-11e7-8d6e-eb9da62e7560.png" width = 225>
<img src="https://user-images.githubusercontent.com/11324150/31854351-b6847b12-b665-11e7-879d-a7668f395267.png" width = 300>
<img src="https://user-images.githubusercontent.com/11324150/31854353-b816a1f8-b665-11e7-86a2-9b3c3c7bde34.png" width = 300>

The optimization is done using coordinate descent and local combinatorial optimization over a grid of the regularization parameter(s) values. We describe the details of the algorithms in our paper: Fast Best Subset Selection: Coordinate Descent and Local Combinatorial Optimization Algorithms ([PDF](https://arxiv.org/abs/1803.01454)). 

The framework is implemented in C++ along with an easy-to-use R interface. Below we provide the installation instructions for the R package.

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
To demonstrate how L0Learn works we will generate the following dummy dataset
* A 500x1000 design matrix X with iid standard normal entries
* A 1000x1 vector B with the first 10 entries set to 1 and the rest are zeros.
* A 500x1 vector e with iid standard normal entries
* Set y  = XB + e
```R
set.seed(1) # fix the seed to get a reproducible result
X = matrix(rnorm(500*1000),nrow=500,ncol=1000)
B = c(rep(1,10),rep(0,990))
e = rnorm(500)
y = X%*%B + e
```
Our objective is to use L0Learn to recover the true vector B by examining X and y only. To do this we first introduce the main data fitting function, `L0Learn.fit`, which has the following interface
```R
L0Learn.fit(X,y, Model="L0", MaxSuppSize=100, NLambda=100, NGamma=10,	GammaMax=10, GammaMin=0.0001)
```
* **X** is the design matrix and **y** is the response vector
* **Model** specifies the objective function to be optimized. The default value is "L0" which regularizes using the L0 norm only. Other options include "L0L1" and "L0L2", which combine the L0 norm with the L1 norm and L2 norm, respectively. Moreover, you can use "L0Swaps", "L0L1Swaps", and "L0L2Swaps" to get higher quality solutions.
* **MaxSuppSize** specifies the maximum support size in the solution path after which the algorithm terminates.
* **NLambda** is the number of Lambda grid points. Note: The actual values of Lambda are data-dependent and are computed automatically by the algorithm.
* **NGamma** is the number of Gamma grid points in case the Model is set to "L0L1" or "L0L2". In such cases, **GammaMax** and **GammaMin** specify the minimum and maximum values of Gamma.

For this example, we are going to fit an L0-regularized model and signal the algorithm to stop when the support size reaches 50. This can be done by executing:
```R
fit = L0Learn.fit(X,y,Model="L0",MaxSuppSize=50)
```
This will generate solutions for a sequence of Lambda values. To view the path of Lambda values along with the associated support sizes, you can execute:
```R
print(fit)
```
Now to print the learned vector B for a specific point in the path we use the function `L0Learn.coef(fit,index)` which takes the object fit as the first parameter and the index of the point in the path as the second paramter. Note that the solution at index 7 has a support of size 10. To print this solution you can execute:
```R
print(L0Learn.coef(fit,7))
```
which prints the non-zero coefficients along with the intercept term. We can also make predictions using a specific solution in the grid using the `L0Learn.predict(fit,x,index)` where x is a testing sample (vector or matrix). For example, to predict the response for the samples in the design matrix X using the solution at index 7 we can issue
```R
L0Learn.predict(fit,X,7)
```
We have demonstrated the simple case of using an L0 penalty alone. For more elaborate penalties/algorithms you can try the other models (i.e., L0Swaps, L0L1, L0L1Swaps, L0L2, and L0L2Swaps).

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
