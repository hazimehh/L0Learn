# L0Learn: Fast Algorithms for L0-regularized Learning
## Hussein Hazimeh and Rahul Mazumder

### Introduction
L0Learn is a highly optimized framework for solving L0-regularized regression (and soon classification) problems. We consider sparse linear regression problems where the loss function is penalized by combinations of the L0, L1, and L2 norms. Specifically, the framework solves the following three problems

<img src="https://user-images.githubusercontent.com/11324150/31854350-b33425ca-b665-11e7-8d6e-eb9da62e7560.png" width = 225>
<img src="https://user-images.githubusercontent.com/11324150/31854351-b6847b12-b665-11e7-879d-a7668f395267.png" width = 300>
<img src="https://user-images.githubusercontent.com/11324150/31854353-b816a1f8-b665-11e7-86a2-9b3c3c7bde34.png" width = 300>

over a grid of the tuning parameters values. 

The framework is implemented in C++ along with an easy-to-use R interface. Below are the installation instructions.

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
If you experience problems during the installation of devtools or L0Learn, please refer to the troubleshooting wiki.

### Usage
To demonstrate how L0Learn works we will generate the following dummy dataset
* A 500x1000 design matrix X with iid standard normal entries
* A 1000x1 vector B with the first 10 entries set to 1 and the rest are zeros.
* A 500x1 vector e with iid standard normal entries
* Set y  = XB + e
```R
X = matrix(rnorm(500*1000),nrow=500,ncol=1000)
B = c(rep(1,10),rep(0,990))
e = rnorm(500)
y = X%*%B + e
```
Our objective is to use L0Learn to recover the true vector B from examining X and y only. To do this we first fit an L0Learn model as follows:
```R
fit = L0Learn.fit(X,y,Model="L0")
```
This will generate solutions for a sequence of lambda values. To view the path of lambda values along with the associated support sizes, you can execute:
```R
print(fit)
```
Now to print the learned vector B for a specific point in the path we use the function L0Learn.coef(fit,index) which takes the object fit as the first parameter and the index of the point in the path as the second paramter. For example, to print the solution with index 7 we can issue
```R
print(L0Learn.coef(fit,7))
```
We can also make predictions using a specific solution in the grid using the L0Learn.predict(fit,x,index) where x is a testing sample (vector or matrix). For example, to predict the response for the samples in X using the solution at index 7 we can issue
```R
L0Learn.predict(fit,X,7)
```
We have demonstrated the simple case of using an L0 penalty alone. For more elaborate penalties/algorithms you can try any of the following methods when calling the L0Learn.fit(X,y,method) function: 
* L0Swaps
* L0L1
* L0L1Swaps
* L0L2
* L0L2Swaps
