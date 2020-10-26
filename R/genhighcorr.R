#' @importFrom stats rnorm 
#' @importFrom MASS mvrnorm
#' @importFrom Rcpp cppFunction
#' @title Generate Expoententiall Correlated Synthetic Data
#'
#' @description Generates a synthetic dataset as follows: 1) Generate a correlation matrix, SIG,  where item [i, j] = A^|i-j|.
#' 2) Draw from a Multivariate Normal Distribution using (mu and SIG) to generate X. 3) Generate a vector B with every ~p/k entry set to 1 and the rest are zeros.
#' 4) Sample every element in the noise vector e from N(0,1). 4) Set y = XB + b0 + e.
#' @param n Number of samples
#' @param p Number of features
#' @param k Number of non-zeros in true vector of coefficients
#' @param seed The seed used for randomly generating the data
#' @param rho The threshold for setting values to 0.  if |X(i, j)| > rho => X(i, j) <- 0
#' @param b0 intercept value to scale y by.
#' @param noise_ratio The multiplier of noise to apply when calculating e. e[i] = noise_ratio*N(0, 1).
#' @param mu The mean for drawing from the Multivariate Normal Distribution. A scalar of vector of length p.
#' @param base_cor The base correlation, A in [i, j] = A^|i-j|.
#' @return A list containing:
#'  the data matrix X,
#'  the response vector y,
#'  the coefficients B,
#'  the error vector e,
#'  the intercept term b0.
#' @examples
#' data <- GenSyntheticHighCorr(n=500,p=1000,k=10,seed=1)
#' X = data$X
#' y = data$y
#' @export
GenSyntheticHighCorr <- function(n, p, k, seed, rho=0, b0=0, noise_ratio=1, mu=0, base_cor=.9)
{
    set.seed(seed) # fix the seed to get a reproducible result
    cor <- cor_matrix(p, base_cor)
    if (length(mu) == 1){
        mu = rep(mu, p)
    } 
    
    X <- mvrnorm(n, mu, forceSymmetric(cor))
    
    X[abs(X) < rho] <- 0.
    
    B_indices = seq(from=1, to=p, by=as.integer(p/k))
    
    B = rep(0, p)
    
    for (i in B_indices){
        B[i] = 1
    }
    
    e = noise_ratio*rnorm(n)
    y = X%*%B + e + b0
    list(X=X, y = y, B=B, e=e, b0=b0)
}

# This R code is quite slow.
# cor = Matrix(NA, nrow=p, ncol=p)
# for (i in 1:p){
#     for (j in 1:p){
#         cor[i, j] = .9^abs(i - j)
#     }
# }

library("Rcpp")
# This C++ code is much much faster
cppFunction('NumericMatrix cor_matrix_(const int p, const double base_cor) {
    NumericMatrix cor(p, p);
    for (int i = 0; i < p; i++){
        for (int j = 0; j < p; j++){
            cor(i, j) = pow(base_cor, std::abs(i - j));
        }
    }
  return cor;
}')

#' @description Generates a exponential correlation matrix: 
#' @export
cor_matrix <- function(p, base_cor=.9){
  cor_matrix_(p, base_cor)
}

