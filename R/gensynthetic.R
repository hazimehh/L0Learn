#' @importFrom stats rnorm
#' @title Generate Synthetic Data
#'
#' @description Generates a synthetic dataset as follows: 1) Sample every element in data matrix X from N(0,1).
#' 2) Generate a vector B with the first k entries set to 1 and the rest are zeros. 3) Sample every element in the noise
#' vector e from N(0,1). 4) Set y = XB + e.
#' @param n Number of samples
#' @param p Number of features
#' @param k Number of non-zeros in true vector of coefficients
#' @param seed The seed used for randomly generating the data
#' @return A list containing the data matrix X and the response vector y.
#' @examples
#' data <- GenSynthetic(n=500,p=1000,k=10,seed=1)
#' X = data$X
#' y = data$y
#' @export
GenSynthetic <- function(n, p, k, seed)
{
    set.seed(seed) # fix the seed to get a reproducible result
    X = matrix(rnorm(n*p),nrow=n,ncol=p)
    B = c(rep(1,k),rep(0,p-k))
    e = rnorm(n)
    y = X%*%B + e
    list(X=X, y = y)
}
