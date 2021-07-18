#' @importFrom stats rnorm rbinom
#' @importFrom MASS mvrnorm
#' @title Generate Logistic Synthetic Data
#'
#' @description Generates a synthetic dataset as follows: 1) Generate a data matrix, 
#' X, drawn from a multivariate Gaussian distribution with mean = 0, sigma = Sigma
#' 2) Generate a vector B with k entries set to 1 and the rest are zeros.
#' 3) Every coordinate yi of the outcome vector y exists in {-1, 1}^n is sampled 
#' independently from a Bernoulli distribution with success probability: 
#' P(yi = 1|xi) = 1/(1 + exp(-s<xi, B>))
#' Source https://arxiv.org/pdf/2001.06471.pdf Section 5.1 Data Generation
#' @param n Number of samples
#' @param p Number of features
#' @param k Number of non-zeros in true vector of coefficients
#' @param seed The seed used for randomly generating the data
#' @param rho The threshold for setting values to 0.  if |X(i, j)| > rho => X(i, j) <- 0
#' @param s Signal-to-noise parameter. As s -> +Inf, the data generated becomes linearly separable. 
#' @param sigma Correlation matrix, defaults to I.
#' @param shuffle_B A boolean flag for whether or not to randomly shuffle the Beta vector, B.
#'  If FALSE, the first k entries in B are set to 1.
#' @return A list containing:
#'  the data matrix X,
#'  the response vector y,
#'  the coefficients B,
#' @examples
#' data <- L0Learn:::GenSyntheticLogistic(n=500,p=1000,k=10,seed=1)
#' X = data$X
#' y = data$y
GenSyntheticLogistic <- function(n, p, k, seed, rho=0, s=1, sigma=NULL, shuffle_B=FALSE) 
{
    if (s < 0){
        stop("s must be fall in the interval [0, +Inf)")   
    }

    X = NULL
    set.seed(seed) 
    
    if (is.null(sigma)){
        X = matrix(rnorm(n*p), n, p)
    } else {
        X = mvrnorm(n, mu=0, Sigma=sigma)
    }
    
    X[abs(X) < rho] <- 0.
    B = c(rep(1,k),rep(0,p-k))
    
    if (shuffle_B){
        B = sample(B)
    }
    
    y = rbinom(n, 1, 1/(1 + exp(-s*X%*%B)))    
    
    return(list(X=X, B=B, y=y, s=s))
}
