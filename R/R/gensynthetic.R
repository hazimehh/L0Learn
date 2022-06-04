#' @importFrom stats rnorm var
#' @title Generate Synthetic Data
#'
#' @description Generates a synthetic dataset as follows:
#' 1) Sample every element in data matrix X from N(0,1).
#' 2) Generate a vector B with the first k entries set to 1 and the rest are
#'    zeros.
#' 3) Sample every element in the noise vector e from N(0,A) where A is
#'    selected so y, X, B have snr as specified.
#' 4) Set y = XB + b0 + e.
#' @param n Number of samples
#' @param p Number of features
#' @param k Number of non-zeros in true vector of coefficients
#' @param seed The seed used for randomly generating the data
#' @param rho The threshold for setting values to 0.
#' If |X(i, j)| > rho => X(i, j) <- 0
#' @param b0 intercept value to translate y by.
#' @param snr desired Signal-to-Noise ratio.
#' This sets the magnitude of the error term 'e'.
#' SNR is defined as  SNR = Var(XB)/Var(e)
#' @return A list containing:
#'  the data matrix X,
#'  the response vector y,
#'  the coefficients B,
#'  the error vector e,
#'  the intercept term b0.
#' @examples
#' data <- GenSynthetic(n=500,p=1000,k=10,seed=1)
#' X = data$X
#' y = data$y
#' @export
GenSynthetic <- function(n, p, k, seed, rho=0, b0=0, snr=1) {
    set.seed(seed) # fix the seed to get a reproducible result
    X <- matrix(rnorm(n * p), nrow = n, ncol = p)
    X[abs(X) < rho] <- 0.
    B <- c(rep(1, k), rep(0, p - k))
    sd_e <- NULL

    if (snr == +Inf) {
        sd_e <- 0
    } else {
        sd_e <- sqrt(var(X %*% B) / snr)
    }

    e <- rnorm(n, sd = sd_e)
    y <- X %*% B + e + b0
    list(X = X, y = y, B = B, e = e, b0 = b0)
}
