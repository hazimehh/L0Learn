# import C++ compiled code
#' @useDynLib L0Learn
#' @importFrom Rcpp evalCpp
#' @importFrom methods as
#' @importFrom graphics arrows plot
#' @import Matrix

#' @title Fit an L0-regularized model
#'
#' @description Computes the regularization path for the specified loss function and
#' penalty function (which can be a combination of the L0, L1, and L2 norms).
#' @param x The data matrix.
#' @param y The response vector.
#' @param loss The loss function to be minimized. Currently we support the choice "SquaredError".
#' @param penalty The type of regularization. This can take either one of the following choices:
#' "L0", "L0L2", and "L0L1".
#' @param algorithm The type of algorithm used to minimize the objective function. Currently "CD" and "CDPSI" are
#' are supported. "CD" is a variant of cyclic coordinate descent and runs very fast. "CDPSI" performs
#' local combinatorial search on top of CD and typically achieves higher quality solutions (at the expense
#' of increased running time).
#' @param maxSuppSize The maximum support size at which to terminate the regularization path. We recommend setting
#' this to a small fraction of min(n,p) (e.g. 0.05 * min(n,p)) as L0 regularization typically selects a small
#' portion of non-zeros.
#' @param nLambda The number of Lambda values to select (recall that Lambda is the regularization parameter
#' corresponding to the L0 norm).
#' @param nGamma The number of Gamma values to select (recall that Gamma is the regularization parameter
#' corresponding to L1 or L2, depending on the chosen penalty).
#' @param gammaMax The maximum value of Gamma when using the L0L2 penalty. For the L0L1 penalty this is
#' automatically selected.
#' @param gammaMin The minimum value of Gamma when using the L0L2 penalty. For the L0L1 penalty, the minimum
#' value of gamma in the grid is set to gammaMin * gammaMax.
#' @param partialSort If TRUE partial sorting will be used for sorting the coordinates to do greedy cycling (see our paper for
#' for details). Otherwise, full sorting is used.
#' @param maxIters The maximum number of iterations (full cycles) for CD per grid point.
#' @param tol The tolerance which decides when to terminate CD (based on the relative change in the objective).
#' @param activeSet If TRUE, performs active set updates.
#' @param activeSetNum The number of consecutive times a support should appear before declaring support stabilization.
#' @param maxSwaps The maximum number of swaps used by CDPSI for each grid point.
#' @param scaleDownFactor This parameter decides how close the selected Lambda values are. The choice should be between
#' strictly between 0 and 1 (i.e., 0 and 1 are not allowed). Larger values lead to closer lambdas and typically to smaller
#' gaps between the support sizes. For details, see our paper - Section 5 on Adaptive Selection of Tuning Parameters).
#' @param screenSize The number of coordinates to cycle over when performing initial correlation screening.
#' @param autoLambda If FALSE, the user specifier a grid of Lambda values through the lambdaGrid parameter. Otherwise,
#' if TRUE, the values of Lambda are automatically selected based on the data.
#' @param lambdaGrid A vector of Lambda values to use in computing the regularization path. This is ignored unless autoLambda = FALSE.
#' @return An S3 object of type "L0Learn" describing the regularization path. The object has the following members.
#' \item{a0}{a0 is a list of intercept sequences. The ith element of the list (i.e., a0[[i]]) is the sequence of intercepts corresponding to the ith gamma value (i.e., gamma[i]).}
#' \item{beta}{This is a list of coefficient matrices. The ith element of the list is a p x \code{length(lambda)} matrix which
#' corresponds to the ith gamma value. The jth column in the coefficient matrix is the vector of coefficients for the jth lambda value.}
#' \item{lambda}{This is the list of lambda sequences used in fitting the model. The ith element of lambda (i.e., lambda[[i]]) is a sequence
#' of Lambda values corresponding to the ith gamma value.}
#' \item{gamma}{This is the sequence of gamma values used in fitting the model.}
#' \item{suppSize}{This is a list of support size sequences. The ith element of the list is a sequences of support sizes (i.e., number of non-zero coefficients)
#' corresponding to the ith gamma value.}
#' \item{converged}{This is a list of sequences. The ith element of the list is a sequence corresponding to the ith value of gamma, where the jth element in
#' in the sequence indicates whether the algorithm has converged at the jth value of lambda.}
#'
#' @examples
#' # Generate synthetic data for this example
#' data <- GenSynthetic(n=500,p=1000,k=10,seed=1)
#' X = data$X
#' y = data$y
#'
#' # Fit an L0 Model with a maximum of 50 non-zeros using coordinate descent
#' fit1 <- L0Learn.fit(X, y, penalty="L0", maxSuppSize=50)
#' print(fit1)
#' # Extract the coefficients at lambda = 0.0325142
#' coef(fit1, lambda=0.0325142)
#' # Apply the fitted model on X to predict the response
#' predict(fit1, newx = X, lambda=0.0325142)
#'
#' # Fit an L0 Model with a maximum of 50 non-zeros using coordinate descent and local search
#' fit2 <- L0Learn.fit(X, y, penalty="L0", algorithm="CDPSI", maxSuppSize=50)
#' print(fit2)
#'
#' # Fit an L0L2 Model with 10 values of Gamma ranging from 0.0001 to 10, using coordinate descent
#' fit3 <- L0Learn.fit(X, y, penalty="L0L2", maxSuppSize=50, nGamma=10, gammaMin=0.0001, gammaMax = 10)
#' print(fit3)
#' # Extract the coefficients at lambda = 0.0361829 and gamma = 0.0001
#' coef(fit3, lambda=0.0361829, gamma=0.0001)
#' # Apply the fitted model on X to predict the response
#' predict(fit3, newx = X, lambda=0.0361829, gamma=0.0001)
#'
#' @export
L0Learn.fit <- function(x,y, loss="SquaredError", penalty="L0", algorithm="CD", maxSuppSize=100, nLambda=100, nGamma=10,
						gammaMax=10, gammaMin=0.0001, partialSort = TRUE, maxIters=200,
						tol=1e-6, activeSet=TRUE, activeSetNum=3, maxSwaps=100, scaleDownFactor=0.8, screenSize=1000, autoLambda = TRUE, lambdaGrid = list(0))
{
	# The C++ function uses LambdaU = 1 for user-specified grid. In R, we use autoLambda0 = 0 for user-specified grid (thus the negation when passing the paramter to the function below)
	M <- .Call('_L0Learn_L0LearnFit', PACKAGE = 'L0Learn', x, y, loss, penalty, algorithm, maxSuppSize, nLambda, nGamma, gammaMax, gammaMin, partialSort, maxIters, tol, activeSet, activeSetNum, maxSwaps, scaleDownFactor, screenSize, !autoLambda, lambdaGrid)

	G <- list(beta = M$beta, lambda=lapply(M$lambda,signif, digits=6), a0=M$a0, converged = M$Converged, suppSize= M$SuppSize, gamma=M$gamma, penalty=penalty, loss=loss)

	if (is.null(colnames(x))){
			varnames <- 1:dim(x)[2]
	} else {
			varnames <- colnames(x)
	}
	G$varnames <- varnames

	class(G) <- "L0Learn"
	G$n <- dim(x)[1]
	G$p <- dim(x)[2]
	G
}
