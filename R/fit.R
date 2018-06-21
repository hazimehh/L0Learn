# import C++ compiled code
#' @useDynLib L0Learn
#' @importFrom Rcpp evalCpp
#' @importFrom methods as
#' @importFrom graphics arrows plot
#' @import Matrix

#' @title Fit an L0-regularized model
#'
#' @description Computes the regularization path for the specified loss function and
#' choice of regularization (which can be a combination of the L0, L1, and L2 (squared) norms).
#' @param x The data matrix.
#' @param y The response vector.
#' @param loss The loss function to be minimized. The currently supported choice is "SquaredError".
#' @param penalty The type of regularization. This can take either one of the following choices:
#' "L0", "L0L2", and "L0L1".
#' @param algorithm The type of algorithm used to minimize the objective. Currently "CD" and "CDPSI" are
#' are supported. "CD" is a variant of cyclic coordinate descent and can run very fast. "CDPSI" performs
#' local combinatorial search on top of CD and thus can achieve higher quality solutions (at the expense
#' of increased running time).
#' @param maxSuppSize The maximum support size to reach in the grid before termination. We recommend setting
#' this to a small fraction of min(n,p) (e.g. 0.05 * min(n,p)) as L0 regularization typically selects a small
#' portion of non-zeros.
#' @param nLambda The number of Lambda values to select (recall that Lambda is the regularization parameter
#' corresponding to the L0 norm).
#' @param nGamma The number of Gamma values to select (recall that Gamma is the regularization parameter
#' corresponding to L1 or L2, depending on the chosen penalty).
#' @param gammaMax The maximum value of Gamma when using the L0L2 penalty. For the L0L1 penalty this is
#' automatically selected by the toolkit.
#' @param gammaMin The minimum value of Gamma when using the L0L2 penalty. For the L0L1 penalty, gammaMin
#' specifies the fraction of gammaMax at which the grid ends.
#' @param partialSort If TRUE partial sorting will be used for sorting the coordinates (see our paper for
#' for details). Otherwise, full sorting is used.
#' @param maxIters The maximum number of iterations (full cycles) for CD per grid point.
#' @param tol The tolerance which decides when to terminate CD (based on the relative change in the objective).
#' @param activeSet If TRUE, performs active set updates.
#' @param activeSetNum The number of consecutive times a support should appear before declaring support stabilization.
#' @param maxSwaps The maximum number of swaps used by CDPSI for each grid point.
#' @param scaleDownFactor This parameter decides how close the selected Lambda values are. The choice should be between
#' strictly between 0 and 1 (i.e., 0 and 1 are not allowed). For details, see our paper - Section 5 on Adaptive Selection
#' of Tuning Parameters).
#' @param screenSize The number of coordinates to cycle over when performing correlation screening.
#' @param autoLambda If FALSE, the user specifier a grid of Lambda0 values through the Lambda0Grid parameter. Otherwise,
#' if TRUE, the values of Lambda0 are automatically selected based on the data.
#' @param lambdaGrid A vector of Lambda0 values to use in computing the regularization path. This is ignored unless autoLambda0 = FALSE.
#' @return An S3 object of type "L0Learn" describing the regularization path. The object has the following members.
#' \item{a0}{For L0, this is a sequence of intercepts. Note for L0L1 and L0L2, a0 is a list of intercept sequences,
#' where each member of the list corresponds to a single gamma value.}
#' \item{beta}{For L0, this is a matrix of coefficients of dimensions p x \code{length(lambda)}, where each column
#' corresponds to a single lambda value. For L0L1 and L0L2, this is a list of coefficient matrices, where each matrix
#' corresponds to a single gamma value.}
#' \item{lambda}{For L0, lambda is a sequence of lambda values. For L0L1 and L0L1, it is a list of lambda sequences,
#' each corresponding to a single gamma value.}
#' \item{gamma}{For L0L1 and L0L2, this is a sequence of gamma values.}
#' \item{suppSize}{For L0, this is a sequence of support sizes (number of non-zero coefficients). For L0L1 and L02,
#' it is a list of support size sequences, each representing a single gamma value.}
#' \item{converged}{For L0, this is a sequence indicating whether the algorithm converged at the current point in the
#' regularization path. For L0L1 and L0L2, this is a list of sequences, each representing a single gamma value.}
#' @export
L0Learn.fit <- function(x,y, loss="SquaredError", penalty="L0", algorithm="CD", maxSuppSize=100, nLambda=100, nGamma=10,
						gammaMax=10, gammaMin=0.0001, partialSort = TRUE, maxIters=200,
						tol=1e-6, activeSet=TRUE, activeSetNum=3, maxSwaps=100, scaleDownFactor=0.8, screenSize=1000, autoLambda = TRUE, lambdaGrid = list(0))
{
	# The C++ function uses LambdaU = 1 for user-specified grid. In R, we use autoLambda0 = 0 for user-specified grid (thus the negation when passing the paramter to the function below)
	M <- .Call('_L0Learn_L0LearnFit', PACKAGE = 'L0Learn', x, y, loss, penalty, algorithm, maxSuppSize, nLambda, nGamma, gammaMax, gammaMin, partialSort, maxIters, tol, activeSet, activeSetNum, maxSwaps, scaleDownFactor, screenSize, !autoLambda, lambdaGrid)

	G <- list(beta = M$beta, lambda=lapply(M$lambda,signif, digits=6), a0=M$a0, converged = M$Converged, suppSize= M$SuppSize, gamma=M$gamma, penalty=penalty, loss=loss)

	class(G) <- "L0Learn"
	G$n <- dim(x)[1]
	G$p <- dim(x)[2]
	G
}
