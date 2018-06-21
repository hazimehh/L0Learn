#' @title Cross Validation
#'
#' @inheritParams L0Learn.fit
#' @description Fits an L0 model on the full data and performs K-fold cross-validation.
#' @param nFolds The number of folds for cross-validation.
#' @param seed The seed used in randomly shuffling the data for cross-validation.
#' @return An S3 object of type "L0Learn" describing the regularization path. The object has the following members.
#' \item{cvMeans}{For L0, this is a sequence of cross-validation errors: cvMeans[i] corresponds to the solution indexed by lambda[i].
#' For L0L1 and L0L2, cvMeans is a list, where each element is a sequence corresponding to a particular gamma, i.e.,
#' cvMeans[[i]] is the sequence of cross-validation errors corresponding to gamma[i].}
#' \item{cvSDs}{For L0, this is a sequence of standard deviations for the cross-validation errors. For L0L1 and L0L2, it is a list of
#' sequences: cvSDs[[i]] corresponds to cvMeans[[i]].}
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
L0Learn.cvfit <- function(x,y, loss="SquaredError", penalty="L0", algorithm="CD", maxSuppSize=100, nLambda=100, nGamma=10,
						gammaMax=10, gammaMin=0.0001, partialSort = TRUE, maxIters=200,
						tol=1e-6, activeSet=TRUE, activeSetNum=3, maxSwaps=100, scaleDownFactor=0.8, screenSize=1000, autoLambda = TRUE, lambdaGrid = list(0), nFolds=10, seed=1)
{
	set.seed(seed)
	# The C++ function uses LambdaU = 1 for user-specified grid. In R, we use AutoLambda0 = 0 for user-specified grid (thus the negation when passing the paramter to the function below)
	M <- .Call('_L0Learn_L0LearnCV', PACKAGE = 'L0Learn', x, y, loss, penalty, algorithm, maxSuppSize, nLambda, nGamma, gammaMax, gammaMin, partialSort, maxIters, tol, activeSet, activeSetNum, maxSwaps, scaleDownFactor, screenSize, !autoLambda, lambdaGrid,nFolds,seed)

	G <- list(beta = M$beta, lambda=lapply(M$lambda,signif, digits=6), a0=M$a0, converged = M$Converged, suppSize= M$SuppSize, gamma=M$gamma, penalty=Penalty, cvMeans=M$CVMeans,cvSDs=M$CVSDs, loss=Loss)

	class(G) <- "L0Learn"
	G$n <- dim(X)[1]
	G$p <- dim(X)[2]
	G
}
