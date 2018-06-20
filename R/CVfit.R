#' @title Cross Validation
#'
#' @inheritParams L0Learn.fit
#' @description Fits an L0 model on the full data and performs K-fold cross-validation.
#' @param Nfolds The number of folds for cross-validation.
#' @param Seed The seed used in randomly shuffling the data for cross-validation.
#' @return An S3 object of type "L0Learn" describing the regularization path. The object has the following members.
#' \item{cvmeans}{For L0, this is a sequence of cross-validation errors: cvmeans[i] corresponds to the solution indexed by lambda[i].
#' For L0L1 and L0L2, cvmeans is a list, where each element is a sequence corresponding to a particular gamma, i.e.,
#' cvmeans[[i]] is the sequence of cross-validation errors corresponding to gamma[i].}
#' \item{cvsds}{For L0, this is a sequence of standard deviations for the cross-validation errors. For L0L1 and L0L2, it is a list of
#' sequences: cvsds[[i]] corresponds to cvmeans[[i]].}
#' \item{a0}{For L0, this is a sequence of intercepts. Note for L0L1 and L0L2, a0 is a list of intercept sequences,
#' where each member of the list corresponds to a single gamma value.}
#' \item{beta}{For L0, this is a matrix of coefficients of dimensions p x \code{length(lambda)}, where each column
#' corresponds to a single lambda value. For L0L1 and L0L2, this is a list of coefficient matrices, where each matrix
#' corresponds to a single gamma value.}
#' \item{lambda}{For L0, lambda is a sequence of lambda values. For L0L1 and L0L1, it is a list of lambda sequences,
#' each corresponding to a single gamma value.}
#' \item{gamma}{For L0L1 and L0L2, this is a sequence of gamma values.}
#' \item{suppsize}{For L0, this is a sequence of support sizes (number of non-zero coefficients). For L0L1 and L02,
#' it is a list of support size sequences, each representing a single gamma value.}
#' \item{converged}{For L0, this is a sequence indicating whether the algorithm converged at the current point in the
#' regularization path. For L0L1 and L0L2, this is a list of sequences, each representing a single gamma value.}
#' @export
L0Learn.cvfit <- function(X,y, Loss="SquaredError", Penalty="L0", Algorithm="CD", MaxSuppSize=100, NLambda=100, NGamma=10,
						GammaMax=10, GammaMin=0.0001, PartialSort = TRUE, MaxIters=200,
						Tol=1e-6, ActiveSet=TRUE, ActiveSetNum=3, MaxSwaps=100, ScaleDownFactor=0.8, ScreenSize=1000, AutoLambda = TRUE, LambdaGrid = list(0), Nfolds=10, Seed=1)
{
	set.seed(Seed)
	# The C++ function uses LambdaU = 1 for user-specified grid. In R, we use AutoLambda0 = 0 for user-specified grid (thus the negation when passing the paramter to the function below)
	M <- .Call('_L0Learn_L0LearnCV', PACKAGE = 'L0Learn', X, y, Loss, Penalty, Algorithm, MaxSuppSize, NLambda, NGamma, GammaMax, GammaMin, PartialSort, MaxIters, Tol, ActiveSet, ActiveSetNum, MaxSwaps, ScaleDownFactor, ScreenSize, !AutoLambda, LambdaGrid,Nfolds,Seed)

	G = list(beta = M$beta, lambda=lapply(M$lambda,signif, digits=6), a0=M$a0, converged = M$Converged, suppsize= M$SuppSize, gamma=M$gamma, penalty=Penalty, cvmeans=M$CVMeans,cvsds=M$CVSDs, loss=Loss)

	class(G) <- "L0Learn"
	G$n <- dim(X)[1]
	G$p <- dim(X)[2]
	G
}
