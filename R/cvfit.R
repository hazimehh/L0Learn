#' @title Cross Validation
#'
#' @inheritParams L0Learn.fit
#' @description Fits an L0 model and performs K-fold cross-validation.
#' @param nFolds The number of folds for cross-validation.
#' @param seed The seed used in randomly shuffling the data for cross-validation.
#' @return An S3 object of type "L0Learn" describing the regularization path. The object has the following members.
#' \item{cvMeans}{This is a list, where the ith element is a sequence of cross-validation errors corresponding to the ith gamma value}
#' \item{cvSDs}{This a list, where the ith element is a sequence of standard deviations for the cross-validation errors: cvSDs[[i]] corresponds to cvMeans[[i]].}
#' \item{fit}{The fitted model with type "L0Learn", i.e., this is the same object returned by \code{\link{L0Learn.fit}}.}
#'
#' @examples
#' # Generate synthetic data for this example
#' data <- GenSynthetic(n=500,p=1000,k=10,seed=1)
#' X = data$X
#' y = data$y
#'
#' # Perform 5-fold cross-validation on an L0L2 Model with 5 values of
#' # Gamma ranging from 0.0001 to 10
#' fit <- L0Learn.cvfit(X, y, nFolds=5, seed=1, penalty="L0L2", maxSuppSize=20, nGamma=5,
#' gammaMin=0.0001, gammaMax = 10)
#' print(fit)
#' # Plot the graph of cross-validation error versus lambda for gamma = 0.0001
#' plot(fit, gamma=0.0001)
#' # Extract the coefficients at lambda = 0.0361829 and gamma = 0.0001
#' coef(fit, lambda=0.0361829, gamma=0.0001)
#' # Apply the fitted model on X to predict the response
#' predict(fit, newx = X, lambda=0.0361829, gamma=0.0001)
#'
#' @export
L0Learn.cvfit <- function(x,y, loss="SquaredError", penalty="L0", algorithm="CD", maxSuppSize=100, nLambda=100, nGamma=10,
						gammaMax=10, gammaMin=0.0001, partialSort = TRUE, maxIters=200,
						tol=1e-6, activeSet=TRUE, activeSetNum=3, maxSwaps=100, scaleDownFactor=0.8, screenSize=1000, autoLambda = TRUE, lambdaGrid = list(0), nFolds=10, seed=1)
{
	set.seed(seed)
	# The C++ function uses LambdaU = 1 for user-specified grid. In R, we use AutoLambda0 = 0 for user-specified grid (thus the negation when passing the paramter to the function below)
	M <- .Call('_L0Learn_L0LearnCV', PACKAGE = 'L0Learn', x, y, loss, penalty, algorithm, maxSuppSize, nLambda, nGamma, gammaMax, gammaMin, partialSort, maxIters, tol, activeSet, activeSetNum, maxSwaps, scaleDownFactor, screenSize, !autoLambda, lambdaGrid,nFolds,seed)
	fit <- list(beta = M$beta, lambda=lapply(M$lambda,signif, digits=6), a0=M$a0, converged = M$Converged, suppSize= M$SuppSize, gamma=M$gamma, penalty=penalty, loss=loss)
	if (is.null(colnames(x))){
			varnames <- 1:dim(x)[2]
	} else {
			varnames <- colnames(x)
	}
	fit$varnames <- varnames
	class(fit) <- "L0Learn"
	fit$n <- dim(x)[1]
	fit$p <- dim(x)[2]
	G <- list(fit=fit, cvMeans=M$CVMeans,cvSDs=M$CVSDs)
	class(G) <- "L0LearnCV"
	G
}
