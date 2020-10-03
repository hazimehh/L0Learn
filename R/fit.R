# import C++ compiled code
#' @useDynLib L0Learn
#' @importFrom Rcpp evalCpp
#' @importFrom methods as
#' @import Matrix

#' @title Fit an L0-regularized model
#'
#' @description Computes the regularization path for the specified loss function and
#' penalty function (which can be a combination of the L0, L1, and L2 norms).
#' @param x The data matrix.
#' @param y The response vector. For classification, we only support binary vectors.
#' @param loss The loss function. Currently we support the choices "SquaredError" (for regression), "Logistic" (for logistic regression), and "SquaredHinge" (for smooth SVM).
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
#' value of gamma in the grid is set to gammaMin * gammaMax. Note that this should be a strictly positive quantity.
#' @param partialSort If TRUE partial sorting will be used for sorting the coordinates to do greedy cycling (see our paper for
#' for details). Otherwise, full sorting is used.
#' @param maxIters The maximum number of iterations (full cycles) for CD per grid point.
#' @param tol The tolerance which decides when to terminate CD (based on the relative change in the objective).
#' @param activeSet If TRUE, performs active set updates.
#' @param activeSetNum The number of consecutive times a support should appear before declaring support stabilization.
#' @param maxSwaps The maximum number of swaps used by CDPSI for each grid point.
#' @param scaleDownFactor This parameter decides how close the selected Lambda values are. The choice should be
#' strictly between 0 and 1 (i.e., 0 and 1 are not allowed). Larger values lead to closer lambdas and typically to smaller
#' gaps between the support sizes. For details, see our paper - Section 5 on Adaptive Selection of Tuning Parameters).
#' @param screenSize The number of coordinates to cycle over when performing initial correlation screening.
#' @param autoLambda If FALSE, the user specifies a grid of Lambda values through the lambdaGrid parameter. Otherwise,
#' if TRUE, the values of Lambda are automatically selected based on the data.
#' @param lambdaGrid A grid of Lambda values to use in computing the regularization path. This is ignored unless autoLambda = FALSE.
#' LambdaGrid should be a list, where the ith element (corresponding to the ith gamma) should be a decreasing sequence of lambda values
#' which are used by the algorithm when fitting for the ith value of gamma (see the vignette for details).
#' @param excludeFirstK This parameter takes non-negative integers. The first excludeFirstK features in x will be excluded from variable selection,
#' i.e., the first excludeFirstK variables will not be included in the L0-norm penalty (they will still be included in the L1 or L2 norm penalties.).
#' @param intercept If FALSE, no intercept term is included in the model.
#' @param lows Lower bounds for coefficients. Either a scalar for all coefficients to have the same bound or a vector of size p (number of columns of X) where lows[i] is the lower bound for coefficient i.
#' @param highs Upper bounds for coefficients. Either a scalar for all coefficients to have the same bound or a vector of size p (number of columns of X) where highs[i] is the upper bound for coefficient i.
#' @return An S3 object of type "L0Learn" describing the regularization path. The object has the following members.
#' \item{a0}{a0 is a list of intercept sequences. The ith element of the list (i.e., a0[[i]]) is the sequence of intercepts corresponding to the ith gamma value (i.e., gamma[i]).}
#' \item{beta}{This is a list of coefficient matrices. The ith element of the list is a p x \code{length(lambda)} matrix which
#' corresponds to the ith gamma value. The jth column in each coefficient matrix is the vector of coefficients for the jth lambda value.}
#' \item{lambda}{This is the list of lambda sequences used in fitting the model. The ith element of lambda (i.e., lambda[[i]]) is the sequence
#' of Lambda values corresponding to the ith gamma value.}
#' \item{gamma}{This is the sequence of gamma values used in fitting the model.}
#' \item{suppSize}{This is a list of support size sequences. The ith element of the list is a sequence of support sizes (i.e., number of non-zero coefficients)
#' corresponding to the ith gamma value.}
#' \item{converged}{This is a list of sequences for checking whether the algorithm has converged at every grid point. The ith element of the list is a sequence
#' corresponding to the ith value of gamma, where the jth element in each sequence indicates whether the algorithm has converged at the jth value of lambda.}
#'
#' @examples
#' # Generate synthetic data for this example
#' data <- GenSynthetic(n=500,p=1000,k=10,seed=1)
#' X = data$X
#' y = data$y
#'
#' # Fit an L0 regression model with a maximum of 50 non-zeros using coordinate descent (CD)
#' fit1 <- L0Learn.fit(X, y, penalty="L0", maxSuppSize=50)
#' print(fit1)
#' # Extract the coefficients at lambda = 0.0325142
#' coef(fit1, lambda=0.0325142)
#' # Apply the fitted model on X to predict the response
#' predict(fit1, newx = X, lambda=0.0325142)
#'
#' # Fit an L0 regression model with a maximum of 50 non-zeros using CD and local search
#' fit2 <- L0Learn.fit(X, y, penalty="L0", algorithm="CDPSI", maxSuppSize=50)
#' print(fit2)
#'
#' # Fit an L0L2 regression model with 10 values of Gamma ranging from 0.0001 to 10, using CD
#' fit3 <- L0Learn.fit(X, y, penalty="L0L2", maxSuppSize=50, nGamma=10, gammaMin=0.0001, gammaMax = 10)
#' print(fit3)
#' # Extract the coefficients at lambda = 0.0361829 and gamma = 0.0001
#' coef(fit3, lambda=0.0361829, gamma=0.0001)
#' # Apply the fitted model on X to predict the response
#' predict(fit3, newx = X, lambda=0.0361829, gamma=0.0001)
#'
#' # Fit an L0 logistic regression model
#' # First, convert the response to binary
#' y = sign(y)
#' fit4 <- L0Learn.fit(X, y, loss="Logistic", maxSuppSize=20)
#' print(fit4)
#'
#' @export
L0Learn.fit <- function(x, y, loss="SquaredError", penalty="L0", algorithm="CD", 
                        maxSuppSize=100, nLambda=100, nGamma=10, gammaMax=10,
                        gammaMin=0.0001, partialSort = TRUE, maxIters=200,
						tol=1e-6, activeSet=TRUE, activeSetNum=3, maxSwaps=100,
						scaleDownFactor=0.8, screenSize=1000, autoLambda = TRUE,
						lambdaGrid = list(0), excludeFirstK=0, intercept = TRUE,
						lows=-Inf, highs=Inf) {

	# Some sanity checks for the inputs
	if ( !(loss %in% c("SquaredError","Logistic","SquaredHinge")) ){
			stop("The specified loss function is not supported.")
	}
	if ( !(penalty %in% c("L0","L0L2","L0L1")) ){
			stop("The specified penalty is not supported.")
	}
	if ( !(algorithm %in% c("CD","CDPSI")) ){
			stop("The specified algorithm is not supported.")
	}
	if (loss=="Logistic" | loss=="SquaredHinge"){
			if (dim(table(y)) != 2){
					stop("Only binary classification is supported. Make sure y has only 2 unique values.")
			}
			y = factor(y,labels=c(-1,1)) # returns a vector of strings
			y = as.numeric(levels(y))[y]

			if (penalty == "L0"){
					# Pure L0 is not supported for classification
					# Below we add a small L2 component.
					penalty = "L0L2"
					nGamma = 1
					gammaMax = 1e-7
					gammaMin = 1e-7
			}
	}
    is.scalar <- function(x) is.atomic(x) && length(x) == 1L && !is.character(x) && Im(x)==0 && !is.nan(x) && !is.na(x)
    
    p = dim(x)[[2]]
    
    if (is.scalar(lows)){
        lows = lows*rep(1, p)
    } else if (!all(sapply(lows, is.scalar)) || length(lows) != p) { 
        stop('Lows must be a vector of real values of length p')
    } 
    
    if (is.scalar(highs)){
        highs = highs*rep(1, p)
    } else if (!all(sapply(highs, is.scalar)) || length(highs) != p) { 
        stop('Highs must be a vector of real values of length p')
    } 
    
    if (any(lows >= highs) || any(lows > 0) || any(highs < 0)){
        stop("Bounds must conform to the following conditions: Lows <= 0, Highs >= 0, Lows < Highs")
    }
    
    if (algorithm == "CDPSI"){
        if (any(lows != -Inf) || any(highs != Inf)){
            stop("Bounds are not YET supported for CDPSI algorithm. Please raise
                 an issue at 'https://github.com/hazimehh/L0Learn' to express 
                 interest in this functionality")
        }
    }


	# The C++ function uses LambdaU = 1 for user-specified grid. In R, we use autoLambda0 = 0 for user-specified grid (thus the negation when passing the parameter to the function below)
	M <- .Call('_L0Learn_L0LearnFit', PACKAGE = 'L0Learn', x, y, loss, penalty, algorithm, maxSuppSize, nLambda, nGamma, gammaMax, gammaMin, partialSort, maxIters, tol, activeSet, activeSetNum, maxSwaps, scaleDownFactor, screenSize, !autoLambda, lambdaGrid, excludeFirstK, intercept, lows, highs)

	settings = list()
	settings[[1]] = intercept # Settings only contains intercept for now. Might include additional elements later.
	names(settings) <- c("intercept")

	# Find potential support sizes exceeding maxSuppSize and remove them (this is due to
	# the C++ core whose last solution can exceed maxSuppSize
	for (i in 1:length(M$SuppSize)){
			last = length(M$SuppSize[[i]])
			if (M$SuppSize[[i]][last] > maxSuppSize){
					if (last == 1){
							print("Warning! Only 1 element in path with support size > maxSuppSize.")
							print("Try increasing maxSuppSize to resolve the issue.")
					} else{
							M$SuppSize[[i]] = M$SuppSize[[i]][-last]
							M$Converged[[i]] = M$Converged[[i]][-last]
							M$lambda[[i]] = M$lambda[[i]][-last]
							M$a0[[i]] = M$a0[[i]][-last]
							M$beta[[i]] = as(M$beta[[i]][,-last], "sparseMatrix") # conversion to sparseMatrix is necessary to handle the case of a single column
					}
			}
	}

	G <- list(beta = M$beta, lambda=lapply(M$lambda,signif, digits=6), a0=M$a0, converged = M$Converged, suppSize= M$SuppSize, gamma=M$gamma, penalty=penalty, loss=loss, settings = settings)


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
