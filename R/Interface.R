#' @title Fit an L0-regularized model
#'
#' @description Computes the regularization path for the specified loss function and
#' choice of regularization (which can be a combination of the L0, L1, and L2 (squared) norms).
#' @param X The data matrix.
#' @param y The response vector.
#' @param Loss The loss function to be minimized. The currently supported choice is "SquaredError".
#' @param Penalty The type of regularization. This can take either one of the following choices:
#' "L0", "L0L2", and "L0L1".
#' @param Algorithm The type of algorithm used to minimize the objective. Currently "CD" and "CDPSI" are
#' are supported. "CD" is a variant of cyclic coordinate descent and can run very fast. "CDPSI" performs
#' local combinatorial search on top of CD and thus can achieve higher quality solutions (at the expense
#' of increased running time).
#' @param MaxSuppSize The maximum support size to reach in the grid before termination. We recommend setting
#' this to a small fraction of min(n,p) (e.g. 0.05 * min(n,p)) as L0 regularization typically selects a small
#' portion of non-zeros.
#' @param NLambda The number of Lambda values to select (recall that Lambda is the regularization parameter
#' corresponding to the L0 norm).
#' @param NGamma The number of Gamma values to select (recall that Gamma is the regularization parameter
#' corresponding to L1 or L2, depending on the chosen penalty).
#' @param GammaMax The maximum value of Gamma when using the L0L2 penalty. For the L0L1 penalty this is
#' automatically selected by the toolkit.
#' @param GammaMin The minimum value of Gamma when using the L0L2 penalty. For the L0L1 penalty, GammaMin
#' specifies the fraction of GammaMax at which the grid ends.
#' @param PartialSort If TRUE partial sorting will be used for sorting the coordinates (see our paper for
#' for details). Otherwise, full sorting is used.
#' @param MaxIters The maximum number of iterations (full cycles) for CD per grid point.
#' @param Tol The tolerance which decides when to terminate CD (based on the relative change in the objective).
#' @param ActiveSet If TRUE, performs active set updates.
#' @param ActiveSetNum The number of consecutive times a support should appear before declaring support stabilization.
#' @param MaxSwaps The maximum number of swaps used by CDPSI for each grid point.
#' @param ScaleDownFactor This parameter decides how close the selected Lambda values are. The choice should be between
#' strictly between 0 and 1 (i.e., 0 and 1 are not allowed). For details, see our paper - Section 5 on Adaptive Selection
#' of Tuning Parameters).
#' @param ScreenSize The number of coordinates to cycle over when performing correlation screening.
#' @param AutoLambda If FALSE, the user specifier a grid of Lambda0 values through the Lambda0Grid parameter. Otherwise,
#' if TRUE, the values of Lambda0 are automatically selected based on the data.
#' @param LambdaGrid A vector of Lambda0 values to use in computing the regularization path. This is ignored unless AutoLambda0 = FALSE.
#' @return An S3 object of type "L0Learn" describing the regularization path. The object has the following members:
#' \item{a0} {For L0, this is a sequence of intercepts. Note for L0L1 and L0L2, a0 is a list of intercept sequences,
#' where each member of the list corresponds to a single gamma value.}
#' \item{beta} {For L0, this is a matrix of coefficients of dimensions p x \code{length(lambda)}, where each column
#' corresponds to a single lambda value. For L0L1 and L0L2, this is a list of coefficient matrices, where each matrix
#' corresponds to a single gamma value. }
#' \item{lambda} {For L0, lambda is a sequence of lambda values. For L0L1 and L0L1, it is a list of lambda sequences,
#' each corresponding to a single gamma value.}
#' \item{gamma} {For L0L1 and L0L2, this is a sequence of gamma values.}
#' \item{suppsize} {For L0, this is a sequence of support sizes (number of non-zero coefficients). For L0L1 and L02,
#' it is a list of support size sequences, each representing a single gamma value.}
#' \item{converged} {For L0, this is a sequence indicating whether the algorithm converged at the current point in the
#' regularization path. For L0L1 and L0L2, this is a list of sequences, each representing a single gamma value.}
#' @export
L0Learn.fit <- function(X,y, Loss="SquaredError", Penalty="L0", Algorithm="CD", MaxSuppSize=100, NLambda=100, NGamma=10,
						GammaMax=10, GammaMin=0.0001, PartialSort = TRUE, MaxIters=200,
						Tol=1e-6, ActiveSet=TRUE, ActiveSetNum=3, MaxSwaps=100, ScaleDownFactor=0.8, ScreenSize=1000, AutoLambda = TRUE, LambdaGrid = list(0))
{
	# The C++ function uses LambdaU = 1 for user-specified grid. In R, we use AutoLambda0 = 0 for user-specified grid (thus the negation when passing the paramter to the function below)
	M <- .Call('_L0Learn_L0LearnFit', PACKAGE = 'L0Learn', X, y, Loss, Penalty, Algorithm, MaxSuppSize, NLambda, NGamma, GammaMax, GammaMin, PartialSort, MaxIters, Tol, ActiveSet, ActiveSetNum, MaxSwaps, ScaleDownFactor, ScreenSize, !AutoLambda, LambdaGrid)

	G = list(beta = M$beta, lambda=lapply(M$lambda,signif, digits=6), a0=M$a0, converged = M$Converged, suppsize= M$SuppSize, gamma=M$gamma, penalty=Penalty)

	class(G) <- "L0Learn"
	G$n <- dim(X)[1]
	G$p <- dim(X)[2]
	G
}


#' @title Cross Validation
#'
#' @description Fits an L0 model on the full data and performs K-fold cross-validation.
#' @param X The data matrix.
#' @param y The response vector.
#' @param Loss The loss function to be minimized. The currently supported choice is "SquaredError".
#' @param Penalty The type of regularization. This can take either one of the following choices:
#' "L0", "L0L2", and "L0L1".
#' @param Algorithm The type of algorithm used to minimize the objective. Currently "CD" and "CDPSI" are
#' are supported. "CD" is a variant of cyclic coordinate descent and can run very fast. "CDPSI" performs
#' local combinatorial search on top of CD and thus can achieve higher quality solutions (at the expense
#' of increased running time).
#' @param MaxSuppSize The maximum support size to reach in the grid before termination. We recommend setting
#' this to a small fraction of min(n,p) (e.g. 0.05 * min(n,p)) as L0 regularization typically selects a small
#' portion of non-zeros.
#' @param NLambda The number of Lambda values to select (recall that Lambda is the regularization parameter
#' corresponding to the L0 norm).
#' @param NGamma The number of Gamma values to select (recall that Gamma is the regularization parameter
#' corresponding to L1 or L2, depending on the chosen penalty).
#' @param GammaMax The maximum value of Gamma when using the L0L2 penalty. For the L0L1 penalty this is
#' automatically selected by the toolkit.
#' @param GammaMin The minimum value of Gamma when using the L0L2 penalty. For the L0L1 penalty, GammaMin
#' specifies the fraction of GammaMax at which the grid ends.
#' @param PartialSort If TRUE partial sorting will be used for sorting the coordinates (see our paper for
#' for details). Otherwise, full sorting is used.
#' @param MaxIters The maximum number of iterations (full cycles) for CD per grid point.
#' @param Tol The tolerance which decides when to terminate CD (based on the relative change in the objective).
#' @param ActiveSet If TRUE, performs active set updates.
#' @param ActiveSetNum The number of consecutive times a support should appear before declaring support stabilization.
#' @param MaxSwaps The maximum number of swaps used by CDPSI for each grid point.
#' @param ScaleDownFactor This parameter decides how close the selected Lambda values are. The choice should be between
#' strictly between 0 and 1 (i.e., 0 and 1 are not allowed). For details, see our paper - Section 5 on Adaptive Selection
#' of Tuning Parameters).
#' @param ScreenSize The number of coordinates to cycle over when performing correlation screening.
#' @param AutoLambda If FALSE, the user specifier a grid of Lambda0 values through the Lambda0Grid parameter. Otherwise,
#' if TRUE, the values of Lambda0 are automatically selected based on the data.
#' @param LambdaGrid A vector of Lambda0 values to use in computing the regularization path. This is ignored unless AutoLambda0 = FALSE.
#' @param Nfolds The number of folds for cross-validation
#' @param Seed The seed used in randomly shuffling the data for cross-validation
#' @return An S3 object of type "L0Learn" describing the regularization path. The object has the following members:
#' \item{a0} {For L0, this is a sequence of intercepts. Note for L0L1 and L0L2, a0 is a list of intercept sequences,
#' where each member of the list corresponds to a single gamma value.}
#' \item{beta} {For L0, this is a matrix of coefficients of dimensions p x \code{length(lambda)}, where each column
#' corresponds to a single lambda value. For L0L1 and L0L2, this is a list of coefficient matrices, where each matrix
#' corresponds to a single gamma value. }
#' \item{lambda} {For L0, lambda is a sequence of lambda values. For L0L1 and L0L1, it is a list of lambda sequences,
#' each corresponding to a single gamma value.}
#' \item{gamma} {For L0L1 and L0L2, this is a sequence of gamma values.}
#' \item{suppsize} {For L0, this is a sequence of support sizes (number of non-zero coefficients). For L0L1 and L02,
#' it is a list of support size sequences, each representing a single gamma value.}
#' \item{converged} {For L0, this is a sequence indicating whether the algorithm converged at the current point in the
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

#' @title Extract Solutions
#'
#' @description Extracts a specific solution in the regularization path
#' @param object The output of L0Learn.fit
#' @param lambda The value(s) of lambda at which to extract the solution.
#' @param gamma The value of gamma at which to extract the solution. Note that, unlike lambda, this can only take single values.
#' @export
coef.L0Learn <- function(object,lambda,gamma=0){
		if (object$penalty=="L0")
		{
				indices = match(lambda,object$lambda)
				t = rbind(object$a0[indices],object$beta[,indices,drop=FALSE])
				rownames(t) = c("Intercept",paste(rep("V",object$p),1:object$p,sep=""))

		}
		else
		{
				gammaindex = which(abs(object$gamma-gamma)==min(abs(object$gamma-gamma)))
				indices = match(lambda,object$lambda[[gammaindex]])
				t = rbind(object$a0[[gammaindex]][indices],object$beta[[gammaindex]][,indices,drop=FALSE])
				rownames(t) = c("Intercept",paste(rep("V",object$p),1:object$p,sep=""))
		}
		t
}



#' @title Predict Response
#'
#' @description Predicts the response for a given sample
#' @param object The output of L0Learn.fit
#' @param newx A matrix on which predictions are made. The matrix should have p columns.
#' @param lambda The value(s) of lambda to use for prediction. A summary of the lambdas in the regularization
#' path can be obtained using \code{print(fit)}.
#' @param gamma The value of gamma to use for prediction. A summary of the gammas in the regularization
#' path can be obtained using \code{print(fit)}.
#' @param ... ignore
#' @export
predict.L0Learn <- function(object,newx,lambda,gamma=0)
{
		beta = coef.L0Learn(object, lambda, gamma)
		# add a column of ones for the intercept
		x = cbind(1,newx)
		prediction = x%*%beta
		if (object$loss == "Logistic" || object$loss == "SquaredHinge")
		{
				prediction = sign(prediction)
		}
		prediction
}

#' @title Print L0Learn.fit object
#'
#' @description Prints a summary of L0Learn.fit
#' @param x L0Learn.fit object
#' @param ... ignore
#' @method print L0Learn
#' @export
print.L0Learn <- function(x, ...){
	if(x$penalty!="L0"){
		gammas = rep(x$gamma, times=lapply(x$lambda, length) )
		data.frame(lambda = unlist(x["lambda"]), gamma = gammas, suppsize = unlist(x["suppsize"]), row.names = NULL)
	}
	else{
		data.frame(lambda = unlist(x["lambda"]), suppsize = x["suppsize"], row.names = NULL)
	}
}

#' @title Plot Cross-validation Errors
#'
#' @description Plots cross-validation errors
#' @param x L0Learn.fit object
#' @param ... ignore
#' @method plot L0Learn
#' @export
plot.L0Learn <- function(x, gamma, ...)
{
		if (x$penalty == "L0")
		{
				xvals = log10(unlist(x$lambda))
				yy = x$cvmeans
				sd = x$cvsds
		}
		else
		{
				#gammaindex = match(gamma, x$gamma)
				gammaindex = which(abs(x$gamma-gamma)==min(abs(x$gamma-gamma)))
				xvals = log10(unlist(x$lambda[[gammaindex]]))
				yy = x$cvmeans[[gammaindex]]
				sd = x$cvsds[[gammaindex]]
		}
		plot(xvals, yy, ylim=range(c(0, yy+sd)),
		    pch=19, xlab="Log(lambda)", ylab="CV Error")
		arrows(xvals, yy-sd, xvals, yy+sd, length=0.05, angle=90, code=3)
}
