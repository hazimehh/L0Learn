#' @title Fit an L0-regularized model
#'
#' @description Computes the regularization path for the specified loss function and
#' choice of regularization (which can be a combination of the L0, L1, and L2 (squared) norms).
#' @param X The data matrix.
#' @param y The response vector.
#' @param Loss The loss function to be minimized. The currently supported choice is"SquaredError".
#' @param Penalty The type of regularization. This can take either one of the following choices:
#' "L0", "L0L2", and "L0L1".
#' @param Algorithm The type of algorithm used to minimize the objective. Currently "CD" and "CDPSI" are
#' are supported. "CD" is a variant of cyclic coordinate descent and can run very fast. "CDPSI" performs
#' local combinatorial search on top of CD and thus can achieve higher quality solutions (at the expense
#' of increased running time).
#' @param MaxSuppSize The maximum support size to reach in the grid before termination. We recommend setting
#' this to a small fraction of min(n,p) (e.g. 0.05 * min(n,p)) as L0 regularization typically selects a small
#' portion of nonzeros.
#' @param NLambda The number of Lambda values to select (recall that Lambda is the regularization parameter
#' corresponding to the L0 norm).
#' @param NGamma The number of Gamma values to select (recall that Gamma is the regularization parameter
#' corresponding to L1 or L2, depending on the chosen penalty).
#' @param GammaMax The maximum value of Gamma when using the L0L2 penalty. For the L0L1 penalty this is
#' automatically selected by the toolkit.
#' @param GammaMin The minimum value of Gamma when using the L0L2 penalty. For the L0L1 penalty, GammaMin
#' specifies the fraction of GammaMax at which the grid ends.
#' @param PartialSort If true partial sorting will be used for sorting the coordinates (see our paper for
#' for details). Otherwise, full sorting is used.
#' @param MaxIters The maximum number of iterations (full cycles) for CD per grid point.
#' @param Tol The tolerance which decides when to terminate CD (based on the relative change in the objective).
#' @param ActiveSet If true, performs active set updates.
#' @param ActiveSetNum The number of consecutive times a support should appear before declaring support stabilization.
#' @param MaxSwaps The maximum number of swaps used by CDPSI for each grid point.
#' @param ScaleDownFactor This parameter decides how close the selected Lambda values are. The choice should be between
#' strictly between 0 and 1 (i.e., 0 and 1 are not allowed). For details, see our paper - Section 5 on Adaptive Selection
#' of Tuning Parameters).
#' @param ScreenSize The number of coordinates to cycle over when performing correlation screening.
#' @param AutoLambda0 If FALSE, the user specifier a grid of Lambda0 values through the Lambda0Grid parameter. Otherwise,
#' if TRUE, the values of Lambda0 are automatically selected based on the data.
#' @param Lambda0Grid A vector of Lambda0 values to use in computing the regularization path. This is ignored unless AutoLambda0 = FALSE.
#' @return An object of type "L0Learn" containing all the solutions in the computed regularization path.
#' @export
L0Learn.fit <- function(X,y, Loss="SquaredError", Penalty="L0", Algorithm="CD", MaxSuppSize=100, NLambda=100, NGamma=10,
						GammaMax=10, GammaMin=0.0001, PartialSort = TRUE, MaxIters=200,
						Tol=1e-6, ActiveSet=TRUE, ActiveSetNum=3, MaxSwaps=100, ScaleDownFactor=0.8, ScreenSize=1000, AutoLambda0 = FALSE, Lambda0Grid = c(0))
{
	# The C++ function uses LambdaU = 1 for user-specified grid. In R, we use AutoLambda0 = 0 for user-specified grid (thus the negation when passing the paramter to the function below)
	G <- .Call('_L0Learn_L0LearnFit', PACKAGE = 'L0Learn', X, y, Loss, Penalty, Algorithm, MaxSuppSize, NLambda, NGamma, GammaMax, GammaMin, PartialSort, MaxIters, Tol, ActiveSet, ActiveSetNum, MaxSwaps, ScaleDownFactor, ScreenSize, !AutoLambda0, Lambda0Grid)

	class(G) <- "L0Learn"
	G$.n <- dim(X)[1]
	G$.p <- dim(X)[2]
	G
}

#' @title Extract Solutions
#'
#' @description Extracts a specific solution in the path
#' @param fit The output of L0Learn.fit
#' @param index The index of the solution to extract. A summary of the solutions
#' and their indices can be viewed by print(fit)
#' @export
L0Learn.coef <- function(fit,index){
	p = fit$.p
	B = sparseVector(unlist(fit["BetaValues"][[1]][index]), unlist(fit["BetaIndices"][[1]][index]),p)
	B = as(B,"sparseMatrix")
	intercept = fit["Intercept"][[1]][index]
	#print(list(summary(B),intercept))
	out = list(Beta=B,Intercept=intercept);
	class(out) <- "L0Learncoef"
	out
}

#' @title Print L0Learn.coef object
#'
#' @description Prints a summary of L0Learn.coef
#' @param x L0Learn.coef object
#' @param ... ignore
#' @method print L0Learncoef
#' @export
print.L0Learncoef <- function(x, ...){
	print(list(Beta=summary(x$Beta),Intercept=x$Intercept))
}


#' @title Prediction
#'
#' @description Predicts the response for a given sample
#' @param fit The output of L0Learn.fit
#' @param x The sample which can be a vector or a matrix
#' @param index The index of the solution to use for prediction. A summary of the solutions
#' and their indices can be viewed by print(fit)
#' @export
L0Learn.predict <- function(fit,x,index){
	c = L0Learn.coef(fit,index)
	c$Intercept + x%*%c$Beta
}

#' @title Print L0Learn.fit object
#'
#' @description Prints a summary of L0Learn.fit
#' @param x L0Learn.fit object
#' @param ... ignore
#' @method print L0Learn
#' @export
print.L0Learn <- function(x, ...){
	if(exists("Gamma",x)){
		data.frame(x["Lambda"], x["Gamma"],x["SuppSize"])
	}
	else{
		data.frame(x["Lambda"],x["SuppSize"])
	}
}
