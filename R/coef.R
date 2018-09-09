#' @title Extract Solutions
#'
#' @description Extracts a specific solution in the regularization path
#' @param object The output of L0Learn.fit or L0Learn.cvfit
#' @param ... ignore
#' @param lambda The value of lambda at which to extract the solution.
#' @param gamma The value of gamma at which to extract the solution.
#' @method coef L0Learn

#' @examples
#' # Generate synthetic data for this example
#' data <- GenSynthetic(n=500,p=1000,k=10,seed=1)
#' X = data$X
#' y = data$y
#'
#' # Fit an L0L2 Model with 10 values of Gamma ranging from 0.0001 to 10, using coordinate descent
#' fit <- L0Learn.fit(X, y, penalty="L0L2", maxSuppSize=50, nGamma=10, gammaMin=0.0001, gammaMax = 10)
#' print(fit)
#' # Extract the coefficients of the solution at lambda = 0.0361829 and gamma = 0.0001
#' coef(fit, lambda=0.0361829, gamma=0.0001)
#'
#' @export
coef.L0Learn <- function(object,lambda,gamma=0, ...){
		diffGamma = abs(object$gamma-gamma)
		gammaindex = which(diffGamma==min(diffGamma))
		diffLambda = abs(lambda - object$lambda[[gammaindex]])
		indices = which(diffLambda == min(diffLambda))
		#indices = match(lambda,object$lambda[[gammaindex]])
		if (object$settings$intercept){
				t = rbind(object$a0[[gammaindex]][indices],object$beta[[gammaindex]][,indices,drop=FALSE])
				rownames(t) = c("Intercept",paste(rep("V",object$p),1:object$p,sep=""))
		}
		else{
				t = object$beta[[gammaindex]][,indices,drop=FALSE]
				rownames(t) = paste(rep("V",object$p),1:object$p,sep="")
		}
		t
}

#' @rdname coef.L0Learn
#' @method coef L0LearnCV
#' @export
coef.L0LearnCV <- function(object,lambda,gamma=0, ...){
    coef.L0Learn(object$fit,lambda,gamma, ...)
}
