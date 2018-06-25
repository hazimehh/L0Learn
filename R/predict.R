#' @title Predict Response
#'
#' @description Predicts the response for a given sample
#' @param object The output of L0Learn.fit or L0Learn.cvfit
#' @param ... ignore
#' @param newx A matrix on which predictions are made. The matrix should have p columns.
#' @param lambda The value(s) of lambda to use for prediction. A summary of the lambdas in the regularization
#' path can be obtained using \code{print(fit)}.
#' @param gamma The value of gamma to use for prediction. A summary of the gammas in the regularization
#' path can be obtained using \code{print(fit)}.
#' @method predict L0Learn
#' @examples
#' # Generate synthetic data for this example
#' data <- GenSynthetic(n=500,p=1000,k=10,seed=1)
#' X = data$X
#' y = data$y
#'
#' # Fit an L0L2 Model with 10 values of Gamma ranging from 0.0001 to 10, using coordinate descent
#' fit <- L0Learn.fit(X,y, penalty="L0L2", maxSuppSize=50, nGamma=10, gammaMin=0.0001, gammaMax = 10)
#' print(fit)
#' # Apply the fitted model with lambda=0.0361829 and gamma=0.0001 on X to predict the response
#' predict(fit, newx = X, lambda=0.0361829, gamma=0.0001)
#'
#' @export
predict.L0Learn <- function(object,newx,lambda,gamma=0, ...)
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

#' @rdname predict.L0Learn
#' @method predict L0LearnCV
#' @export
predict.L0LearnCV <- function(object,newx,lambda,gamma=0, ...)
{
    predict.L0Learn(object$fit,newx,lambda,gamma, ...)
}
