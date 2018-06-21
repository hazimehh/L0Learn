#' @title Predict Response
#'
#' @description Predicts the response for a given sample
#' @param object The output of L0Learn.fit
#' @param ... ignore
#' @param newx A matrix on which predictions are made. The matrix should have p columns.
#' @param lambda The value(s) of lambda to use for prediction. A summary of the lambdas in the regularization
#' path can be obtained using \code{print(fit)}.
#' @param gamma The value of gamma to use for prediction. A summary of the gammas in the regularization
#' path can be obtained using \code{print(fit)}.
#' @method predict L0Learn
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
