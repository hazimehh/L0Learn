#' @title Extract Solutions
#'
#' @description Extracts a specific solution in the regularization path
#' @param object The output of L0Learn.fit
#' @param ... ignore
#' @param lambda The value(s) of lambda at which to extract the solution.
#' @param gamma The value of gamma at which to extract the solution. Note that, unlike lambda, this can only take single values.
#' @method coef L0Learn
#' @export
coef.L0Learn <- function(object,lambda,gamma=0, ...){
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
