#' @title Extract Solutions
#'
#' @description Extracts a specific solution in the regularization path
#' @param object The output of L0Learn.fit or L0Learn.cvfit
#' @param ... ignore
#' @param lambda The value(s) of lambda at which to extract the solution.
#' @param gamma The value of gamma at which to extract the solution. Note that, unlike lambda, this can only take single values.
#' @method coef L0Learn
#' @export
coef.L0Learn <- function(object,lambda,gamma=0, ...){
		gammaindex = which(abs(object$gamma-gamma)==min(abs(object$gamma-gamma)))
		indices = match(lambda,object$lambda[[gammaindex]])
		t = rbind(object$a0[[gammaindex]][indices],object$beta[[gammaindex]][,indices,drop=FALSE])
		rownames(t) = c("Intercept",paste(rep("V",object$p),1:object$p,sep=""))
		t
}

#' @rdname coef.L0Learn
#' @method coef L0LearnCV
#' @export
coef.L0LearnCV <- function(object,lambda,gamma=0, ...){
    coef.L0Learn(object$fit,lambda,gamma=0, ...)
}
