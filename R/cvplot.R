#' @title Plot Cross-validation Errors
#'
#' @description Plots cross-validation errors
#' @param x L0Learn.fit object
#' @param gamma The gamma value for L0L1 and L0L2 models. This is ignored for L0.
#' @param ... ignore
#' @method plot L0LearnCV
#' @export
plot.L0LearnCV <- function(x, gamma, ...)
{
		#gammaindex = match(gamma, x$gamma)
		gammaindex = which(abs(x$fit$gamma-gamma)==min(abs(x$fit$gamma-gamma)))
		xvals = log10(unlist(x$fit$lambda[[gammaindex]]))
		yy = x$cvMeans[[gammaindex]]
		sd = x$cvSDs[[gammaindex]]
		plot(xvals, yy, ylim=range(c(0, yy+sd)),
		    pch=19, xlab="Log(lambda)", ylab="CV Error")
		arrows(xvals, yy-sd, xvals, yy+sd, length=0.05, angle=90, code=3)
}
