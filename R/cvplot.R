#' @title Plot Cross-validation Errors
#'
#' @description Plots cross-validation errors
#' @param x L0Learn.fit object
#' @param gamma The gamma value for L0L1 and L0L2 models. This is ignored for L0.
#' @param ... ignore
#' @method plot L0Learn
#' @export
plot.L0Learn <- function(x, gamma, ...)
{
		if (x$penalty == "L0")
		{
				xvals = log10(unlist(x$lambda))
				yy = x$cvMeans
				sd = x$cvSDs
		}
		else
		{
				#gammaindex = match(gamma, x$gamma)
				gammaindex = which(abs(x$gamma-gamma)==min(abs(x$gamma-gamma)))
				xvals = log10(unlist(x$lambda[[gammaindex]]))
				yy = x$cvMeans[[gammaindex]]
				sd = x$cvSDs[[gammaindex]]
		}
		plot(xvals, yy, ylim=range(c(0, yy+sd)),
		    pch=19, xlab="Log(lambda)", ylab="CV Error")
		arrows(xvals, yy-sd, xvals, yy+sd, length=0.05, angle=90, code=3)
}
