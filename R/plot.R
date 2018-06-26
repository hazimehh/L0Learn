#' @title Plot Regularization Path
#'
#' @description Plots the regularization path for a given gamma.
#' @param gamma The value of gamma at which to plot.
#' @param x The output of L0Learn.fit
#' @param ... ignore
#'
#' @examples
#' # Generate synthetic data for this example
#' data <- GenSynthetic(n=500,p=1000,k=10,seed=1)
#' X = data$X
#' y = data$y
#' # Fit an L0 Model with a maximum of 50 non-zeros
#' fit <- L0Learn.fit(X, y, penalty="L0", maxSuppSize=50)
#' plot(fit, gamma=0)
#'
#' @import ggplot2
#' @importFrom reshape2 melt
#' @method plot L0Learn
#' @export
plot.L0Learn <- function(x, gamma=0, ...)
{
		j = which(abs(x$gamma-gamma)==min(abs(x$gamma-gamma)))
		p = x$p
		allin = c() # contains all the non-zero variables in the path
		for (i in 1:length(x$lambda[[j]])){
		    BetaTemp = x$beta[[j]][,i]
		    supp = which(as.matrix(BetaTemp != 0))
		    allin = c(allin, supp)
		}
		allin = unique(allin)

		#ggplot needs a dataframe
		yy = t(as.matrix(x$beta[[j]][allin,])) # length(lambda) x length(allin) matrix
		data <- as.data.frame(yy)

		colnames(data)  = x$varnames[allin]

		#id variable for position in matrix
		data$id <- x$suppSize[[j]]

		#reshape to long format
		plot_data <- melt(data,id.var="id")

		#breaks = x$suppSize[[j]]

		#plot
		ggplot(plot_data, aes_string(x="id",y="value",group="variable",colour="variable")) + geom_point() + geom_line(aes_string(lty="variable")) +
		labs(x = "Support Size", y = "Coefficient") + theme(axis.title=element_text(size=14)) # + scale_x_continuous(breaks = breaks) + theme(axis.text = element_text(size = 12))
}

#' @title Plot Cross-validation Errors
#'
#' @description Plots cross-validation errors for a given gamma.
#' @param x The output of L0Learn.cvfit
#' @inheritParams plot.L0Learn
#' @examples
#' # Generate synthetic data for this example
#' data <- GenSynthetic(n=500,p=1000,k=10,seed=1)
#' X = data$X
#' y = data$y
#'
#' # Perform 5-fold cross-validation on an L0L2 Model with 5 values of
#' # Gamma ranging from 0.0001 to 10
#' fit <- L0Learn.cvfit(X, y, nFolds=5, seed=1, penalty="L0L2",
#' maxSuppSize=20, nGamma=5, gammaMin=0.0001, gammaMax = 10)
#' # Plot the graph of cross-validation error versus lambda for gamma = 0.0001
#' plot(fit, gamma=0.0001)
#'
#' @method plot L0LearnCV
#' @export
plot.L0LearnCV <- function(x, gamma=0, ...)
{
		j = which(abs(x$fit$gamma-gamma)==min(abs(x$fit$gamma-gamma)))
		data = data.frame(x=x$fit$suppSize[[j]], y=x$cvMeans[[j]], sd=x$cvSDs[[j]])
		ggplot(data, aes_string(x="x",y="y")) + geom_point() + geom_errorbar(aes_string(ymin="y-sd", ymax="y+sd"))+
		labs(x = "Support Size", y = "Cross-validation Error") + theme(axis.title=element_text(size=14)) + theme(axis.text = element_text(size = 12))
}
