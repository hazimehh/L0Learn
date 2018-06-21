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
		data.frame(lambda = unlist(x["lambda"]), gamma = gammas, suppsize = unlist(x["suppSize"]), row.names = NULL)
	}
	else{
		data.frame(lambda = unlist(x["lambda"]), suppsize = x["suppSize"], row.names = NULL)
	}
}
