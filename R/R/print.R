#' @title Print L0Learn.fit object
#'
#' @description Prints a summary of L0Learn.fit
#' @param x The output of L0Learn.fit or L0Learn.cvfit
#' @param ... ignore
#' @method print L0Learn
#' @export
print.L0Learn <- function(x, ...) {
  gammas <- rep(x$gamma, times = lapply(x$lambda, length))
  data.frame(
    lambda = unlist(x["lambda"]),
    gamma = gammas,
    suppSize = unlist(x["suppSize"]),
    row.names = NULL
  )
}

#' @rdname print.L0Learn
#' @method print L0LearnCV
#' @export
print.L0LearnCV <- function(x, ...) {
  print.L0Learn(x$fit)
}
