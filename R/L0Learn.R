#' @docType package
#' @name L0Learn-package
#' @title A package for L0-regularized learning
#'
#' @description L0Learn fits regularization paths for L0-regularized least squares problems. Specifically,
#' it can solve either one of the following problems:
#' \deqn{\min_{\beta} \frac{1}{2} || y - X \beta ||^2 + \lambda ||\beta||_0 \quad \quad (L0)}
#' \deqn{\min_{\beta} \frac{1}{2} || y - X \beta ||^2 + \lambda ||\beta||_0 + \gamma||\beta||_1 \quad (L0L1)}
#' \deqn{\min_{\beta} \frac{1}{2} || y - X \beta ||^2 + \lambda ||\beta||_0 + \gamma||\beta||_2^2  \quad (L0L2)}
#' over a grid of \eqn{\lambda} and \eqn{\gamma} values. Optimization can be done using cyclic coordinate descent (CD)
#' or local combinatorial search. The core of the toolkit is implemented in C++ and employs many computational
#' tricks and heuristics, leading to very competitive running times. CD runs very fast and typically leads to relatively
#' good solutions. Local combinatorial search leads to higher quality solutions (at the expense of increased running times).
#' The toolkit has the following six main methods:
#' \itemize{
#' \item{\code{\link{L0Learn.fit}}: }{Fits an L0-regularized model.}
#' \item{\code{\link{L0Learn.cvfit}}: }{Performs k-fold cross-validation.}
#' \item{\code{\link[=print.L0Learn]{print}}: }{Prints a summary of the path.}
#' \item{\code{\link[=coef.L0Learn]{coef}}: }{Extracts solutions(s) from the path.}
#' \item{\code{\link[=predict.L0Learn]{predict}}: }{Predicts response using a solution in the path.}
#' \item{\code{\link[=plot.L0Learn]{plot}}: }{Plots the regularization path or cross-validation error.}
#' }
NULL
