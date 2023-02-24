#' @title Extract Solutions
#'
#' @description Extracts a specific solution in the regularization path.
#' @param object The output of L0Learn.fit or L0Learn.cvfit
#' @param lambda The value of lambda at which to extract the solution.
#' @param gamma The value of gamma at which to extract the solution.
#' @param supportSize The number of non-zeros each solution extracted will
#' contain. If no solutions have `supportSize` non-zeros, solutions with
#' the closest number will be extracted.
#' @param ... ignore
#' @method coef L0Learn
#' @details
#' If both lambda and gamma are not supplied, then a matrix of coefficients
#' for all the solutions in the regularization path is returned. If lambda is
#' supplied but gamma is not, the smallest value of gamma is used.
#' @examples
#' # Generate synthetic data for this example
#' data <- GenSynthetic(n=500,p=1000,k=10,seed=1)
#' X = data$X
#' y = data$y
#'
#' # Fit an L0L2 Model with 10 values of Gamma ranging from 0.0001 to 10,
#' # using coordinate descent
#' fit <- L0Learn.fit(X, y, penalty="L0L2", maxSuppSize=50,
#'                    nGamma=10, gammaMin=0.0001, gammaMax = 10)
#' print(fit)
#' # Extract the coefficients of the solution at lambda = 0.0361829
#' # and gamma = 0.0001
#' coef(fit, lambda=0.0361829, gamma=0.0001)
#' # Extract the coefficients of all the solutions in the path
#' coef(fit)
#' # Extract the coefficients of the solution at supportSize = 10
#' # and gamma = 0.0001
#' coef(fit, supportSize=10, gamma=0.0001)
#' @export
coef.L0Learn <- function(object,
                         lambda = NULL,
                         gamma = NULL,
                         supportSize = NULL, ...) {
    if (!is.null(supportSize) && !is.null(lambda)) {
        stop("If `supportSize` is provided to `coef` only `gamma` can also be provided.")
    }
    
    
    if (is.null(lambda) && is.null(gamma) && is.null(supportSize)) {
        # If all three are null, return all solutions
        t <- do.call(cbind, object$beta)
        if (object$settings$intercept) {
            intercepts <- unlist(object$a0)
            t <- rbind(intercepts, t)
        }
        return(t)
    }
        
    if (is.null(gamma)) {
        # if lambda is present but gamma is not, use smallest value of gamma
        gamma <- object$gamma[1]
    }
    
    diffGamma <- abs(object$gamma - gamma)
    gammaindex <- which(diffGamma == min(diffGamma))

    indices <- NULL
    if (!is.null(lambda)) {
        diffLambda <- abs(lambda - object$lambda[[gammaindex]])
        indices <- which(diffLambda == min(diffLambda))
    } else if(!is.null(supportSize)) {
        diffSupportSize <- abs(supportSize - object$suppSize[[gammaindex]])
        indices <- which(diffSupportSize == min(diffSupportSize))
    } else {
        indices <- seq_along(object$lambda[[gammaindex]])
    }

    if (object$settings$intercept) {
        t <- rbind(object$a0[[gammaindex]][indices],
                   object$beta[[gammaindex]][, indices, drop = FALSE])
        rownames(t) <- c("Intercept",
                         paste(rep("V", object$p),
                               1:object$p,
                               sep = ""))
    } else {
        t <- object$beta[[gammaindex]][, indices, drop = FALSE]
        rownames(t) <- paste(rep("V", object$p),
                             1:object$p,
                             sep = "")
    }

    t
}
    
    
#' @rdname coef.L0Learn
#' @method coef L0LearnCV
#' @export
coef.L0LearnCV <- function(object, lambda=NULL, gamma=NULL, ...) {
    coef.L0Learn(object$fit, lambda, gamma, ...)
}
    