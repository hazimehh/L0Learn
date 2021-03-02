## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(comment = "#>", warning=FALSE, message=FALSE)

## ----echo = FALSE-------------------------------------------------------------
# Thanks to Yihui Xie for providing this code
library(knitr)
hook_output <- knit_hooks$get("output")
knit_hooks$set(output = function(x, options) {
   lines <- options$output.lines
   if (is.null(lines)) {
     return(hook_output(x, options))  # pass to default hook
   }
   x <- unlist(strsplit(x, "\n"))
   more <- "..."
   if (length(lines)==1) {        # first n lines
     if (length(x) > lines) {
       # truncate the output, but add ....
       x <- c(head(x, lines), more)
     }
   } else {
     x <- c(more, x[lines], more)
   }
   # paste these lines together
   x <- paste(c(x, ""), collapse = "\n")
   hook_output(x, options)
 })

## ---- eval=FALSE--------------------------------------------------------------
#  install.packages("L0Learn")

## -----------------------------------------------------------------------------
set.seed(1) # fix the seed to get a reproducible result
X = matrix(rnorm(500*1000),nrow=500,ncol=1000)
B = c(rep(1,10),rep(0,990))
e = rnorm(500)
y = X%*%B + e

## ---- results="hide"----------------------------------------------------------
library(L0Learn)

## -----------------------------------------------------------------------------
fit <- L0Learn.fit(X, y, penalty="L0", maxSuppSize=20)

## -----------------------------------------------------------------------------
print(fit)

## ----output.lines=15----------------------------------------------------------
coef(fit, lambda=0.0325142, gamma=0)

## ----output.lines=15----------------------------------------------------------
predict(fit, newx=X, lambda=0.0325142, gamma=0)

## ---- fig.height = 4.7, fig.width = 7, out.width="90%", dpi=300---------------
plot(fit, gamma=0)

## -----------------------------------------------------------------------------
fit <- L0Learn.fit(X, y, penalty="L0L2", nGamma = 5, gammaMin = 0.0001, gammaMax = 10, maxSuppSize=20)

## ----output.lines=30----------------------------------------------------------
print(fit)

## ----output.lines=15----------------------------------------------------------
coef(fit,lambda=0.0011539, gamma=10)

## ----eval=FALSE---------------------------------------------------------------
#  predict(fit, newx=X, lambda=0.0011539, gamma=10)

## -----------------------------------------------------------------------------
cvfit = L0Learn.cvfit(X, y, nFolds=5, seed=1, penalty="L0L2", nGamma=5, gammaMin=0.0001, gammaMax=0.1, maxSuppSize=50)

## -----------------------------------------------------------------------------
lapply(cvfit$cvMeans, min)

## ---- fig.height = 4.7, fig.width = 7, out.width="90%", dpi=300---------------
plot(cvfit, gamma=cvfit$fit$gamma[4])

## -----------------------------------------------------------------------------
optimalGammaIndex = 4 # index of the optimal gamma identified previously
optimalLambdaIndex = which.min(cvfit$cvMeans[[optimalGammaIndex]])
optimalLambda = cvfit$fit$lambda[[optimalGammaIndex]][optimalLambdaIndex]
optimalLambda

## ----output.lines=15----------------------------------------------------------
coef(cvfit, lambda=optimalLambda, gamma=cvfit$fit$gamma[4])

## -----------------------------------------------------------------------------
set.seed(1) # fix the seed to get a reproducible result
X = matrix(rnorm(500*1000),nrow=500,ncol=1000)
B = c(rep(1,10),rep(0,990))
e = rnorm(500)
y = sign(X%*%B + e)

## ----output.lines=15----------------------------------------------------------
fit = L0Learn.fit(X,y,loss="Logistic")
print(fit)

## ----output.lines=15----------------------------------------------------------
coef(fit, lambda=8.69435, gamma=1e-7)

## ----output.lines=15----------------------------------------------------------
predict(fit, newx=X, lambda=8.69435, gamma=1e-7)

## -----------------------------------------------------------------------------
X_sparse <- as(X, "dgCMatrix")
fit_dense <- L0Learn.fit(X_sparse, y, penalty="L0")

## -----------------------------------------------------------------------------
fit <- L0Learn.fit(X, y, penalty="L0", maxSuppSize=10, excludeFirstK=3)

## ---- fig.height = 4.7, fig.width = 7, out.width="90%", dpi=300---------------
plot(fit, gamma=0)

## -----------------------------------------------------------------------------
L0Learn.fit(X, y, penalty="L0", lows=-0.5)
L0Learn.fit(X, y, penalty="L0", highs=0.5)
L0Learn.fit(X, y, penalty="L0", lows=-0.5, highs=0.5)

low_vector <- c(rep(-0.1, 500), rep(-0.125, 500))
fit <- L0Learn.fit(X, y, penalty="L0", lows=low_vector, highs=0.25)

## -----------------------------------------------------------------------------
print(max(fit$beta[[1]]))
print(min(fit$beta[[1]][1:500, ]))
print(min(fit$beta[[1]][501:1000, ]))

## -----------------------------------------------------------------------------
userLambda <- list()
userLambda[[1]] <- c(1, 1e-1, 1e-2, 1e-3, 1e-4)
fit <- L0Learn.fit(X, y, penalty="L0", lambdaGrid=userLambda, maxSuppSize=1000)

## -----------------------------------------------------------------------------
print(fit)

## -----------------------------------------------------------------------------
userLambda <- list()
userLambda[[1]] <- c(1, 1e-1, 1e-2, 1e-3, 1e-4)
userLambda[[2]] <- c(10, 2, 1, 0.01, 0.002, 0.001, 1e-5) 
userLambda[[3]] <- c(1e-4, 1e-5) 
# userLambda[[i]] must be a vector of positive decreasing reals.
fit <- L0Learn.fit(X, y, penalty="L0L2", lambdaGrid=userLambda, maxSuppSize=1000)

## -----------------------------------------------------------------------------
print(fit)

