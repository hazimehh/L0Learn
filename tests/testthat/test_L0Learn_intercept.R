library("Matrix")
library("testthat")
library("L0Learn")
source("utils.R")

quad <- function(n, p, k=10, thr=.9){
    means = runif(p)
    X = matrix(runif(n*p),nrow=n,ncol=p)
    m = matrix(runif(n*p),nrow=n,ncol=p) <= thr
    X[m] <- 0.0
    B = c(rep(1,k),rep(0,p-k))
    e = rnorm(n)/100
    y = ((X - means)**2)%*%B + e
    list(X=X, y = y)
}
tmp <- quad(500, 4, k=2,thr=.9)
X <- tmp[[1]]
Y <- tmp[[2]]

if (sum(apply(X, 2, sd) == 0)) {
  stop("X needs to have non-zero std for each column")
}

X_sparse <- as(X, "dgCMatrix")


fit_sparse_no_intercept = L0Learn.fit(X_sparse, y, intercept = FALSE)
fit_sparse_intercept = L0Learn.fit(X_sparse, y, intercept = TRUE)
fit_dense_intercept = L0Learn.fit(X, y, intercept = TRUE)
fit_dense_no_intercept = L0Learn.fit(X, y, intercept = FALSE)