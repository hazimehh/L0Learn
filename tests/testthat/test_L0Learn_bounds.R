library("Matrix")
library("testthat")
library("L0Learn")
source("utils.R")

tmp <-  L0Learn::GenSynthetic(n=100, p=1000, k=10, seed=1, rho=1.5)
X <- tmp[[1]]
y <- tmp[[2]]
y_bin <- sign(y)
tol = 1e-4
epsilon = 1e-16

if (sum(apply(X, 2, sd) == 0)) {
    stop("X needs to have non-zero std for each column")
}

X_sparse <- as(X, "dgCMatrix")


test_that('L0Learn Fails on in-proper Bounds', {
    for (f in c(L0Learn.fit, L0Learn.cvfit)){
        for (m in list(X, X_sparse)){
            f1 <- function(){
                f(m, y, intercept = FALSE, low=NaN)
            }
            f2 <- function(){
                f(m, y, intercept = FALSE, high=NaN)
            }
            f3 <- function(){
                f(m, y, intercept = FALSE, low=1, high=0)
            }
            f4 <- function(){
                f(m, y, intercept = FALSE, low=0, high=0)
            }
            expect_error(f1())
            expect_error(f2())
            expect_error(f3())
            expect_error(f4())
        } 
    }
})

test_that("L0Learn fit respect bounds", {
    low = -.1
    high = .2
    for (m in list(X, X_sparse)){
        for (p in c("L0", "L0L1", "L0L2")){
            fit <- L0Learn.fit(m, y, intercept = FALSE, penalty=p, low=low, high=high)
            expect_gte(min(fit$beta[[1]]), low-epsilon)
            expect_lte(max(fit$beta[[1]]), high+epsilon)
        }
    }
})

test_that("L0Learn cvfit respect bounds", {
    low = -.1
    high = .2
    for (m in list(X, X_sparse)){
        for (p in c("L0", "L0L1", "L0L2")){
            fit <- L0Learn.cvfit(m, y, intercept = FALSE, penalty=p, low=low, high=high)
            expect_gte(min(fit$fit$beta[[1]]), low-epsilon)
            expect_lte(max(fit$fit$beta[[1]]), high+epsilon)
        }
    }
})

test_that("L0Learn respects bounds for all Losses", {
    low = -.1
    high = .2
    for (m in list(X, X_sparse)){
        for (p in c("L0", "L0L1", "L0L2")){
            for (l in c("Logistic", "SquaredHinge")){ 
                fit <- L0Learn.cvfit(m, y_bin, loss=l, intercept = FALSE, penalty=p, low=low, high=high)
                expect_gte(min(fit$fit$beta[[1]]), low-epsilon)
                expect_lte(max(fit$fit$beta[[1]]), high+epsilon)
                
                fit <- L0Learn.fit(m, y_bin, loss=l, intercept = FALSE, penalty=p, low=low, high=high)
                expect_gte(min(fit$beta[[1]]), low-epsilon)
                expect_lte(max(fit$beta[[1]]), high+epsilon)
            }
        }
    }
})

