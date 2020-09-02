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
                f(m, y, intercept = FALSE, lows=NaN)
            }
            f2 <- function(){
                f(m, y, intercept = FALSE, highs=NaN)
            }
            f3 <- function(){
                f(m, y, intercept = FALSE, lows=1, highs=0)
            }
            f4 <- function(){
                f(m, y, intercept = FALSE, lows=0, highs=0)
            }
            f5 <- function(){
                f(m, y, intercept = FALSE, lows=rep(1, dim(m)[[2]]), highs=0) 
            }
            f6 <- function(){
                f(m, y, intercept = FALSE, lows=rep(0, dim(m)[[2]]), highs=0) 
            }
            f7 <- function(){
                f(m, y, intercept = FALSE, lows=1, highs=rep(1, dim(m)[[2]])) 
            }
            f8 <- function(){
                f(m, y, intercept = FALSE, lows=1, highs=rep(0, dim(m)[[2]])) 
            }
            f9 <- function(){
                f(m, y, intercept = FALSE, lows=1, highs=2) 
            }
            f10 <- function(){
                f(m, y, intercept = FALSE, lows=-2, highs=-1) 
            }
            f11 <- function(){
                f(m, y, intercept = FALSE, lows=c(1, rep(0, dim(m)[[2]]-1)), highs=rep(1, dim(m)[[2]])) 
            }
            expect_error(f1())
            expect_error(f2())
            expect_error(f3())
            expect_error(f4())
            expect_error(f5())
            expect_error(f6())
            expect_error(f7())
            expect_error(f8())
            expect_error(f9())
            expect_error(f10())
            expect_error(f11())
            
        } 
    }
})

test_that("L0Learn fit respect bounds", {
    low = -.1
    high = .2
    for (m in list(X, X_sparse)){
        for (p in c("L0", "L0L1", "L0L2")){
            fit <- L0Learn.fit(m, y, intercept = FALSE, penalty=p, lows=low, highs=high)
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
            fit <- L0Learn.cvfit(m, y, intercept = FALSE, penalty=p, lows=low, highs=high)
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
                fit <- L0Learn.cvfit(m, y_bin, loss=l, intercept = FALSE, penalty=p, lows=low, highs=high)
                expect_gte(min(fit$fit$beta[[1]]), low-epsilon)
                expect_lte(max(fit$fit$beta[[1]]), high+epsilon)
                
                fit <- L0Learn.fit(m, y_bin, loss=l, intercept = FALSE, penalty=p, lows=low, highs=high)
                expect_gte(min(fit$beta[[1]]), low-epsilon)
                expect_lte(max(fit$beta[[1]]), high+epsilon)
            }
        }
    }
})

test_that("L0Learn respects vector bounds", {
    p = dim(X)[[2]]
    bounds = rnorm(p, 0, .5)
    lows = -(bounds^2) - .01
    highs = (bounds^2) + .01
    for (m in list(X, X_sparse)){
        fit <- L0Learn.fit(m, y, intercept = FALSE, lows=lows, highs=highs)
        for (i in 1:ncol(fit$beta[[1]])){
            expect_true(all((lows <= fit$beta[[1]][,i ]) &&  (fit$beta[[1]][,i ]<= highs)))
        }
    }
})

