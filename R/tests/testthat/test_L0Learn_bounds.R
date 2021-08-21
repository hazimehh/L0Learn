library("Matrix")
library("testthat")
library("L0Learn")
library("raster")

tmp <-  L0Learn::GenSynthetic(n=100, p=5000, k=10, seed=1, rho=1.5)
X <- tmp[[1]]
y <- tmp[[2]]
y_bin <- sign(y + rnorm(100))
tol = 1e-4
epsilon = 1e-12

if (sum(apply(X, 2, sd) == 0)) {
    stop("X needs to have non-zero std for each column")
}

X_sparse <- as(X, "dgCMatrix")

#test_that("L0Learn finds the same solution with LOOSE bounds", {
#    x1 <- L0Learn.fit(X, y, lows=-10000, highs=10000)
#    x2 <- L0Learn.fit(X, y)
#    expect_equal(x1, x2)
#})


test_that('L0Learn Fails on in-proper Bounds', {
    skip_on_cran()
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

test_that("L0Learn fit fails on CDPSI with bound", {
    skip_on_cran()
    f1 <- function(){
        L0Learn.fit(X, y, algorithm = "CDPSI", lows=0)
    }
    f2 <- function(){
        L0Learn.fit(X, y, algorithm = "CDPSI", highs=0)
    }
    f3 <- function(){
        L0Learn.fit(X, y, algorithm = "CDPSI", lows=rep(0, 5000), highs=rep(1, 5000))
    }
    expect_error(f1())
    expect_error(f2())
    expect_error(f3())
})

test_that("L0Learn fit respect bounds", {
    skip_on_cran()
    low = -.04
    high = .05
    for (m in list(X, X_sparse)){
        for (p in c("L0", "L0L1", "L0L2")){
            fit <- L0Learn.fit(m, y, intercept = FALSE, penalty=p, lows=low, highs=high)
            for(i in 1:length(fit$beta)){
                expect_gte(min(fit$beta[[i]]), low-epsilon)
                expect_lte(max(fit$beta[[i]]), high+epsilon)   
            }
        }
    }
})

test_that("L0Learn cvfit respect bounds", {
    skip_on_cran()
    low = -.04
    high = .05
    for (m in list(X, X_sparse)){
        for (p in c("L0", "L0L1", "L0L2")){
            fit <- L0Learn.cvfit(m, y, intercept = FALSE, penalty=p, lows=low, highs=high)
            for (i in 1:length(fit$fit$beta)){
                expect_gte(min(fit$fit$beta[[i]]), low-epsilon)
                expect_lte(max(fit$fit$beta[[i]]), high+epsilon)
            }
        }
    }
})

test_that("L0Learn respects bounds for all Losses", {
    skip_on_cran()
    low = -.04
    high = .05
    maxIters = 2
    maxSwaps = 2
    for (a in c("CD")){ #for (a in c("CD", "CDPSI")){
        for (m in list(X, X_sparse)){
            for (p in c("L0", "L0L1", "L0L2")){
                for (l in c("Logistic", "SquaredHinge")){ 
                    fit <- L0Learn.fit(m, y_bin, loss=l, intercept = FALSE,
                                         penalty=p, algorithm = a, lows=low,
                                         highs=high, maxIters = maxIters, maxSwaps = maxSwaps)
                    for (i in 1:length(fit$beta)){
                        expect_gte(min(fit$beta[[i]]), low-epsilon)
                        expect_lte(max(fit$beta[[i]]), high+epsilon)
                    }
                }
                    
                fit <- L0Learn.fit(m, y, loss='SquaredError', intercept = FALSE,
                                     penalty=p, algorithm = a, lows=low, 
                                     highs=high, maxIters = maxIters, maxSwaps = maxSwaps)
                for (i in 1:length(fit$beta)){
                    expect_gte(min(fit$beta[[i]]), low-epsilon)
                    expect_lte(max(fit$beta[[i]]), high+epsilon)
                }
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
            expect_true(all(lows - 1e-9 <= fit$beta[[1]][,i ]))
            expect_true(all(fit$beta[[1]][,i ] <= highs + 1e-9))
        }
    }
})

find <- function(x, inside){
    which(sapply(inside, FUN=function(X) x %in% X), arr.ind = TRUE)
}

test_that("L0Learn with bounds is better than no-bounds", {
    skip_on_cran()
    lows = -.02
    highs = .02
    fit_wb <- L0Learn.fit(X, y, intercept = FALSE, lows=lows, highs=highs)
    fit_nb <- L0Learn.fit(X, y, intercept = FALSE, scaleDownFactor = .8, nLambda = 300)
    
    for (i in 1:length(fit_wb$suppSize[[1]])){
        nnz_wb = fit_wb$suppSize[[1]][i]
        if (nnz_wb > 10){ #Don't look at NoSelectK
            solution_with_same_nnz =  find(nnz_wb, fit_nb$suppSize[[1]])[1]
            if (is.finite(solution_with_same_nnz)){
                # If there is a solution in fit_nb that has the same number of nnz.
                beta_wb = fit_wb$beta[[1]][, i]
                beta_nb = clamp(fit_nb$beta[[1]][, solution_with_same_nnz], lows, highs)
                
                beta_wb = fit_wb$beta[[1]][, i]
                beta_nb = clamp(fit_nb$beta[[1]][, i], lows, highs)
                
                r_wb = y - X %*% beta_wb
                r_nb = y - X %*% beta_nb
                
                expect_gte(norm(r_nb, "2"), norm(r_wb, "2"))
            }
        }
    }
})

# test_that("L0Learn and glmnet find similar solutions", {
#     lows = -0.02
#     highs = 0.02
#     for (i in 1:2){
#         if (i == 1){
#             p = "L0L1"
#             alpha = 1
#         } else{
#             p = "L0L2"
#             alpha = 0
#         }
#         fit_L0 = L0Learn.fit(X, y, penalty = p, autoLambda= FALSE, lambdaGrid = list(c(1e-6)), lows=lows,highs=highs)
#         fit_glmnet = glmnet(X, y, alpha = alpha, lower=lows,upper=highs)
#     }
#         
#     }
#     fit_L0 = L0Learn.fit(X, y, penalty = "L0L1", lows=-.02,highs=.02)
#     fit_glmnet = glmnet(X,y,lower=-.02,upper=.02)
# })
