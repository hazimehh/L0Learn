library("Matrix")
library("testthat")
library("L0Learn")
library("pracma")

K = 10

tmp <-  L0Learn::GenSynthetic(n=100, p=1000, k=K, seed=1, rho=.5, snr=+Inf)
X <- tmp[[1]]
y <- tmp[[2]]
tol = 1e-4

if (norm(X%*%tmp$B + tmp$b0 - y) >= 1e-9){
    stop()
}

if(0 %in% y){
    stop()
}


norm_vec <- function(x) {Norm(as.matrix(x), p = Inf)}

test_that('L0Learn recovers coefficients with no error for L0', {
    skip_on_cran()
    fit <- L0Learn.fit(X, y, loss="SquaredError", penalty = "L0")
    
    for (j in 1:length(fit$suppSize[[1]])){
        # With only L0 penalty, therefore, once the support size is 10, all coefficients should be 1.
        if (fit$suppSize[[1]][[j]] >= 10){
            expect_equal(norm_vec(fit$beta[[1]][,j] - tmp$B), 0, tolerance=1e-3, info=j)
        }
    }
})

test_that('L0Learn seperates data with no error for L0', {
    skip_on_cran()
    for (l in c("Logisitic", "SquaredHinge")){
        fit <- L0Learn.fit(X, sign(y), loss="Logistic", penalty = "L0")
        
        predict_ <- function(index){
            sign(X %*% fit$beta[[1]][,index] + fit$a0[[1]][index])
        }
        
        for (j in 1:length(fit$suppSize[[1]])){
            if (fit$suppSize[[1]][[j]] >= 10){
                expect_equal(predict_(j), sign(y))
            }
        }
    }
})
    

test_that('L0Learn recovers coefficients with no error for L0L1/L0L2', {
    skip_on_cran()
    for (p in c("L0L1", "L0L2")){
        fit <- L0Learn.fit(X, y, loss="SquaredError", penalty = p)

        for (i in 1:length(fit$suppSize)){
            past_K_support_error = Inf
            for (j in 1:length(fit$suppSize[[i]])){
                # With L0 and L1/L2 penalty, once the support size is 10 (dictated by L0 and L1 together), the coefficients
                # will most likely not be 1 due the L1/L2 penalty. Therefore, as the L1/L2 penalty decreases, the coefficients
                # should approach 1.
                # Each iteration, the norm should decrease
                if (fit$suppSize[[i]][[j]] >= K){
                    new_K_support_error = norm_vec(fit$beta[[i]][,j] - tmp$B)
                    expect_lte(new_K_support_error, past_K_support_error)
                    new_K_support_error = past_K_support_error   
                }
            }
        }
    }
})


test_that('L0Learn seperates data with no error for L0L1/L0L2', {
    skip_on_cran()
    for (l in c("Logistic", "SquaredHinge")){
        for (p in c("L0L1", "L0L2")){
            fit <- L0Learn.fit(X, sign(y), loss=l, penalty = p)
            
            predict_ <- function(index1, index2){
                sign(X %*% fit$beta[[index1]][,index2] + fit$a0[[index1]][index2])
            }
            
            for (i in 1:length(fit$suppSize)){
                past_K_support_error = Inf
                for (j in 1:length(fit$suppSize[[i]])){
                    # With L0 and L1/L2 penalty, once the support size is 10 (dictated by L0 and L1 together), the coefficients
                    # will most likely not be 1 due the L1/L2 penalty. Therefore, as the L1/L2 penalty decreases, the coefficients
                    # should approach 1.
                    # Each iteration, the norm should decrease
                    if (fit$suppSize[[i]][[j]] >= K){
                        new_K_support_error = Norm(predict_(i, j) - sign(y))
                        expect_lte(new_K_support_error, past_K_support_error)
                        new_K_support_error = past_K_support_error   
                    }
                }
            }
        }
    }
})