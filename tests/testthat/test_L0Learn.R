library("Matrix")
library("testthat")
library("L0Learn")

check_similar_fit_solution <- function(x, y, tolerance=1e-5, cv_tolerance=1e-3){
    for (s in c("p", "n", "varnames", "settings", "loss", "penalty", "gamma")){
        expect_identical(x[[s]], y[[s]])
    }
    
    # This is not an ideal test. Somehow, even though I set the seed before each
    # test there is a difference in solution size. Indictating a bug in the Rcpp
    # code.
    
    num_s = min(x[["beta"]][[1]]@Dim[[2]],
                y[["beta"]][[1]]@Dim[[2]])
    
    for (s in c("lambda", "a0", "converged", "suppSize")){
        expect_equal(x[[s]][[1]][1:num_s],
                     y[[s]][[1]][1:num_s],
                     tolerance=tolerance)
    }
    
    expect_equal(as.matrix(x[["beta"]][[1]])[, 1:num_s],
                 as.matrix(y[["beta"]][[1]])[, 1:num_s],
                 tolerance=tolerance)
    
}

expect_equal_cv <- function(x, y, tolerance=1e-5, cv_tolerance=1e-3) {
  for (i in seq_along(x)){
    if (startsWith(names(x)[i], 'cv')){
      expect_equal(x[i], y[i], tolerance=cv_tolerance)
    } else {
      expect_equal(x[i], y[i], tolerance=tolerance)
    }
  }
}


tmp <-  L0Learn::GenSynthetic(n=500, p=1000, k=10, seed=1, rho=2)
X <- tmp[[1]]
y <- tmp[[2]]
tol = 1e-4

if (sum(apply(X, 2, sd) == 0)) {
  stop("X needs to have non-zero std for each column")
}

X_sparse <- as(X, "dgCMatrix")


test_that('L0Learn Accepts Proper Matricies', {
    ignore <- L0Learn.fit(X, y)
    ignore <- L0Learn.cvfit(X, y)
    ignore <- L0Learn.fit(X_sparse, y, intercept = FALSE)
    ignore <- L0Learn.cvfit(X_sparse, y, intercept = FALSE)
    succeed()
})


test_that("L0Learn fit and cvfit are deterministic for Dense", {
    for (f in c(L0Learn.cvfit, L0Learn.fit)){
        set.seed(1)
        x1 <- f(X, y, intercept = FALSE)
        set.seed(1)
        x2 <- f(X, y, intercept = FALSE)
        expect_equal_cv(x1, x2)
    }
})

test_that("L0Learn fit and cvfit are deterministic for Sparse", {
  for (f in c(L0Learn.cvfit, L0Learn.fit)){
    set.seed(1)
    x1 <-f(X_sparse, y, intercept = FALSE)
    set.seed(1)
    x2 <- f(X_sparse, y, intercept = FALSE)
    expect_equal_cv(x1, x2)
  }
})


test_that("L0Learn fit and cvfit find same solution for different matrix representations", {
    for (f in c(L0Learn.fit, L0Learn.cvfit)){
        set.seed(1)
        x1 <- f(X, y, intercept = FALSE, maxSuppSize = 10)
        set.seed(1)
        x2 <- f(X_sparse, y, intercept = FALSE, maxSuppSize = 10)
        expect_equal_cv(x1, x2)
    }
})


test_that("L0Learn fit and cvfit fail with Sparse Matricies and Intercepts", {
    f1 <- function(){
        L0Learn.fit(X_sparse, y, intercept = TRUE)
    }
    f2 <- function(){
      L0Learn.cvfit(X_sparse, y, intercept = TRUE)
    }
    expect_error(f1())
    expect_error(f2())
})


test_that("L0Learn.Fit runs for all penalty for Sparse and Dense Matrices", {
    for (p in c("L0", "L0L2", "L0L1")){
      for (f in c(L0Learn.cvfit, L0Learn.fit)){
        set.seed(1)
        x1 <- f(X, y, penalty = p, intercept = FALSE)
        set.seed(1)
        x2 <- f(X_sparse, y, penalty = p, intercept = FALSE)
        expect_equal_cv(x1, x2)
      }
    }
})



test_that("L0Learn.Fit runs for all Loss for Sparse and Dense Matrices", {
    y_bin = matrix(rbinom(dim(y)[1], 1, 0.5))
    for (l in c("Logistic", "SquaredHinge")){
        set.seed(1)
        x1 <- L0Learn.fit(X, y_bin, loss=l, intercept = FALSE)
        set.seed(1)
        x2 <- L0Learn.fit(X_sparse, y_bin, loss=l, intercept = FALSE)
        expect_equal_cv(x1, x2)
        
        set.seed(1)
        x1 <- L0Learn.cvfit(X, y_bin, loss=l, intercept = FALSE)
        set.seed(1)
        x2 <- L0Learn.cvfit(X_sparse, y_bin, loss=l, intercept = FALSE)
        expect_equal_cv(x1, x2)
    }
    set.seed(1)
    x1 <- L0Learn.fit(X, y, loss='SquaredError', intercept = FALSE)
    set.seed(1)
    x2 <- L0Learn.fit(X_sparse, y, loss='SquaredError', intercept = FALSE)
    expect_equal_cv(x1, x2)
    
    set.seed(1)
    x1 <- L0Learn.cvfit(X, y, loss='SquaredError', intercept = FALSE)
    set.seed(1)
    x2 <- L0Learn.cvfit(X_sparse, y, loss='SquaredError', intercept = FALSE)
    expect_equal_cv(x1, x2)

})


test_that("L0Learn.Fit runs for all algorithm for Sparse and Dense Matrices", {
    for (p in c("L0", "L0L2", "L0L1")){
        set.seed(1)
        x1 <- L0Learn.fit(X, y, penalty=p, algorithm='CDPSI', intercept = FALSE)
        set.seed(1)
        x2 <- L0Learn.fit(X_sparse, y, penalty=p, algorithm='CDPSI', intercept = FALSE)
        expect_equal_cv(x1, x2)
        
        set.seed(1)
        x1 <- L0Learn.cvfit(X, y, penalty=p, algorithm='CDPSI', intercept = FALSE)
        set.seed(1)
        x2 <- L0Learn.cvfit(X_sparse, y, penalty=p, algorithm='CDPSI', intercept = FALSE)
        expect_equal_cv(x1, x2)
    }
})
