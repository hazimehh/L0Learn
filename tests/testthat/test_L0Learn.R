library("Matrix")
library("testthat")
library("L0Learn")
source("utils.R")

tmp <-  L0Learn::GenSynthetic(n=100, p=1000, k=10, seed=1, rho=1.5)
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

test_that("L0Learn fails on CDPSI and SquaredHinge", {
   f1 <- function(){
     L0Learn.fit(X, sign(y), algorithm = "CDPSI", loss = "SquaredHinge")
   }
   
   f1 <- function(){
     L0Learn.cvfit(X, sign(y), algorithm = "CDPSI", loss = "SquaredHinge")
   }
   
   expect_failure(f1())
   expect_failure(f2())
   
   f2 <- function(){
     L0Learn.fit(X, sign(y), algorithm = "CD", loss = "SquaredHinge")
     L0Learn.cvfit(X, sign(y), algorithm = "CD", loss = "SquaredHinge")
   }
   
   f2()
   succeed()
   
   f3 <- function(){
     L0Learn.fit(X, sign(y), algorithm = "CDPSI", loss = "Logistic")
     L0Learn.cvfit(X, sign(y), algorithm = "CDPSI", loss = "Logistic")
   }
   f3()
   succeed()
})

test_that("L0Learn respects excludeFirstK for large L0", {
  BIGuserLambda = list()
  BIGuserLambda[[1]] <- c(10)
  for (k in c(0, 1, 10)){
    x1 <- L0Learn.fit(X, y, penalty = "L0", autoLambda=FALSE,
                      lambdaGrid=BIGuserLambda, excludeFirstK = k)

    expect_equal(x1$suppSize[[1]][1], k)
  }
})

test_that("L0Learn excludeFirstK is still subject to L1 norms", {
  K = p =  10
  n = 100

  tmp <-  L0Learn::GenSynthetic(n=n, p=p, k=5, seed=1)
  X_real <- tmp[[1]]

  tmp <-  L0Learn::GenSynthetic(n=n, p=p, k=5, seed=2)
  y_fake <- tmp[[2]]

  # X_real has little to do with generation of y_fake.
  # Therefore, as L1 grows we can expect that the columns go to 0.


  x1 <- L0Learn.fit(X_real, y_fake, penalty = "L0", excludeFirstK = K, maxSuppSize = p)

  expect_equal(length(x1$suppSize[[1]]), 1)
  expect_equal(x1$suppSize[[1]][1], 10)

  # TODO: Fix Crash when excludeFirstK >= p
  # x2 <- L0Learn.fit(X_real, y_fake, penalty = "L0L1", excludeFirstK = K, maxSuppSize = 10)

  # TODO: Fix issue when support is not maximized in first iteration for
  # x2 <- L0Learn.fit(X_real, y_fake, penalty = "L0L1", excludeFirstK = K-1, maxSuppSize = 10)
  # All coefficients should only be regularized by L1, the x2$suppSize is strange.


  x2 <- L0Learn.fit(X_real, y_fake, penalty = "L0L1", excludeFirstK = K-1, maxSuppSize = p)
  for (s in x2$suppSize[[1]]){
    expect_lt(s, p)
  }
})


test_that("L0Learn fit are deterministic for Dense fit", {
    for (p in c("L0", "L0L2", "L0L1")){
      set.seed(1)
      x1 <- L0Learn.fit(X, y, penalty=p, intercept = FALSE)
      set.seed(1)
      x2 <- L0Learn.fit(X, y, penalty=p, intercept = FALSE)
      expect_equal(x1, x2, info=p)
    }
})


test_that("L0Learn cvfit are deterministic for Dense cvfit", {
    for (p in c("L0", "L0L2", "L0L1")){
      set.seed(1)
      x1 <- L0Learn.cvfit(X, y, penalty=p, intercept = FALSE)
      set.seed(1)
      x2 <- L0Learn.cvfit(X, y, penalty=p, intercept = FALSE)
      expect_equal_cv(x1, x2, info=p)
    }
})

test_that("L0Learn fit and cvfit are deterministic for Dense fit", {
  for (p in c("L0", "L0L2", "L0L1")){
    set.seed(1)
    x1 <- L0Learn.fit(X_sparse, y, penalty=p, intercept = FALSE)
    set.seed(1)
    x2 <- L0Learn.fit(X_sparse, y, penalty=p, intercept = FALSE)
    expect_equal(x1, x2, info=p)
  }
})

test_that("L0Learn fit and cvfit are deterministic for Dense cvfit", {
  for (p in c("L0", "L0L2", "L0L1")){
    set.seed(1)
    x1 <- L0Learn.cvfit(X_sparse, y, penalty=p, intercept = FALSE)
    set.seed(1)
    x2 <- L0Learn.cvfit(X_sparse, y, penalty=p, intercept = FALSE)
    expect_equal_cv(x1, x2, info=p)
  }
})


test_that("L0Learn fit find same solution for different matrix representations", {
    for (p in c("L0", "L0L2", "L0L1")){
      set.seed(1)
      x1 <- L0Learn.fit(X_sparse, y, penalty=p, intercept = FALSE)
      set.seed(1)
      x2 <- L0Learn.fit(X, y, penalty=p, intercept = FALSE)
      expect_equal(x1, x2, info=p)
    }
})

test_that("L0Learn cvfit find same solution for different matrix representations", {
  for (p in c("L0", "L0L2", "L0L1")){
    set.seed(1)
    x1 <- L0Learn.cvfit(X_sparse, y, penalty=p, intercept = FALSE)
    set.seed(1)
    x2 <- L0Learn.cvfit(X, y, penalty=p, intercept = FALSE)
    expect_equal_cv(x1, x2, info=p)
  }
})


test_that("L0Learn fit and cvfit run with sparse X and intercepts", {
    L0Learn.fit(X_sparse, y, intercept = TRUE)
    L0Learn.cvfit(X_sparse, y, intercept = TRUE)
    succeed()
})


test_that("L0Learn matchs for all penalty for Sparse and Dense Matrices", {
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
        expect_equal(x1, x2, info = paste("fit", l))

        set.seed(1)
        x1 <- L0Learn.cvfit(X, y_bin, loss=l, intercept = FALSE)
        set.seed(1)
        x2 <- L0Learn.cvfit(X_sparse, y_bin, loss=l, intercept = FALSE)
        expect_equal(x1, x2, info = paste("fit", l))
    }
})


test_that("L0Learn.Fit runs for all algorithm for Sparse and Dense Matrices", {
    for (p in c("L0", "L0L2", "L0L1")){
      for (intercept in c(TRUE, FALSE)){
        set.seed(1)
        x1 <- L0Learn.fit(X, y, penalty=p, algorithm='CDPSI', intercept = intercept)
        set.seed(1)
        x2 <- L0Learn.fit(X, y, penalty=p, algorithm='CDPSI', intercept = intercept)
        expect_equal(x1, x2, info = paste(p, intercept))
      }
    }
})
