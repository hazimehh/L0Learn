library("Matrix")
library("testthat")
library("L0Learn")

tmp <-  L0Learn::GenSynthetic(n=100, p=1000, k=20, seed=1, rho=.5)
X <- tmp[[1]]
y <- tmp[[2]]
tol = 1e-4

if (sum(apply(X, 2, sd) == 0)) {
  stop("X needs to have non-zero std for each column")
}

X_sparse <- as(X, "dgCMatrix")

test_that('L0Learn Accepts Proper Matricies', {
    skip_on_cran()
    ignore <- L0Learn.fit(X, y)
    ignore <- L0Learn.cvfit(X, y)
    ignore <- L0Learn.fit(X_sparse, y, intercept = FALSE)
    ignore <- L0Learn.cvfit(X_sparse, y, intercept = FALSE)
    succeed()
})

# test_that("L0Learn fails on CDPSI and SquaredHinge", {
#    f1 <- function(){
#      L0Learn.fit(X, sign(y), algorithm = "CDPSI", loss = "SquaredHinge")
#    }
#    
#    f1 <- function(){
#      L0Learn.cvfit(X, sign(y), algorithm = "CDPSI", loss = "SquaredHinge")
#    }
#    
#    expect_failure(f1())
#    expect_failure(f2())
#    
#    f2 <- function(){
#      L0Learn.fit(X, sign(y), algorithm = "CD", loss = "SquaredHinge")
#      L0Learn.cvfit(X, sign(y), algorithm = "CD", loss = "SquaredHinge")
#    }
#    
#    f2()
#    succeed()
#    
#    f3 <- function(){
#      L0Learn.fit(X, sign(y), algorithm = "CDPSI", loss = "Logistic")
#      L0Learn.cvfit(X, sign(y), algorithm = "CDPSI", loss = "Logistic")
#    }
#    f3()
#    succeed()
# })

test_that("L0Learn respects excludeFirstK for large L0", {
  skip_on_cran()
  BIGuserLambda = list()
  BIGuserLambda[[1]] <- c(10)
  for (k in c(0, 1, 10)){
    x1 <- L0Learn.fit(X, y, penalty = "L0", autoLambda=FALSE,
                      lambdaGrid=BIGuserLambda, excludeFirstK = k)

    expect_equal(x1$suppSize[[1]][1], k)
  }
})

test_that("L0Learn excludeFirstK is still subject to L1 norms", {
  skip_on_cran()
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
    skip_on_cran()
    for (p in c("L0", "L0L2", "L0L1")){
      set.seed(1)
      x1 <- L0Learn.fit(X, y, penalty=p, intercept = FALSE)
      set.seed(1)
      x2 <- L0Learn.fit(X, y, penalty=p, intercept = FALSE)
      expect_equal(x1, x2, info=p)
    }
})


test_that("L0Learn cvfit are deterministic for Dense cvfit", {
    skip_on_cran()
    for (p in c("L0", "L0L2", "L0L1")){
      set.seed(1)
      x1 <- L0Learn.cvfit(X, y, penalty=p, intercept = FALSE)
      set.seed(1)
      x2 <- L0Learn.cvfit(X, y, penalty=p, intercept = FALSE)
      expect_equal(x1, x2, info=p)
    }
})

test_that("L0Learn fit and cvfit are deterministic for Dense fit", {
  skip_on_cran()
  for (p in c("L0", "L0L2", "L0L1")){
    set.seed(1)
    x1 <- L0Learn.fit(X_sparse, y, penalty=p, intercept = FALSE)
    set.seed(1)
    x2 <- L0Learn.fit(X_sparse, y, penalty=p, intercept = FALSE)
    expect_equal(x1, x2, info=p)
  }
})

test_that("L0Learn fit and cvfit are deterministic for Dense cvfit", {
  skip_on_cran()
  for (p in c("L0", "L0L2", "L0L1")){
    set.seed(1)
    x1 <- L0Learn.cvfit(X_sparse, y, penalty=p, intercept = FALSE)
    set.seed(1)
    x2 <- L0Learn.cvfit(X_sparse, y, penalty=p, intercept = FALSE)
    expect_equal(x1, x2, info=p)
  }
})


test_that("L0Learn fit find same solution for different matrix representations", {
    skip_on_cran()
    for (p in c("L0", "L0L2", "L0L1")){
        set.seed(1)
        x1 <- L0Learn.fit(X_sparse, y, penalty=p, intercept = FALSE)
        set.seed(1)
        x2 <- L0Learn.fit(X, y, penalty=p, intercept = FALSE)
        expect_equal(x1, x2, info=p)
    }
})

test_that("L0Learn fit find same solution for different matrix representations", {
  skip_on_cran()
  for (p in c("L0", "L0L2", "L0L1")){
    if (p != "L0L2"){ # TODO: Slight difference in results wtih penalty = "L0L2"
      for (lows in (c(-Inf, 0))){
        set.seed(1)
        x1 <- L0Learn.fit(X_sparse, y, penalty=p, intercept = FALSE, lows=lows)
        set.seed(1)
        x2 <- L0Learn.fit(X, y, penalty=p, intercept = FALSE, lows=lows)
        expect_equal(x1, x2, info=p)
      }
    }
  }
})

test_that("L0Learn cvfit find same solution for different matrix representations", {
  skip_on_cran()
  for (p in c("L0", "L0L2", "L0L1")){
    set.seed(1)
    x1 <- L0Learn.cvfit(X_sparse, y, penalty=p, intercept = FALSE)
    set.seed(1)
    x2 <- L0Learn.cvfit(X, y, penalty=p, intercept = FALSE)
    expect_equal(x1, x2, info=p)
  }
})


test_that("L0Learn fit and cvfit run with sparse X and intercepts", {
    skip_on_cran()
    L0Learn.fit(X_sparse, y, intercept = TRUE)
    L0Learn.cvfit(X_sparse, y, intercept = TRUE)
    succeed()
})

test_that("L0Learn fit and cvfit run with sparse X and intercepts and CDPSI", {
  skip_on_cran()
  L0Learn.fit(X_sparse, y, intercept = TRUE, algorithm = "CDPSI", maxSwaps = 2);
  L0Learn.cvfit(X_sparse, y, intercept = TRUE, algorithm = "CDPSI", maxSwaps = 2);
  succeed()
})


test_that("L0Learn matches for all penalty for Sparse and Dense Matrices", {
    skip_on_cran()
    for (p in c("L0", "L0L2", "L0L1")){
      for (f in c(L0Learn.cvfit, L0Learn.fit)){
        set.seed(1)
        x1 <- f(X, y, penalty = p, intercept = FALSE)
        set.seed(1)
        x2 <- f(X_sparse, y, penalty = p, intercept = FALSE)
        expect_equal(x1, x2)
      }
    }
})



test_that("L0Learn.Fit runs for all Loss for Sparse and Dense Matrices", {
    skip_on_cran()
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
    skip_on_cran()
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



test_that('Utilities for processing regression fit and cv objects run', {
  skip_on_cran()
  # Test utils for L0Learn.fit
  fit <- L0Learn.fit(X, y)
  print(fit)
  coef(fit, lambda=0.01);
  coef(fit, lambda=0.01, gamma=0);
  coef(fit);
  plot(fit)
  plot(fit, showlines=FALSE)
  predict(fit, newx=X, lambda=0.01);
  predict(fit, newx=X, lambda=0.01, gamma=0);
  
  # Test utils for L0Learn.cvfit
  fit <- L0Learn.cvfit(X, y)
  print(fit)
  coef(fit, lambda=0.01);
  coef(fit, lambda=0.01, gamma=0);
  coef(fit);
  plot(fit)
  plot(fit, showlines=FALSE)
  predict(fit, newx=X, lambda=0.01);
  predict(fit, newx=X, lambda=0.01, gamma=0);
  succeed()
})

test_that('Utilities for processing logistic fit and cv objects run', {
  skip_on_cran()
  # Test utils for L0Learn.fit
  fit <- L0Learn.fit(X, sign(y), loss="Logistic")
  print(fit)
  coef(fit, lambda=0.01);
  coef(fit, lambda=0.01, gamma=0);
  coef(fit);
  plot(fit)
  plot(fit, showlines=FALSE)
  predict(fit, newx=X, lambda=0.01);
  predict(fit, newx=X, lambda=0.01, gamma=0);
  
  # Test utils for L0Learn.cvfit
  fit <- L0Learn.cvfit(X, sign(y), loss="Logistic")
  print(fit)
  coef(fit, lambda=0.01);
  coef(fit, lambda=0.01, gamma=0);
  coef(fit);
  plot(fit)
  plot(fit, showlines=FALSE)
  predict(fit, newx=X, lambda=0.01);
  predict(fit, newx=X, lambda=0.01, gamma=0);
  succeed()
})

test_that('Utilities for processing non-intercept fit and cv objects run', {
  skip_on_cran()
  # Test utils for L0Learn.fit
  fit <- L0Learn.fit(X, y, intercept=FALSE)
  print(fit)
  coef(fit, lambda=0.01);
  coef(fit, lambda=0.01, gamma=0);
  coef(fit);
  plot(fit)
  plot(fit, showlines=FALSE)
  predict(fit, newx=X, lambda=0.01);
  predict(fit, newx=X, lambda=0.01, gamma=0);
  
  # Test utils for L0Learn.cvfit
  fit <- L0Learn.cvfit(X, y, intercept=FALSE)
  print(fit)
  coef(fit, lambda=0.01);
  coef(fit, lambda=0.01, gamma=0);
  coef(fit);
  plot(fit)
  plot(fit, showlines=FALSE)
  predict(fit, newx=X, lambda=0.01);
  predict(fit, newx=X, lambda=0.01, gamma=0);
  succeed()
})


test_that('The CDPSI algorithm runs for different losses.', {
  skip_on_cran()
  # Test utils for L0Learn.fit
  L0Learn.fit(X, y, algorithm = "CDPSI", loss = "SquaredError", maxSuppSize=5);
  L0Learn.fit(X, sign(y), algorithm = "CDPSI", loss = "Logistic", maxSuppSize=5);
  L0Learn.fit(X, sign(y), algorithm = "CDPSI", loss = "SquaredHinge", maxSuppSize=5);
  succeed()
})

test_that('The fit and cvfit gracefully error on bad rtol.', {
  skip_on_cran()
  
  f1 <- function(){L0Learn.fit(X, y, rtol=1.1);}
  f2 <- function(){L0Learn.fit(X, y, rtol=-.1);}
  f3 <- function(){L0Learn.fit(X, y, atol=-.1);}
  f4 <- function(){L0Learn.cvfit(X, y, rtol=1.1);}
  f5 <- function(){L0Learn.cvfit(X, y, rtol=-.1);}
  f6 <- function(){L0Learn.cvfit(X, y, atol=-.1);}

  expect_error(f1())
  expect_error(f2())
  expect_error(f3())
  expect_error(f4())
  expect_error(f5())
  expect_error(f6())

})

test_that('The fit and cvfit gracefully error on bad loss specifications', {
  skip_on_cran()
  
  f1 <- function(){L0Learn.fit(X, y, loss="NOT A LOSS");}
  f2 <- function(){L0Learn.cvfit(X, y, loss="NOT A LOSS");}
  
  expect_error(f1())
  expect_error(f2())
})

test_that('The fit and cvfit gracefully error on bad penalty specifications', {
  skip_on_cran()
  
  f1 <- function(){L0Learn.fit(X, y, penalty="NOT A PENALTY");}
  f2 <- function(){L0Learn.cvfit(X, y, penalty="NOT A PENALTY");}
  
  expect_error(f1())
  expect_error(f2())
})

test_that('The fit and cvfit gracefully error on bad algorithim specifications', {
  skip_on_cran()
  
  f1 <- function(){L0Learn.fit(X, y, algorithm="NOT A ALGO");}
  f2 <- function(){L0Learn.cvfit(X, y, algorithm="NOT A ALGO");}
  
  expect_error(f1())
  expect_error(f2())
})

test_that('The fit and cvfit gracefully error on non classifcation y when for classicaiton', {
  skip_on_cran()
  
  f1 <- function(){L0Learn.fit(X, y, loss="Logistic");}
  f2 <- function(){L0Learn.fit(X, y, loss="SquaredHinge");}
  f1 <- function(){L0Learn.cvfit(X, y, loss="Logistic");}
  f2 <- function(){L0Learn.cvfit(X, y, loss="SquaredHinge");}
  
  expect_error(f1())
  expect_error(f2())
  expect_error(f3())
  expect_error(f4())
})


test_that('The fit and cvfit gracefully error on L0 classifcation when lambdagrid is the wrong size', {
  skip_on_cran()
  
  lambda_grid <- list()
  lambda_grid[[1]] <- c(10:1)
  lambda_grid[[2]] <- c(10:1)
  f1 <- function(){L0Learn.fit(X, sign(y), loss="Logistic", penalty="L0", lambdaGrid=lambda_grid);}
  f2 <- function(){L0Learn.fit(X, sign(y), loss="SquaredHinge", penalty="L0", lambdaGrid=lambda_grid);}
  f1 <- function(){L0Learn.cvfit(X, sign(y), loss="Logistic", penalty="L0", lambdaGrid=lambda_grid);}
  f2 <- function(){L0Learn.cvfit(X, sign(y), loss="SquaredHinge", penalty="L0", lambdaGrid=lambda_grid);}
  
  expect_error(f1())
  expect_error(f2())
  expect_error(f3())
  expect_error(f4())
})

test_that('The fit and cvfit gracefully error on L0 when lambdagrid is the wrong size', {
  skip_on_cran()
  
  lambda_grid <- list()
  lambda_grid[[1]] <- c(10:1)
  lambda_grid[[2]] <- c(10:1)
  f1 <- function(){L0Learn.fit(X, y, penalty="L0", lambdaGrid=lambda_grid);}
  f2 <- function(){L0Learn.cvfit(X, y, penalty="L0", lambdaGrid=lambda_grid);}
  
  expect_error(f1())
  expect_error(f2())
})

test_that('The fit and cvfit gracefully error on L0 when lambdagrid has not decreasing values', {
  skip_on_cran()
  
  lambda_grid <- list()
  lambda_grid[[1]] <- c(1:10)
  f1 <- function(){L0Learn.fit(X, y, penalty="L0", lambdaGrid=lambda_grid);}
  f2 <- function(){L0Learn.cvfit(X, y, penalty="L0", lambdaGrid=lambda_grid);}
  
  expect_error(f1())
  expect_error(f2())
})

test_that('The fit and cvfit gracefully error on L0LX when lambdagrid has not decreasing values', {
  skip_on_cran()
  
  lambda_grid <- list()
  lambda_grid[[1]] <- c(10:1)
  lambda_grid[[1]] <- c(1:10)
  f1 <- function(){L0Learn.fit(X, y, penalty="L0L1", lambdaGrid=lambda_grid);}
  f2 <- function(){L0Learn.cvfit(X, y, penalty="L0L1", lambdaGrid=lambda_grid);}
  f3 <- function(){L0Learn.fit(X, y, penalty="L0L2", lambdaGrid=lambda_grid);}
  f4 <- function(){L0Learn.cvfit(X, y, penalty="L0L2", lambdaGrid=lambda_grid);}
  
  expect_error(f1())
  expect_error(f2())  
  expect_error(f3())
  expect_error(f4())
})

test_that('The fit and cvfit gracefully error on CDPSI when bounds are supplied', {
  skip_on_cran()
  

  f1 <- function(){L0Learn.fit(X, y, algorithm="CDPSI", lows=0);}
  f2 <- function(){L0Learn.cvfit(X, y, algorithm="CDPSI", lows=0);}
  
  expect_error(f1())
  expect_error(f2())  
})
