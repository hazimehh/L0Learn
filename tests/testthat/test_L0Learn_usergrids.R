library("testthat")
library("L0Learn")

tmp <-  L0Learn::GenSynthetic(n=100, p=1000, k=10, seed=1)
X <- tmp[[1]]
y <- tmp[[2]]

test_that("L0Learn L0 grid works", {
  skip_on_cran()
  userLambda = list()
  userLambda[[1]] <- c(10, 1, 0.1, 0.01)
  x1 <- L0Learn.fit(X, y, penalty = "L0",
                    lambdaGrid=userLambda)
  
  expect_equal(length(x1$lambda[[1]]), 4) 
  for (l in c("SquaredError", "SquaredHinge")){
    x1 <- L0Learn.fit(X, sign(y), penalty = "L0", loss=l,
                      lambdaGrid=userLambda)
    expect_equal(length(x1$lambda[[1]]), 4) 
    
    x1 <- L0Learn.fit(X, sign(y), penalty = "L0", loss=l,
                      lambdaGrid=userLambda)
    expect_equal(length(x1$lambda[[1]]), 4) 
    
  }
})

test_that("L0Learn L0 fails on bad userLambda", {
  skip_on_cran()
  userLambda = list()
  userLambda[[1]] <- c(10, 11, 0.1, 0.01)
  f1 <- function(){
    L0Learn.fit(X, y, penalty = "L0",
                lambdaGrid=userLambda)
  }
  expect_error(f1())
  
  for (l in c("SquaredError", "SquaredHinge")){
    f2 <- function(){
      L0Learn.fit(X, sign(y), penalty = "L0", loss=l,
                        lambdaGrid=userLambda)
    }
    expect_error(f2())
    
    f3 <- function(){
      L0Learn.fit(X, sign(y), penalty = "L0", loss=l,
                        lambdaGrid=userLambda)
      }
    expect_error(f3())
    
    
  }
})

test_that("L0Learn L0 grid ignores nGamma ", {
  skip_on_cran()
  userLambda = list()
  userLambda[[1]] <- c(10, 1, 0.1, 0.01)
  x1 <- L0Learn.fit(X, y, penalty = "L0", nGamma = 1,
                    lambdaGrid=userLambda)
  
  expect_equal(length(x1$lambda), 1) 
  expect_equal(length(x1$lambda[[1]]), 4) 
})


test_that("L0Learn L0L1/2 grid works", {
  skip_on_cran()
  userLambda = list()
  userLambda[[1]] <- c(10, 1, 0.1, 0.01)
  userLambda[[2]] <- c(11, 1.1, 0.11, 0.011, 0.0011)
  userLambda[[3]] <- c(12, 1.2, 0.12)
  x1 <- L0Learn.fit(X, y, penalty = "L0L1", 
                    lambdaGrid=userLambda, nGamma=3)
  
  expect_equal(length(x1$lambda), 3) 
  expect_equal(length(x1$lambda[[1]]), 4) 
  expect_equal(length(x1$lambda[[2]]), 5) 
  expect_equal(length(x1$lambda[[3]]), 3) 
  
  x1 <- L0Learn.fit(X, y, penalty = "L0L2", 
                    lambdaGrid=userLambda, nGamma=3)
  
  expect_equal(length(x1$lambda), 3) 
  expect_equal(length(x1$lambda[[1]]), 4) 
  expect_equal(length(x1$lambda[[2]]), 5) 
  expect_equal(length(x1$lambda[[3]]), 3) 
  
  for (l in c("SquaredError", "SquaredHinge")){
    x1 <- L0Learn.fit(X, sign(y), penalty = "L0L1", loss=l,
                      lambdaGrid=userLambda, nGamma=3)
    expect_equal(length(x1$lambda), 3) 
    expect_equal(length(x1$lambda[[1]]), 4) 
    expect_equal(length(x1$lambda[[2]]), 5) 
    expect_equal(length(x1$lambda[[3]]), 3) 
    
    x1 <- L0Learn.fit(X, sign(y), penalty = "L0L2", loss=l,
                      lambdaGrid=userLambda, nGamma=3, maxSuppSize = 1000)
    expect_equal(length(x1$lambda), 3) 
    expect_equal(length(x1$lambda[[1]]), 4) 
    expect_equal(length(x1$lambda[[2]]), 5) 
    expect_equal(length(x1$lambda[[3]]), 3) 
    
  }
  
  
  succeed()
})

test_that("L0Learn L0L1/2 ignores with wrong nGamma in v2.0.0", {
  skip_on_cran()
  # This changed between v1.2.0 and v2.0.0
  userLambda = list()
  userLambda[[1]] <- c(10, 1, 0.1, 0.01)
  userLambda[[2]] <- c(11, 1.1, 0.11, 0.011, 0.0011)
  userLambda[[3]] <- c(12, 1.2, 0.12)
  
  f1 <- function(){
    L0Learn.fit(X, y, penalty = "L0L1", lambdaGrid=userLambda, nGamma=4)
  }
  f2 <- function(){
    L0Learn.fit(X, y, penalty = "L0L2", lambdaGrid=userLambda, nGamma=4)
  }
    
  if (packageVersion("L0Learn") >= '2.0.0'){
    f1()
    f2()
    succeed()
  } else{
    expect_error(f1())
    expect_error(f2())
  }

  for (l in c("SquaredError", "SquaredHinge")){
    f1 <- function(){
      L0Learn.fit(X, sign(y), penalty = "L0L1", loss=l, lambdaGrid=userLambda, nGamma=4)
    }
    f2 <- function(){
      L0Learn.fit(X, sign(y), penalty = "L0L2", loss=l, lambdaGrid=userLambda, nGamma=4)
    }
    
    if (packageVersion("L0Learn") >= '2.0.0'){
      f1()
      f2()
      succeed()
    } else {
      expect_error(f1())
      expect_error(f2())
    }

  }
})

test_that("L0Learn L0L1/2 grid fails with bad userLambda", {
  skip_on_cran()
  userLambda = list()
  userLambda[[1]] <- c(10, 1, 0.1, 0.01)
  userLambda[[2]] <- c(11, 12, 0.11, 0.011, 0.0011)
  userLambda[[3]] <- c(12, 1.2, 0.12)
  
  f1 <- function(){
    L0Learn.fit(X, y, penalty = "L0L1", lambdaGrid=userLambda, nGamma=3)
  }
  expect_error(f1())
  
  f2 <- function(){
    L0Learn.fit(X, y, penalty = "L0L2", lambdaGrid=userLambda, nGamma=3)
  }
  expect_error(f2())
  
  for (l in c("SquaredError", "SquaredHinge")){
    f1 <- function(){
      L0Learn.fit(X, sign(y), penalty = "L0L1", loss=l, lambdaGrid=userLambda, nGamma=3)
    }
    expect_error(f1())
    
    f2 <- function(){
      L0Learn.fit(X, sign(y), penalty = "L0L2", loss=l, lambdaGrid=userLambda, nGamma=3)
    }
    expect_error(f2())
  }
})
  