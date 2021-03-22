library("Matrix")
library("testthat")
library("L0Learn")
library("pracma")

# quad <- function(n, p, k=10, thr=.9){
#     means = runif(p)
#     X = matrix(runif(n*p),nrow=n,ncol=p)
#     m = matrix(runif(n*p),nrow=n,ncol=p) <= thr
#     X[m] <- 0.0
#     B = c(rep(1,k),rep(0,p-k))
#     e = rnorm(n)/100
#     y = ((X - means)**2)%*%B + e
#     list(X=X, y = y)
# }

tmp <-  L0Learn::GenSynthetic(n=50, p=200, k=10, seed=1, rho=1, b0=0)
Xsmall <- tmp[[1]]
ysmall <- tmp[[2]]
tol = 1e-4
if (sum(apply(Xsmall, 2, sd) == 0)) {
  stop("X needs to have non-zero std for each column")
}

Xsmall_sparse <- as(Xsmall, "dgCMatrix")

userLambda <- list()
userLambda[[1]] <- c(logspace(-1, -10, 100))

test_that("Intercepts are supported for all losses, algorithims, penalites, and matrix types", {
  skip_on_cran()
  # Try all losses
  for (p in c("L0", "L0L1", "L0L2")){
    L0Learn.fit(Xsmall_sparse, ysmall, penalty=p, intercept = TRUE)
    L0Learn.cvfit(Xsmall_sparse, ysmall, penalty=p, nFolds=2, intercept = TRUE)
  }
  
  for (a in c("CD", "CDPSI")){
    L0Learn.fit(Xsmall_sparse, ysmall, algorithm=a, intercept = TRUE)
    L0Learn.cvfit(Xsmall_sparse, ysmall, algorithm=a, nFolds=2, intercept = TRUE)
  }
  
  for (l in c("Logistic", "SquaredHinge")){
    L0Learn.fit(Xsmall_sparse, sign(ysmall), loss=l, intercept = TRUE)
    L0Learn.cvfit(Xsmall_sparse, ysmall, algorithm=a, nFolds=2, intercept = TRUE)
  }
  succeed()
})

test_that("Intercepts for Sparse Matricies are deterministic", {
  skip_on_cran()
  # Try all losses
  for (p in c("L0", "L0L1", "L0L2")){
    set.seed(1)
    x1 <- L0Learn.fit(Xsmall_sparse, ysmall, penalty=p)
    set.seed(1)
    x2 <- L0Learn.fit(Xsmall_sparse, ysmall, penalty=p)
    expect_equal(x1$a0, x2$a0, info=p)
  }
  
  for (a in c("CD", "CDPSI")){
    set.seed(1)
    x1 <- L0Learn.fit(Xsmall_sparse, ysmall, algorithm=a)
    set.seed(1)
    x2 <- L0Learn.fit(Xsmall_sparse, ysmall, algorithm=a)
    expect_equal(x1$a0, x2$a0, info=a)
  }
  
  for (l in c("Logistic", "SquaredHinge")){
    set.seed(1)
    x1 <- L0Learn.fit(Xsmall_sparse, sign(ysmall), loss=l)
    set.seed(1)
    x2 <- L0Learn.fit(Xsmall_sparse, sign(ysmall), loss=l)
    expect_equal(x1$a0, x2$a0, info=l)

  }
})

test_that("Intercepts are passed between Swap iterations", {
  skip_on_cran()
  # TODO : Implement test case
})

tmp <-  L0Learn::GenSynthetic(n=100, p=1000, k=10, seed=1, rho=1.5, b0=0)
X <- tmp[[1]]
y <- tmp[[2]]
tol = 1e-4

if (sum(apply(X, 2, sd) == 0)) {
  stop("X needs to have non-zero std for each column")
}

X_sparse <- as(X, "dgCMatrix")

test_that("When lambda0 is large, intecepts should be found similar for both sparse and dense methods", {
  skip_on_cran()
  BIGuserLambda <- list()
  BIGuserLambda[[1]] <- c(logspace(2, -2, 10))
  
  # TODO: Prevent crash if lambdaGrid is not "acceptable.
  for (a in c("CD", "CDPSI")){
    set.seed(1)
    x1 <- L0Learn.fit(X_sparse, y, penalty="L0", intercept = TRUE, algorithm = a,
                      autoLambda=FALSE, lambdaGrid=BIGuserLambda, maxSuppSize=100)
    set.seed(1)
    x2 <- L0Learn.fit(X, y, penalty="L0", intercept = TRUE, algorithm = a,
                      autoLambda=FALSE, lambdaGrid=BIGuserLambda, maxSuppSize=100)
    
    for (i in 1:length(x1$a0)){
      if ((x1$suppSize[[1]][i] == 0) && (x2$suppSize[[1]][i] == 0)){
        expect_equal(x1$a0[[1]][i], x2$a0[[1]][i])
      } else if (x1$suppSize[[1]][i] == x2$suppSize[[1]][i]){
        expect_equal(x1$a0[[1]][i], x2$a0[[1]][i], tolerance=1e-6, scale=x1$a0[[1]][i])
      }
    }
  }
})

# test_that("Intercepts achieve a lower insample-error", {
#   skip_on_cran()
#   
#   for (a in c("CD", "CDPSI")){ 
#     y_scaled = y*2 + 10
#     set.seed(1)
#     x1 <- L0Learn.fit(X_sparse, y_scaled, penalty="L0", intercept = TRUE,
#                       algorithm = a,
#                       autoLambda=FALSE, lambdaGrid=userLambda, maxSuppSize=100)
#     set.seed(1)
#     x2 <- L0Learn.fit(X_sparse, y_scaled, penalty="L0", intercept = FALSE,
#                       algorithm = a,
#                       autoLambda=FALSE, lambdaGrid=userLambda, maxSuppSize=100)
#     
#     min_length = min(length(x1$a0[[1]]), length(x1$a0[[1]]))
#     for (i in 1:min_length){
#       if (TRUE){ # x1$suppSize[[1]][i] >= x2$suppSize[[1]][i]
#         x1_loss = norm(X%*%x1$beta[[1]][,i] + x1$a0[[1]][i] - y_scaled, '2')
#         x2_loss = norm(X%*%x2$beta[[1]][,i] + x2$a0[[1]][i] - y_scaled, '2')
#         expect_lte(x1_loss, x2_loss)
#       }
#     }
#     
    # logistic <- function(x){1/(1+exp(-x))};
    # logit <- sum(log(logistic))
    # 
    # x1 <- L0Learn.fit(X_sparse, sign(y), penalty="L0", intercept = TRUE,
    #                   algorithm = a,
    #                   loss = "Logistic", autoLambda=FALSE, lambdaGrid=userLambda, 
    #                   maxSuppSize=1000)
    # x2 <- L0Learn.fit(X_sparse, sign(y), penalty="L0", intercept = FALSE, 
    #                   algorithm = a,
    #                   loss = "Logistic", autoLambda=FALSE, lambdaGrid=userLambda, 
    #                   maxSuppSize=1000)
    # 
    # for (i in 1:min_length){
    #   
    #   x1_loss = sum(sign(y)*logistic(X%*%x1$beta[[1]][,i] + x1$a0[[1]][i])) # more 1s
    #   x2_loss = sum(sign(y)*logistic(X%*%x2$beta[[1]][,i] + x2$a0[[1]][i])) # more -1s
    #   print(paste(i, x1_loss - x2_loss))
    #   #expect_lt(x1_loss, x2_loss)
    # }
    # 
    # squaredHinge <- function(y, yhat){max(0, 1-y*yhat)**2}
    # 
    # x1 <- L0Learn.fit(X_sparse, sign(y), penalty="L0", intercept = TRUE,
    #                   algorithm = a,
    #                   loss = "SquaredHinge", autoLambda=FALSE, lambdaGrid=userLambda, 
    #                   maxSuppSize=1000)
    # x2 <- L0Learn.fit(X_sparse, sign(y), penalty="L0", intercept = FALSE, 
    #                   algorithm = a,
    #                   loss = "SquaredHinge", autoLambda=FALSE, lambdaGrid=userLambda, 
    #                   maxSuppSize=1000)
    # 
    # for (i in 1:min_length){
    #   x1_loss = sum(squaredHinge(sign(X%*%x1$beta[[1]][,i] + x1$a0[[1]][i]), sign(y)))
    #   x2_loss = sum(squaredHinge(sign(X%*%x2$beta[[1]][,i] + x2$a0[[1]][i]), sign(y)))
    #   #print(paste(i, x1_loss - x2_loss))
    #   expect_lt(x1_loss, x2_loss)
    # }
# }
# })

test_that("Intercepts are learned close to real values", {
  skip_on_cran()
  fineuserLambda <- list()
  fineuserLambda[[1]] <- c(logspace(-1, -10, 100))
  
  k = 10 
  for (a in c("CD", "CDPSI")){
    for (b0 in c(-100, -10, -2, 2, 10, 100)){
      tmp <-  L0Learn::GenSynthetic(n=500, p=200, k=k, seed=1, rho=1, b0=b0)
      X2 <- tmp[[1]]
      y2 <- tmp[[2]]
      
      tol = 1e-4
      if (sum(apply(X2, 2, sd) == 0)) {
        stop("X needs to have non-zero std for each column")
      }
      X2_sparse <- as(X2, "dgCMatrix") 
      
      x1 <- L0Learn.fit(X2_sparse, y2, penalty="L0", intercept = TRUE, algorithm = a, 
                        autoLambda=FALSE, lambdaGrid=fineuserLambda, maxSuppSize=1000)
      
      x2 <- L0Learn.fit(X2, y2, penalty="L0", intercept = TRUE, algorithm = a, 
                        autoLambda=FALSE, lambdaGrid=fineuserLambda, maxSuppSize=1000)
      
      for (i in 1:length(x1$suppSize[[1]])){
        if (x1$suppSize[[1]][i] ==  k){
          expect_lt(abs(x1$a0[[1]][i] - b0), abs(.01*b0))
        }
      }
      
      for (i in 1:length(x2$suppSize[[1]])){
        if (x2$suppSize[[1]][i] ==  k){
          expect_lt(abs(x2$a0[[1]][i] - b0), abs(.01*b0))
        }
      }
    }
  }
})