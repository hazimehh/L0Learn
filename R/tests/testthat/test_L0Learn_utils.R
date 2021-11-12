library("Matrix")
library("testthat")
library("L0Learn")

tmp <-  L0Learn::GenSynthetic(n=1000, p=500, k=10, seed=1, rho=1)
X <- tmp[[1]] 
y <- tmp[[2]] + 1
tol = 1e-4

if (sum(apply(X, 2, sd) == 0)) {
    stop("X needs to have non-zero std for each column")
}

X_sparse <- as(X, "dgCMatrix")

test_that('matrix_column_get dense', {
    skip_on_cran()
    x1 <- as.matrix(X[, 1])
    x2 <- .Call('_L0Learn_R_matrix_column_get_dense', X, 0) # C++ and R use different indexes
    expect_equal(x1, x2)
})

test_that('matrix_column_get sparse', {
    skip_on_cran()
    x1 <- as.matrix(X_sparse[, 1])
    x2 <- .Call('_L0Learn_R_matrix_column_get_sparse', X_sparse, 0) # C++ and R use different indexes
    expect_equal(x1, x2)
})

test_that('matrix_rows_get dense', {
    skip_on_cran()
    x1 <- X[1:4, ]
    x2 <- .Call("_L0Learn_R_matrix_rows_get_dense", X, 0:3)
    expect_equal(x1, x2)
})

test_that('matrix_rows_get sparse', {
    skip_on_cran()
    x1 <- X_sparse[1:4, ]
    x2 <- .Call("_L0Learn_R_matrix_rows_get_sparse", X_sparse, 0:3)
    expect_equal(x1, x2)
})

test_that('matrix_vector_schur_produce dense', {
    skip_on_cran()
    x1 <- X*as.vector(y)
    x2 <- .Call('_L0Learn_R_matrix_vector_schur_product_dense', X, y)
    expect_equal(x1, x2)
})

test_that('matrix_vector_schur_produce sparse', {
    skip_on_cran()
    x1 <- X_sparse*as.vector(y)
    x2 <- .Call('_L0Learn_R_matrix_vector_schur_product_sparse', X_sparse, y)
    expect_equal(x1, x2)
})


test_that('matrix_vector_divide dense', {
    skip_on_cran()
    x1 <- X/as.vector(y)
    x2 <- .Call('_L0Learn_R_matrix_vector_divide_dense', X, y)
    expect_equal(x1, x2)
})

test_that('matrix_vector_divide sparse', {
    skip_on_cran()
    x1 <- X_sparse/as.vector(y)
    x2 <- .Call('_L0Learn_R_matrix_vector_divide_sparse', X_sparse, y)
    expect_equal(x1, x2)
})

test_that('matrix_column_sums dense', {
    skip_on_cran()
    x1 <- colSums(X)
    x2 <- as.vector(.Call('_L0Learn_R_matrix_column_sums_dense', X))
    expect_equal(x1, x2)
})

test_that('matrix_column_sums sparse', {
    skip_on_cran()
    x1 <- colSums(X_sparse)
    x2 <- as.vector(.Call('_L0Learn_R_matrix_column_sums_sparse', X_sparse))
    expect_equal(x1, x2)
})

test_that('matrix_column_dot dense', {
    skip_on_cran()
    x1 <- X[,1]%*%y
    x2 <- .Call('_L0Learn_R_matrix_column_dot_dense', X, 0, y)
    expect_equal(as.double(x1), as.double(x2))
})

test_that('matrix_column_dot sparse', {
    skip_on_cran()
    x1 <- X_sparse[,1]%*%y
    x2 <- .Call('_L0Learn_R_matrix_column_dot_sparse', X_sparse, 0, y)
    expect_equal(as.double(x1), as.double(x2))
})

test_that('matrix_column_mult dense', {
    skip_on_cran()
    c = 3.14
    x1 <- X[,1]*c
    x2 <- .Call('_L0Learn_R_matrix_column_mult_dense', X, 0, c)
    expect_equal(as.double(x1), as.double(x2))
})

test_that('matrix_column_mult sparse', {
    skip_on_cran()
    c = 3.14
    x1 <- X_sparse[,1]*c
    x2 <- .Call('_L0Learn_R_matrix_column_mult_sparse', X_sparse, 0, c)
    expect_equal(as.double(x1), as.double(x2))
})

center_colmeans <- function(x) {
    skip_on_cran()
    xcenter = colMeans(x)
    x - rep(xcenter, rep.int(nrow(x), ncol(x)))
}

colNorms <- function(x){
    apply(x, 2, function(x){sqrt(sum(x^2))})
}

test_that('matrix_normalize dense', {
    skip_on_cran()
    for (norm in c(TRUE, FALSE)){
        if (norm){
            X_norm <- center_colmeans(X)
            expect_equal(colMeans(X_norm), 0*colMeans(X_norm))
        } else {
            X_norm <- as.matrix(X)
        }
        X_norm_copy = as.matrix(X_norm)
        expect_equal(X_norm, X_norm_copy)
        
        x1 <- .Call("_L0Learn_R_matrix_normalize_dense", X_norm)
        
        expect_equal(X_norm, X_norm_copy) # R should not modify X_norm
        
        expect_equal(colNorms(X_norm), as.vector(x1$ScaleX))
        expect_equal(X_norm %*% diag(1/colNorms(X_norm)), x1$mat_norm)
    }
})

test_that('matrix_normalize sparse', {
    skip_on_cran()
    X_norm <- as(X, "dgCMatrix")
    X_norm_copy = as(X, "dgCMatrix")
    
    expect_equal(X_norm, X_norm_copy)
    
    x1 <- .Call("_L0Learn_R_matrix_normalize_sparse", X_norm)
    
    expect_equal(X_norm, X_norm_copy) # R should not modify X_norm
    
    expect_equal(colNorms(X_sparse), as.vector(x1$ScaleX))
    expect_equal(as.matrix(X_norm %*% diag(1/colNorms(X_sparse))), as.matrix(x1$mat_norm))
})

test_that('matrix_center dense', {
    skip_on_cran()
    for (intercept in c(TRUE, FALSE)){
        x_norm <- 0*X
        x1 <- .Call("_L0Learn_R_matrix_center_dense", X, x_norm, intercept)
        
        if (intercept){
            expect_equal(as.vector(x1$MeanX), colMeans(X))
            expect_equal(x1$mat_norm, center_colmeans(X))
        } else {
            expect_equal(as.vector(x1$MeanX), 0*colMeans(X))
            expect_equal(x1$mat_norm, X)
        }
    }
})


test_that('matrix_center sparse', {
    skip_on_cran()
    for (intercept in c(TRUE, FALSE)){
        x_norm = 0*X_sparse
        x1 <- .Call("_L0Learn_R_matrix_center_sparse", X_sparse, x_norm, intercept)
            expect_equal(as.vector(x1$MeanX), 0*colMeans(X))
            expect_equal(x1$mat_norm, X_sparse)
    }
})