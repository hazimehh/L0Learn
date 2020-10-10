library("Matrix")
library("testthat")
library("L0Learn")
library("tilting")

tmp <-  L0Learn::GenSynthetic(n=500, p=1000, k=10, seed=1)
X <- tmp[[1]]
y <- tmp[[2]]
tol = 1e-4

if (sum(apply(X, 2, sd) == 0)) {
    stop("X needs to have non-zero std for each column")
}

test_that('Normalize matches Normalizev_1_2_0', {
    for (Normalizey in c(TRUE, FALSE)){
        X_norm = 0*X
        y_norm = 0*y
        x1 <- .Call("_L0Learn_R_Normalize_dense", X, y, X_norm, y_norm, Normalizey, TRUE)
        x2 <- .Call("_L0Learn_R_Normalizev_1_2_0_dense", X, y, X_norm, y_norm, Normalizey, TRUE)
        expect_equal(x1, x2)
        
        X_norm = 0*X
        y_norm = 0*y
        x1 <- .Call("_L0Learn_R_Normalize_dense", X, y, X_norm, y_norm, Normalizey, FALSE)
        x2 <- .Call("_L0Learn_R_Normalizev_1_2_0_dense", X, y, X_norm, y_norm, Normalizey, FALSE)
        expect_equal(x1$X, x2$X)
        expect_equal(x1$y, x2$y)
        expect_equal(x1$y_norm, x2$y_norm)
        expect_equal(x1$meanX, x2$meanX)
        expect_equal(x1$meany, x2$meany)
        
        expect_failure(expect_equal(x1$X_norm, x2$X_norm))
    }
}) 