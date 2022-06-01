library("Matrix")
library("testthat")
library("L0Learn")

test_that("CDPSI recovers true support when the correlation is high.", {
  n <- 200
  p <- 1000
  k <- 25

  tmp <- L0Learn:::GenSyntheticHighCorr(n, p, k, seed = 1, snr = +Inf, base_cor = .95)

  X <- tmp$X
  y <- tmp$y
  B <- tmp$B

  fitCD <- L0Learn.fit(X, y, penalty = "L0")
  fitSWAPS <- L0Learn.fit(X, y, penalty = "L0", algorithm = "CDPSI")

  k_support_index <- function(l, k) {
    # The closest support size to k
    sort(abs(l$suppSize[[1]] - k), index.return = TRUE)$ix[1]
  }


  fitCD_k <- k_support_index(fitCD, k)
  # Expected to fail
  expect_false(all(which(B != 0, arr.ind = TRUE) == which(fitCD$beta[[1]][, fitCD_k] != 0, arr.ind = TRUE)))

  fitSWAPS_k <- k_support_index(fitSWAPS, k)
  expect_equal(
    which(B != 0, arr.ind = TRUE),
    which(fitSWAPS$beta[[1]][, fitSWAPS_k] != 0, arr.ind = TRUE)
  )
})

test_that("CDPSI Logistic recovers true support when the correlation is high.", {
  n <- 1000
  p <- 500
  k <- 10

  tmp <- L0Learn:::GenSyntheticHighCorr(n, p, k, seed = 1, snr = +Inf, base_cor = .95)

  X <- tmp$X
  y <- sign(tmp$y)
  B <- tmp$B

  fitCD <- L0Learn.fit(X, y, penalty = "L0", loss = "Logistic")
  fitSWAPS <- L0Learn.fit(X, y, penalty = "L0", algorithm = "CDPSI", loss = "Logistic")


  k_support_index <- function(l, k) {
    # The closest support size to k
    sort(abs(l$suppSize[[1]] - k), index.return = TRUE)$ix[1]
  }


  fitCD_k <- k_support_index(fitCD, k)
  # Expected to fail
  if (length(which(fitCD$beta[[1]][, fitCD_k] != 0, arr.ind = TRUE)) != k) {
    expect_false(FALSE)
  } else {
    expect_false(all(which(B != 0, arr.ind = TRUE) == which(fitCD$beta[[1]][, fitCD_k] != 0, arr.ind = TRUE)))
  }

  fitSWAPS_k <- k_support_index(fitSWAPS, k)
  expect_equal(
    which(B != 0, arr.ind = TRUE),
    which(fitSWAPS$beta[[1]][, fitSWAPS_k] != 0, arr.ind = TRUE)
  )
})

test_that("CDPSI SquaredHinge recovers true support when the correlation is high.", {
  n <- 1000
  p <- 500
  k <- 10

  tmp <- L0Learn:::GenSyntheticHighCorr(n, p, k, seed = 1, snr = +Inf, base_cor = .95)

  X <- tmp$X
  y <- sign(tmp$y)
  B <- tmp$B

  fitCD <- L0Learn.fit(X, y, penalty = "L0", loss = "SquaredHinge")
  fitSWAPS <- L0Learn.fit(X, y, penalty = "L0", algorithm = "CDPSI", loss = "SquaredHinge")


  k_support_index <- function(l, k) {
    # The closest support size to k
    sort(abs(l$suppSize[[1]] - k), index.return = TRUE)$ix[1]
  }


  fitCD_k <- k_support_index(fitCD, k)
  # Expected to fail
  if (length(which(fitCD$beta[[1]][, fitCD_k] != 0, arr.ind = TRUE)) != k) {
    expect_false(FALSE)
  } else {
    expect_false(all(which(B != 0, arr.ind = TRUE) == which(fitCD$beta[[1]][, fitCD_k] != 0, arr.ind = TRUE)))
  }

  fitSWAPS_k <- k_support_index(fitSWAPS, k)
  expect_equal(
    which(B != 0, arr.ind = TRUE),
    which(fitSWAPS$beta[[1]][, fitSWAPS_k] != 0, arr.ind = TRUE)
  )
})
