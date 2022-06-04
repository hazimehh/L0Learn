library("testthat")
library("L0Learn")


test_that("L0Learn GenSyntheticLogistic fails for improper s", {
  expect_error(L0Learn:::GenSyntheticLogistic(n = 1000, p = 1000, k = 10, seed = 1, s = -1))
})

test_that("L0Learn GenSyntheticLogistic accepts Null and Diagonal Sigma", {
  L0Learn:::GenSyntheticLogistic(n = 1000, p = 1000, k = 10, seed = 1, sigma = NULL)
  L0Learn:::GenSyntheticLogistic(
    n = 1000, p = 1000, k = 10, seed = 1,
    sigma = diag(1:1000)
  )

  expect_error(L0Learn:::GenSyntheticLogistic(
    n = 1000, p = 1000, k = 10, seed = 1,
    sigma = diag(1:999)
  ))

  succeed()
})

test_that("L0Learn GenSyntheticLogistic shuffles B", {
  L0Learn:::GenSyntheticLogistic(n = 1000, p = 1000, k = 10, seed = 1, shuffle_B = TRUE)
  L0Learn:::GenSyntheticLogistic(n = 1000, p = 1000, k = 10, seed = 1, shuffle_B = FALSE)

  succeed()
})
