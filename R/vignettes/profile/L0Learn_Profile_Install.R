#!/usr/bin/env Rscript

library(devtools)

dev_mode()

repo <- 'TNonet/L0Learn'
branch <- 'template'
install_github(repo, ref=branch)
# install_github("hazimehh/L0Learn", force = TRUE)
print('finished installing')

library(L0Learn)
library(Matrix)

set.seed(1) # fix the seed to get a reproducible result
X = matrix(rnorm(500*1000),nrow=500,ncol=1000)
B = c(rep(1,10),rep(0,990))
e = rnorm(500)
y = X%*%B + e

fit <- L0Learn.fit(X, y, penalty="L0", maxSuppSize=20)

print('done')
dev_mode()