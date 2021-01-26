#ifndef RINTERFACE_H
#define RINTERFACE_H

#include <vector>
#include <string>
#include "RcppArmadillo.h"
#include "Grid.h"
#include "GridParams.h"
#include "FitResult.h"


inline void to_arma_error() {
    Rcpp::stop("L0Learn.fit only supports sparse matricies (dgCMatrix), 2D arrays (Dense Matricies)");
}

#endif // RINTERFACE_H
