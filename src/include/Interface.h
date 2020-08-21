#ifndef RINTERFACE_H
#define RINTERFACE_H

#include <vector>
#include <string>
#include "RcppArmadillo.h"
#include "Grid.h"
#include "GridParams.h"
#include "FitResult.h"


inline void to_arma_error() {
    Rcpp::stop("L0Learn.fit only supports sparse matricies (dgCMatrix), 2D arrays, and Dense Matricies (dgeMatrix)");
}

template <typename T>
void sparse_intercept_check(T m, bool Intercept);

template <typename T>
Rcpp::List _L0LearnFit(const T& X, const arma::vec& y, const std::string Loss, const std::string Penalty,
                       const std::string Algorithm, const unsigned int NnzStopNum, const unsigned int G_ncols,
                       const unsigned int G_nrows, const double Lambda2Max, const double Lambda2Min,
                       const bool PartialSort, const unsigned int MaxIters, const double Tol, const bool ActiveSet,
                       const unsigned int ActiveSetNum, const unsigned int MaxNumSwaps, const double ScaleDownFactor,
                       unsigned int ScreenSize, const bool LambdaU, const std::vector< std::vector<double> > Lambdas,
                       const unsigned int ExcludeFirstK, const bool Intercept);

template <typename T>
Rcpp::List _L0LearnCV(const T& X, const arma::vec& y, const std::string Loss, const std::string Penalty,
                      const std::string Algorithm, const unsigned int NnzStopNum, const unsigned int G_ncols,
                      const unsigned int G_nrows, const double Lambda2Max, const double Lambda2Min, const bool PartialSort,
                      const unsigned int MaxIters, const double Tol, const bool ActiveSet, const unsigned int ActiveSetNum,
                      const unsigned int MaxNumSwaps, const double ScaleDownFactor, unsigned int ScreenSize, const bool LambdaU,
                      const std::vector< std::vector<double> > Lambdas, const unsigned int nfolds, const double seed,
                      const unsigned int ExcludeFirstK, const bool Intercept);

#endif // RINTERFACE_H
