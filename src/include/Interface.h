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

// template <typename T>
// void sparse_intercept_check(T m, bool Intercept);
// 
// 
// template <typename T>
// Rcpp::List _L0LearnFit(const T& X, const arma::vec& y, const std::string Loss, const std::string Penalty,
//                        const std::string Algorithm, const std::size_t NnzStopNum, const std::size_t G_ncols,
//                        const std::size_t G_nrows, const double Lambda2Max, const double Lambda2Min,
//                        const bool PartialSort, const std::size_t MaxIters, const double Tol, const bool ActiveSet,
//                        const std::size_t ActiveSetNum, const std::size_t MaxNumSwaps, const double ScaleDownFactor,
//                        std::size_t ScreenSize, const bool LambdaU, const std::vector< std::vector<double> > Lambdas,
//                        const std::size_t ExcludeFirstK, const bool Intercept, const arma::vec &Lows, const arma::vec &Highs);
// 
// template <typename T>
// Rcpp::List _L0LearnCV(const T& X, const arma::vec& y, const std::string Loss, const std::string Penalty,
//                       const std::string Algorithm, const std::size_t NnzStopNum, const std::size_t G_ncols,
//                       const std::size_t G_nrows, const double Lambda2Max, const double Lambda2Min,
//                       const bool PartialSort, const std::size_t MaxIters, const double Tol, const bool ActiveSet,
//                       const std::size_t ActiveSetNum, const std::size_t MaxNumSwaps, const double ScaleDownFactor,
//                       std::size_t ScreenSize, const bool LambdaU, const std::vector< std::vector<double> > Lambdas,
//                       const std::size_t nfolds, const double seed, const std::size_t ExcludeFirstK,
//                       const bool Intercept, const arma::vec &Lows, const arma::vec &Highs);

#endif // RINTERFACE_H
