// [[Rcpp::depends(RcppArmadillo)]]
#ifndef R_L0LEARN_INTERFACE_H
#define R_L0LEARN_INTERFACE_H

#include <vector>
#include <string>
#include <tuple>
#include "RcppArmadillo.h"
#include "L0LearnCore.h"
#include <chrono>
#include <thread>

// [[Rcpp::export]]
Rcpp::List L0LearnFit_sparse(const arma::sp_mat& X, const arma::vec& y, const std::string Loss, const std::string Penalty,
                             const std::string Algorithm, const std::size_t NnzStopNum, const std::size_t G_ncols,
                             const std::size_t G_nrows, const double Lambda2Max, const double Lambda2Min,
                             const bool PartialSort, const std::size_t MaxIters, const double rtol,
                             const double atol, const bool ActiveSet, const std::size_t ActiveSetNum,
                             const std::size_t MaxNumSwaps, const double ScaleDownFactor,
                             const std::size_t ScreenSize, const bool LambdaU,
                             const std::vector< std::vector<double> > Lambdas,
                             const std::size_t ExcludeFirstK, const bool Intercept,
                             const bool withBounds, const arma::vec &Lows, const arma::vec &Highs) {
    
    fitmodel l = L0LearnFit(X, y, Loss, Penalty, Algorithm, NnzStopNum, G_ncols, G_nrows, Lambda2Max, Lambda2Min,
                            PartialSort, MaxIters, rtol, atol, ActiveSet, ActiveSetNum, MaxNumSwaps, ScaleDownFactor, ScreenSize, LambdaU,
                            Lambdas, ExcludeFirstK, Intercept, withBounds, Lows, Highs);

    return Rcpp::List::create(Rcpp::Named("lambda") = l.Lambda0,
                              Rcpp::Named("gamma") = l.Lambda12,
                              Rcpp::Named("SuppSize") = l.NnzCount,
                              Rcpp::Named("beta") = l.Beta,
                              Rcpp::Named("a0") = l.Intercept,
                              Rcpp::Named("Converged") = l.Converged);

}


// [[Rcpp::export]]
Rcpp::List L0LearnFit_dense(const arma::mat& X, const arma::vec& y, const std::string Loss, const std::string Penalty,
                            const std::string Algorithm, const std::size_t NnzStopNum, const std::size_t G_ncols,
                            const std::size_t G_nrows, const double Lambda2Max, const double Lambda2Min,
                            const bool PartialSort, const std::size_t MaxIters, const double rtol,
                            const double atol, const bool ActiveSet, const std::size_t ActiveSetNum,
                            const std::size_t MaxNumSwaps, const double ScaleDownFactor,
                            const std::size_t ScreenSize, const bool LambdaU,
                            const std::vector< std::vector<double> > Lambdas,
                            const std::size_t ExcludeFirstK, const bool Intercept,
                            const bool withBounds, const arma::vec &Lows, const arma::vec &Highs) {
    
    fitmodel l = L0LearnFit(X, y, Loss, Penalty, Algorithm, NnzStopNum, G_ncols, G_nrows, Lambda2Max, Lambda2Min,
                            PartialSort, MaxIters, rtol, atol, ActiveSet, ActiveSetNum, MaxNumSwaps, ScaleDownFactor,
                            ScreenSize, LambdaU, Lambdas, ExcludeFirstK, Intercept, withBounds, Lows, Highs);

    return Rcpp::List::create(Rcpp::Named("lambda") = l.Lambda0,
                              Rcpp::Named("gamma") = l.Lambda12,
                              Rcpp::Named("SuppSize") = l.NnzCount,
                              Rcpp::Named("beta") = l.Beta,
                              Rcpp::Named("a0") = l.Intercept,
                              Rcpp::Named("Converged") = l.Converged);
}


// [[Rcpp::export]]
Rcpp::List L0LearnCV_sparse(const arma::sp_mat& X, const arma::vec& y, const std::string Loss, const std::string Penalty,
                            const std::string Algorithm, const std::size_t NnzStopNum, const std::size_t G_ncols,
                            const std::size_t G_nrows, const double Lambda2Max, const double Lambda2Min,
                            const bool PartialSort, const std::size_t MaxIters, const double rtol, const double atol,
                            const bool ActiveSet, const std::size_t ActiveSetNum, const std::size_t MaxNumSwaps,
                            const double ScaleDownFactor, const std::size_t ScreenSize, const bool LambdaU,
                            const std::vector< std::vector<double> > Lambdas, const std::size_t nfolds,
                            const double seed, const std::size_t ExcludeFirstK, const bool Intercept,
                            const bool withBounds, const arma::vec &Lows, const arma::vec &Highs){
    
    cvfitmodel l = L0LearnCV(X, y, Loss, Penalty, Algorithm, NnzStopNum, G_ncols,
                             G_nrows, Lambda2Max, Lambda2Min, PartialSort,
                             MaxIters, rtol, atol, ActiveSet, ActiveSetNum,
                             MaxNumSwaps, ScaleDownFactor, ScreenSize, LambdaU,
                             Lambdas, nfolds, seed, ExcludeFirstK, Intercept,
                             withBounds, Lows, Highs);

    return Rcpp::List::create(Rcpp::Named("lambda") = l.Lambda0,
                              Rcpp::Named("gamma") = l.Lambda12,
                              Rcpp::Named("SuppSize") = l.NnzCount,
                              Rcpp::Named("beta") = l.Beta,
                              Rcpp::Named("a0") = l.Intercept,
                              Rcpp::Named("Converged") = l.Converged,
                              Rcpp::Named("CVMeans") = l.CVMeans,
                              Rcpp::Named("CVSDs") = l.CVSDs);
}

// [[Rcpp::export]]
Rcpp::List L0LearnCV_dense(const arma::mat& X, const arma::vec& y, const std::string Loss, const std::string Penalty,
                           const std::string Algorithm, const std::size_t NnzStopNum, const std::size_t G_ncols,
                           const std::size_t G_nrows, const double Lambda2Max, const double Lambda2Min,
                           const bool PartialSort, const std::size_t MaxIters, const double rtol, const double atol,
                           const bool ActiveSet, const std::size_t ActiveSetNum, const std::size_t MaxNumSwaps,
                           const double ScaleDownFactor, const std::size_t ScreenSize, const bool LambdaU,
                           const std::vector< std::vector<double> > Lambdas, const std::size_t nfolds,
                           const double seed, const std::size_t ExcludeFirstK, const bool Intercept,
                           const bool withBounds, const arma::vec &Lows, const arma::vec &Highs){

    cvfitmodel l = L0LearnCV(X, y, Loss, Penalty, Algorithm, NnzStopNum, G_ncols,
                             G_nrows, Lambda2Max, Lambda2Min, PartialSort,
                             MaxIters, rtol, atol, ActiveSet, ActiveSetNum,
                             MaxNumSwaps, ScaleDownFactor, ScreenSize, LambdaU,
                             Lambdas, nfolds, seed, ExcludeFirstK, Intercept,
                             withBounds, Lows, Highs);

    return Rcpp::List::create(Rcpp::Named("lambda") = l.Lambda0,
                              Rcpp::Named("gamma") = l.Lambda12,
                              Rcpp::Named("SuppSize") = l.NnzCount,
                              Rcpp::Named("beta") = l.Beta,
                              Rcpp::Named("a0") = l.Intercept,
                              Rcpp::Named("Converged") = l.Converged,
                              Rcpp::Named("CVMeans") = l.CVMeans,
                              Rcpp::Named("CVSDs") = l.CVSDs);
}


#endif // R_L0LEARN_INTERFACE_H
