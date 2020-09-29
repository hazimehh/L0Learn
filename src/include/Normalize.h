#ifndef NORMALIZE_H
#define NORMALIZE_H

#include <tuple>
#include "RcppArmadillo.h"
#include "utils.h"

std::tuple<arma::sp_mat, double> DeNormalize(arma::sp_mat & B_scaled, 
                                             arma::vec & BetaMultiplier, 
                                             arma::vec & meanX, double meany);

template <typename T>
std::tuple<T, arma::vec, arma::vec, double, double>  Normalize(const T& X, 
                                                       const arma::vec& y, 
                                                       arma::vec & y_normalized, 
                                                       bool Normalizey, 
                                                       bool intercept) {
    // std::size_t n = X.n_rows;
    // std::size_t p = X.n_cols;
    
    auto martrix_center_return = matrix_center(X, intercept);
    T X_normalized = std::get<0>(martrix_center_return);
    arma::rowvec meanX = std::get<1>(martrix_center_return);
    
    arma::rowvec scaleX = matrix_normalize(X, X_normalized);
    // X_normalized = std::get<0>(matrix_normailize_return);
    // arma::rowvec scaleX = std::get<1>(matrix_normailize_return);
    
    arma::vec BetaMultiplier;
    double meany = 0;
    double scaley;
    if (Normalizey) {
        if (intercept){
            meany = arma::mean(y);
        }
        y_normalized = y - meany;
        
        // TODO: Use l2_norm
        scaley = arma::norm(y_normalized, 2);
        
        // properly handle cases where y is constant
        if (scaley == 0){
            scaley = 1;
        }
        
        y_normalized = y_normalized / scaley;
        BetaMultiplier = scaley / (scaleX.t()); // transpose scale to get a col vec
        // Multiplying the learned Beta by BetaMultiplier gives the optimal Beta on the original scale
    } else {
        y_normalized = y;
        BetaMultiplier = 1 / (scaleX.t()); // transpose scale to get a col vec
        scaley = 1;
    }
    return std::make_tuple(X_normalized, BetaMultiplier, meanX.t(), meany, scaley);
}

#endif // NORMALIZE_H
