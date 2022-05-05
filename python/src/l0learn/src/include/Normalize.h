#ifndef NORMALIZE_H
#define NORMALIZE_H

#include <tuple>
#include <armadillo>
#include "utils.h"
#include "BetaVector.h"

std::tuple<beta_vector, double> DeNormalize(beta_vector & B_scaled, 
                                             arma::vec & BetaMultiplier, 
                                             arma::vec & meanX, double meany);

template <typename T>
std::tuple<arma::vec, arma::vec, double, double>  Normalize(const T& X, 
                                                            const arma::vec& y, 
                                                            T& X_normalized, 
                                                            arma::vec & y_normalized, 
                                                            bool Normalizey, 
                                                            bool intercept) {
    
    arma::rowvec meanX = matrix_center(X, X_normalized, intercept);
    arma::rowvec scaleX = matrix_normalize(X_normalized);
    
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
    return std::make_tuple(BetaMultiplier, meanX.t(), meany, scaley);
}

#endif // NORMALIZE_H
