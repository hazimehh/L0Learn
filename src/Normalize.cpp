#include "Normalize.h"

std::tuple<arma::sp_mat, double> DeNormalize(arma::sp_mat & B_scaled, 
                                             arma::vec & BetaMultiplier, 
                                             arma::vec & meanX, double meany) {
    arma::sp_mat B_unscaled = B_scaled % BetaMultiplier;
    double intercept = meany - arma::dot(B_unscaled, meanX);
    // Matrix Type, Intercept
    // Dense,            True -> meanX = colMeans(X)
    // Dense,            False -> meanX = 0 Vector (meany = 0)
    // Sparse,            True -> meanX = 0 Vector
    // Sparse,            False -> meanX = 0 Vector
    return std::make_tuple(B_unscaled, intercept);
}
