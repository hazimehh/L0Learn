#include "Normalize.hpp"

std::tuple<beta_vector, double> DeNormalize(beta_vector & B_scaled, 
                                             arma::vec & BetaMultiplier, 
                                             arma::vec & meanX, double meany) {
    beta_vector B_unscaled = B_scaled % BetaMultiplier;
    double intercept = meany - arma::dot(B_unscaled, meanX);
    // Matrix Type, Intercept
    // Dense,            True -> meanX = colMeans(X)
    // Dense,            False -> meanX = 0 Vector (meany = 0)
    // Sparse,            True -> meanX = 0 Vector
    // Sparse,            False -> meanX = 0 Vector
    return std::make_tuple(B_unscaled, intercept);
}
