#include <tuple>
//#include <armadillo>
#include "RcppArmadillo.h"

std::tuple<arma::vec, arma::vec, double>  Normalize(const arma::mat& X, const arma::vec& y, arma::mat & X_normalized, arma::vec & y_normalized, bool Normalizey, bool intercept);

std::tuple<arma::sp_mat, double> DeNormalize(arma::sp_mat & B_scaled, arma::vec & BetaMultiplier, arma::vec & meanX, double meany);

std::tuple<arma::vec, arma::vec, double>  Normalizev1_2_0(const arma::mat& X,
                                                          const arma::vec& y, 
                                                          arma::mat & X_normalized,
                                                          arma::vec & y_normalized, 
                                                          bool Normalizey = false, 
                                                          bool intercept = true);