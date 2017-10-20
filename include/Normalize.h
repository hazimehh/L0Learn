#include <tuple>
//#include <armadillo>
#include "RcppArmadillo.h"

std::tuple<arma::vec, arma::vec, double>  Normalize(const arma::mat& X, const arma::vec& y, arma::mat & X_normalized, arma::vec & y_normalized);

std::tuple<arma::sp_mat,double> DeNormalize(arma::sp_mat & B_scaled, arma::vec & BetaMultiplier, arma::vec & meanX, double meany);
