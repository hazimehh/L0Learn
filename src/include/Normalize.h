#ifndef NORMALIZE_H
#define NORMALIZE_H
#include <tuple>
//#include <armadillo>
#include "RcppArmadillo.h"

std::tuple<arma::vec, arma::vec, double>  Normalize(const arma::mat& X, 
                                                    const arma::vec& y, 
                                                    arma::mat & X_normalized,
                                                    arma::vec & y_normalized,
                                                    bool Normalizey, 
                                                    bool intercept);

std::tuple<arma::vec, arma::vec, double>  Normalize(const arma::sp_mat& X, 
                                                    const arma::vec& y, 
                                                    arma::sp_mat& X_normalized,
                                                    arma::vec & y_normalized,
                                                    bool Normalizey, 
                                                    bool intercept);


std::tuple<arma::sp_mat, double> DeNormalize(arma::sp_mat & B_scaled,
                                             arma::vec & BetaMultiplier,
                                             arma::vec & meanX, double meany);

#endif // NORMALIZE_H
