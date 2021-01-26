#ifndef BETA_VECTOR_H
#define BETA_VECTOR_H
#include <vector>
#include "RcppArmadillo.h"

/*
 * arma::vec implementation
 */


using beta_vector = arma::vec;
//using beta_vector = arma::sp_mat;

std::vector<std::size_t> nnzIndicies(const arma::vec& B);

std::vector<std::size_t> nnzIndicies(const arma::sp_mat& B);

std::vector<std::size_t> nnzIndicies(const arma::vec& B, const std::size_t low);

std::vector<std::size_t> nnzIndicies(const arma::sp_mat& B, const std::size_t low);

std::size_t n_nonzero(const arma::vec& B);

std::size_t n_nonzero(const arma::sp_mat& B);

bool has_same_support(const arma::vec& B1, const arma::vec& B2);

bool has_same_support(const arma::sp_mat& B1, const arma::sp_mat& B2);


#endif // BETA_VECTOR_H