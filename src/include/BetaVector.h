#ifndef BETA_VECTOR_H
#define BETA_VECTOR_H
#include <vector>
#include "RcppArmadillo.h"

/*
 * arma::vec implementation
 */
using beta_vector = arma::vec;

std::vector<std::size_t> nnzIndicies(const arma::vec& B);

std::vector<std::size_t> nnzIndicies(const arma::vec& B, const std::size_t low);

std::size_t n_nonzero(const arma::vec& B);

bool has_same_support(const arma::vec& B1, const arma::vec& B2);


#endif // BETA_VECTOR_H