//
// Created by Tim Nonet on 7/27/20.
//
#ifndef L0LEARN_UTILS_H
#define L0LEARN_UTILS_H
#include "RcppArmadillo.h"


template <typename T1>
arma::vec inline matrix_column_get(const arma::mat &mat, T1 col){
    return mat.unsafe_col(col);
}

template <typename T1, typename T2>
arma::vec inline matrix_column_get(const arma::sp_mat &mat, T1 col){
    return arma::vec(mat.col(col));
}

template <typename T1, typename T2>
double inline matrix_column_dot(const arma::mat &mat, T1 col, const T2 &u){
    return arma::dot(matrix_column_get(mat, col), u);
}

template <typename T1, typename T2>
double inline matrix_column_dot(const arma::sp_mat &mat, T1 col, const T2 &u){
    return arma::dot(matrix_column_get(mat, col), u);
}

template <typename T1, typename T2>
arma::vec inline matrix_column_mult(const arma::mat &mat, T1 col, const T2 &u){
    return matrix_column_get(mat, col)*u;
}

template <typename T1, typename T2>
arma::vec inline matrix_column_mult(const arma::sp_mat &mat, T1 col, const T2 &u){
    return matrix_column_get(mat, col)*u;
}

#endif //L0LEARN_UTILS_H