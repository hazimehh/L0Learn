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

template <typename T1>
arma::vec inline matrix_column_get(const arma::sp_mat &mat, T1 col){
    return arma::vec(mat.col(col));
}

template <typename T1>
arma::mat inline matrix_rows_get(const arma::mat &mat, const T1 vector_of_row_indices){
    return mat.rows(vector_of_row_indices);
}

template <typename T1>
arma::sp_mat inline matrix_rows_get(const arma::sp_mat &mat, const T1 vector_of_row_indices){
    arma::sp_mat row_mat = arma::sp_mat(vector_of_row_indices.n_elem, mat.n_cols);
    for (int i = 0; i < vector_of_row_indices.n_elem; i++){
        int row_index = vector_of_row_indices(i);
        arma::sp_mat::const_row_iterator begin = mat.begin_row(row_index);
        arma::sp_mat::const_row_iterator end = mat.end_row(row_index);
        
        for ( ; begin != end; ++begin){
            row_mat(i, begin.col()) = *begin;
        }
    }
}

template <typename T1>
arma::mat inline matrix_vector_schur_product(const arma::mat &mat, const T1 &y){
    return mat.each_col() % y;
}

template <typename T1>
arma::sp_mat inline matrix_vector_schur_product(const arma::sp_mat &mat, const T1 &y){
    arma::sp_mat Xy = arma::sp_mat(mat);
    arma::sp_mat::iterator begin = Xy.begin();
    arma::sp_mat::iterator end = Xy.end();
    for (; begin != end; ++begin){
        *begin = (*begin)  * (y)(begin.row());
    }
    return Xy;
}

template <typename T1>
std::tuple<arma::sp_mat, arma::vec> matrix_normailize(const arma::sp_mat &mat,
                                                      arma::sp_mat &mat_norm){
    unsigned int p = mat.n_cols;
    mat_norm = arma::sp_mat(mat); // creates a copy of the data.
    arma::rowvec scaleX = arma::zeros<arma::rowvec>(p); // will contain the l2norm of every col
    
    for (int col = 0; col < p; col++ ){
        double l2norm = arma::norm(mat.col(col), 2);
        scaleX(col) = l2norm;
        
        arma::sp_mat::col_iterator begin = mat_norm.begin_col(col);
        arma::sp_mat::col_iterator end = mat_norm.end_col(col);
        for (; begin != end; ++begin){
            (*begin) = (*begin)/l2norm;
        }
        
    }
    if (mat_norm.has_nan()){mat_norm.replace(arma::datum::nan, 0); } // can handle numerical instabilities.
    return std::make_tuple(mat_norm, scaleX);
}

template <typename T1>
std::tuple<arma::mat, arma::vec> matrix_normailize(const arma::mat &mat,
                                                      arma::mat &mat_norm){
    unsigned int n = mat.n_rows;
    arma::rowvec scaleX = std::sqrt(n) * arma::stddev(mat_norm, 1, 0); // contains the l2norm of every col
    scaleX.replace(0, -1);
    mat_norm.each_row() /= scaleX;
    if (mat_norm.has_nan()){
        mat_norm.replace(arma::datum::nan, 0); // can handle numerical instabilities.
    } 
    return std::make_tuple(mat_norm, scaleX);
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