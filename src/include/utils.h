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
    // Option for CV for random splitting or contigous splitting.
    // 1 - N without permutations splitting at floor(N/n_folds)
    arma::sp_mat row_mat = arma::sp_mat(vector_of_row_indices.n_elem, mat.n_cols);
    
    for (auto i = 0; i < vector_of_row_indices.n_elem; i++){
        auto row_index = vector_of_row_indices(i);
        arma::sp_mat::const_row_iterator begin = mat.begin_row(row_index);
        arma::sp_mat::const_row_iterator end = mat.end_row(row_index);
        
        for (; begin != end; ++begin){
            row_mat(i, begin.col()) = *begin;
        }
    }
    return row_mat;
}

template <typename T1>
arma::mat inline matrix_vector_schur_product(const arma::mat &mat, const T1 &y){
    // return mat[i, j] * y[i] for each j
    return mat.each_col() % *y;
}

template <typename T1>
arma::sp_mat inline matrix_vector_schur_product(const arma::sp_mat &mat, const T1 &y){

    arma::sp_mat Xy = arma::sp_mat(mat);
    arma::sp_mat::iterator begin = Xy.begin();
    arma::sp_mat::iterator end = Xy.end();
    
    auto yp = (*y);
    for (; begin != end; ++begin){
        auto row = begin.row();
        *begin = (*begin)  * yp(row);
    }
    return Xy;
}

std::tuple<arma::sp_mat, arma::rowvec> matrix_normailize(const arma::sp_mat &mat,
                                                         arma::sp_mat &mat_norm){
    auto p = mat.n_cols;
    arma::rowvec scaleX = arma::zeros<arma::rowvec>(p); // will contain the l2norm of every col
    
    for (auto col = 0; col < p; col++){
        double l2norm = arma::norm(mat.col(col), 2);
        scaleX(col) = l2norm;
        
        arma::sp_mat::col_iterator begin = mat_norm.begin_col(col);
        arma::sp_mat::col_iterator end = mat_norm.end_col(col);
        for (; begin != end; ++begin)
            (*begin) = (*begin)/l2norm;
        
    }
    
    if (mat_norm.has_nan())
        mat_norm.replace(arma::datum::nan, 0);  // can handle numerical instabilities.
    
    return std::make_tuple(mat_norm, scaleX);
}

std::tuple<arma::mat, arma::rowvec> matrix_normailize(const arma::mat &mat, arma::mat &mat_norm){
    auto n = mat.n_rows;
    arma::rowvec scaleX = std::sqrt(n) * arma::stddev(mat_norm, 1, 0); // contains the l2norm of every col
    scaleX.replace(0, -1);
    mat_norm.each_row() /= scaleX;
    
    if (mat_norm.has_nan()) 
        mat_norm.replace(arma::datum::nan, 0); // can handle numerical instabilities.
    
    return std::make_tuple(mat_norm, scaleX);
}

std::tuple<arma::mat, arma::rowvec> matrix_center(const arma::mat& X,
                                                  arma::mat& X_normalized,
                                                  bool intercept){
    auto p = X.n_cols;
    arma::rowvec meanX;
    if (intercept){
        meanX = arma::mean(X, 0);
        X_normalized = X.each_row() - meanX;
    } else {
        meanX = arma::zeros<arma::rowvec>(p);
        X_normalized = arma::mat(X);
    }
    
    return std::make_tuple(X_normalized, meanX);
}

std::tuple<arma::sp_mat, arma::rowvec> matrix_center(const arma::sp_mat& X,
                                                     arma::sp_mat& X_normalized,
                                                     bool intercept){
    auto p = X.n_cols;
    arma::rowvec meanX = arma::zeros<arma::rowvec>(p);
    X_normalized = arma::sp_mat(X);
    return std::make_tuple(X_normalized, meanX);
}

template <typename T1>
arma::sp_mat inline matrix_vector_divide(const arma::sp_mat& mat, const T1 &u){
    arma::sp_mat divided_mat = arma::sp_mat(mat);
    
    //auto up = (*u);
    arma::sp_mat::iterator begin = divided_mat.begin();
    arma::sp_mat::iterator end = divided_mat.end();
    for ( ; begin != end; ++begin){
        *begin = (*begin) / u(begin.row());
    }
    return divided_mat;
}

template <typename T1>
arma::mat inline matrix_vector_divide(const arma::mat& mat, const T1 &u){
    return mat.each_col() / u;
}

arma::rowvec inline matrix_column_sums(const arma::mat& mat){
    return arma::sum(mat, 0);
}

arma::rowvec inline matrix_column_sums(const arma::sp_mat& mat){
    return arma::rowvec(arma::sum(mat, 0));
}


template <typename T1, typename T2>
double inline matrix_column_dot(const arma::mat &mat, T1 col, const T2 &u){
    // Rcpp::Rcout << "matrix_column_dot Starts\n"
    // Rcpp::Rcout << "(" << mat.n_rows << "," << mat.n_cols << ") x (" << u.n_elem << ") \n";;
    return arma::dot(matrix_column_get(mat, col), u);
}

template <typename T1, typename T2>
double inline matrix_column_dot(const arma::sp_mat &mat, T1 col, const T2 &u){
    // Rcpp::Rcout << "matrix_column_dot Starts\n";
    // Rcpp::Rcout << "(" << mat.n_rows << "," << mat.n_cols << ") x (" << u.n_elem << ") \n";
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