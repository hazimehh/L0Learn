/*
#include "Test_Interface.h"
// [[Rcpp::depends(RcppArmadillo)]]


// [[Rcpp::export]]
arma::vec R_matrix_column_get_dense(const arma::mat &mat, int col) {
    return matrix_column_get(mat, col);
}

// [[Rcpp::export]]
arma::vec R_matrix_column_get_sparse(const arma::sp_mat &mat, int col) {
    return matrix_column_get(mat, col);
}

// [[Rcpp::export]]
arma::mat R_matrix_rows_get_dense(const arma::mat &mat, const arma::ucolvec rows){
    return matrix_rows_get(mat, rows);
}
    
// [[Rcpp::export]]
arma::sp_mat R_matrix_rows_get_sparse(const arma::sp_mat &mat, const arma::ucolvec rows){
    return matrix_rows_get(mat, rows);
}

// [[Rcpp::export]]
arma::mat R_matrix_vector_schur_product_dense(const arma::mat &mat, const arma::vec &u){
    return matrix_vector_schur_product(mat, &u);
}

// [[Rcpp::export]]
arma::sp_mat R_matrix_vector_schur_product_sparse(const arma::sp_mat &mat, const arma::vec &u){
    return matrix_vector_schur_product(mat, &u);
}


// [[Rcpp::export]]
arma::mat R_matrix_vector_divide_dense(const arma::mat &mat, const arma::vec &u){
    return matrix_vector_divide(mat, u);
}

// [[Rcpp::export]]
arma::sp_mat R_matrix_vector_divide_sparse(const arma::sp_mat &mat, const arma::vec &u){
    return matrix_vector_divide(mat, u);
}

// [[Rcpp::export]]
arma::rowvec R_matrix_column_sums_dense(const arma::mat &mat){
    return matrix_column_sums(mat);
}

// [[Rcpp::export]]
arma::rowvec R_matrix_column_sums_sparse(const arma::sp_mat &mat){
    return matrix_column_sums(mat);
}


// [[Rcpp::export]]
double R_matrix_column_dot_dense(const arma::mat &mat, int col, const arma::vec u){
    return matrix_column_dot(mat, col, u);
}

// [[Rcpp::export]]
double R_matrix_column_dot_sparse(const arma::sp_mat &mat, int col, const arma::vec u){
    return matrix_column_dot(mat, col, u);
}

// [[Rcpp::export]]
arma::vec R_matrix_column_mult_dense(const arma::mat &mat, int col, double u){
    return matrix_column_mult(mat, col, u);
}

// [[Rcpp::export]]
arma::vec R_matrix_column_mult_sparse(const arma::sp_mat &mat, int col, double u){
    return matrix_column_mult(mat, col, u);
}

// [[Rcpp::export]]
Rcpp::List R_matrix_normalize_dense(arma::mat mat_norm){
    arma::rowvec ScaleX = matrix_normalize(mat_norm);
    return Rcpp::List::create(Rcpp::Named("mat_norm") = mat_norm,
                              Rcpp::Named("ScaleX") = ScaleX);
};

// [[Rcpp::export]]
Rcpp::List R_matrix_normalize_sparse(arma::sp_mat mat_norm){
    arma::rowvec ScaleX = matrix_normalize(mat_norm);
    return Rcpp::List::create(Rcpp::Named("mat_norm") = mat_norm,
                              Rcpp::Named("ScaleX") = ScaleX);
};

// [[Rcpp::export]]
Rcpp::List R_matrix_center_dense(const arma::mat mat, arma::mat X_normalized, bool intercept){
    arma::rowvec meanX = matrix_center(mat, X_normalized, intercept);
    return Rcpp::List::create(Rcpp::Named("mat_norm") = X_normalized,
                              Rcpp::Named("MeanX") = meanX);
};

// [[Rcpp::export]]
Rcpp::List R_matrix_center_sparse(const arma::sp_mat mat, arma::sp_mat X_normalized,bool intercept){
    arma::rowvec meanX = matrix_center(mat, X_normalized, intercept);
    return Rcpp::List::create(Rcpp::Named("mat_norm") = X_normalized,
                              Rcpp::Named("MeanX") = meanX);
};*/
