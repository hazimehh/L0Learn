#include "utils.h"

arma::rowvec matrix_normalize(const arma::sp_mat &mat, arma::sp_mat &mat_norm){
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
    
    return scaleX;
}

arma::rowvec matrix_normalize(const arma::mat& mat, arma::mat& mat_norm){

    auto p = mat.n_cols;
    arma::rowvec scaleX = arma::zeros<arma::rowvec>(p); // will contain the l2norm of every col
    
    for (auto col = 0; col < p; col++) {
        double l2norm = arma::norm(matrix_column_get(mat, col), 2);
        scaleX(col) = l2norm;
    }
    
    scaleX.replace(0, -1);
    mat_norm.each_row() /= scaleX;
    
    if (mat_norm.has_nan()){
        mat_norm.replace(arma::datum::nan, 0); // can handle numerical instabilities.   
    }
    
    return scaleX;
}

std::tuple<arma::mat, arma::rowvec> matrix_center(const arma::mat& X, bool intercept){
    auto p = X.n_cols;
    arma::rowvec meanX;
    arma::mat X_normalized;
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
                                                     bool intercept){
    auto p = X.n_cols;
    arma::rowvec meanX = arma::zeros<arma::rowvec>(p);
    arma::sp_mat X_normalized = arma::sp_mat(X);
    return std::make_tuple(X_normalized, meanX);
}
