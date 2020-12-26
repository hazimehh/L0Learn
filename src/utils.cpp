#include "utils.h"

beta_vector clamp_by_vector(beta_vector B, const arma::vec& lows, const arma::vec& highs){
    std::size_t n = B.n_rows;
    for (std::size_t i = 0; i < n; i++){
        B.at(i) = clamp(B.at(i), lows.at(i), highs.at(i));
    }
    return B;
}

arma::sp_mat clamp_by_vector(arma::sp_mat B, const arma::vec lows, const arma::vec highs){
    // See above implementation without filter for error.
    auto begin = B.begin();
    auto end = B.end();
    
    std::vector<std::size_t> inds;
    for (; begin != end; ++begin)
        inds.push_back(begin.row());

    auto n = B.size();
    inds.erase(std::remove_if(inds.begin(),
                              inds.end(),
                              [n](size_t x){return (x > n) && (x < 0);}),
                              inds.end());
    for (auto& it : inds) { 
        double B_item = B(it, 0);
        double low = lows(it);
        double high = highs(it);
        B(it, 0) = clamp(B_item, low, high);
    }
    
    return B;
}

arma::rowvec matrix_normalize(arma::sp_mat &mat_norm){
    auto p = mat_norm.n_cols;
    arma::rowvec scaleX = arma::zeros<arma::rowvec>(p); // will contain the l2norm of every col
    
    for (auto col = 0; col < p; col++){
        double l2norm = arma::norm(matrix_column_get(mat_norm, col), 2);
        scaleX(col) = l2norm;
    }
    
    scaleX.replace(0, -1);
    
    for (auto col = 0; col < p; col++){
        arma::sp_mat::col_iterator begin = mat_norm.begin_col(col);
        arma::sp_mat::col_iterator end = mat_norm.end_col(col);
        for (; begin != end; ++begin)
            (*begin) = (*begin)/scaleX(col);
    }
    
    if (mat_norm.has_nan())
        mat_norm.replace(arma::datum::nan, 0);  // can handle numerical instabilities.
    
    return scaleX;
}

arma::rowvec matrix_normalize(arma::mat& mat_norm){

    auto p = mat_norm.n_cols;
    arma::rowvec scaleX = arma::zeros<arma::rowvec>(p); // will contain the l2norm of every col
    
    for (auto col = 0; col < p; col++) {
        double l2norm = arma::norm(matrix_column_get(mat_norm, col), 2);
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
