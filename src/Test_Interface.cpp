#include "Test_Interface.h" 
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
Rcpp::List R_Normalize_dense(const arma::mat& X, const arma::vec& y,
                             arma::mat & X_normalized, arma::vec & y_normalized,  
                             bool Normalizey, bool intercept){

    arma::vec BetaMultiplier;
    arma::vec meanX;
    double meany;
    
    std::tie(BetaMultiplier, meanX, meany) = Normalize(X, y, X_normalized, y_normalized, Normalizey, intercept);
    
    return Rcpp::List::create(Rcpp::Named("X") = X,
                              Rcpp::Named("y") = y,
                              Rcpp::Named("X_norm") = X_normalized,
                              Rcpp::Named("y_norm") = y_normalized,
                              Rcpp::Named("BetaMultiplier") = BetaMultiplier,
                              Rcpp::Named("meanX") = meanX,
                              Rcpp::Named("meany") = meany);
};

// [[Rcpp::export]]
Rcpp::List R_Normalizev_1_2_0_dense(const arma::mat& X, const arma::vec& y,
                             arma::mat & X_normalized, arma::vec & y_normalized,  
                             bool Normalizey, bool intercept){
    
    arma::vec BetaMultiplier;
    arma::vec meanX;
    double meany;
    
    std::tie(BetaMultiplier, meanX, meany) = Normalizev1_2_0(X, y, X_normalized, y_normalized, Normalizey, intercept);
    
    return Rcpp::List::create(Rcpp::Named("X") = X,
                              Rcpp::Named("y") = y,
                              Rcpp::Named("X_norm") = X_normalized,
                              Rcpp::Named("y_norm") = y_normalized,
                              Rcpp::Named("BetaMultiplier") = BetaMultiplier,
                              Rcpp::Named("meanX") = meanX,
                              Rcpp::Named("meany") = meany);
}
