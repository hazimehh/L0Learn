#include "R_L0Learn_Interface.h"

// [[Rcpp::export]]
Rcpp::NumericMatrix cor_matrix(const int p, const double base_cor) {
    Rcpp::NumericMatrix cor(p, p);
    for (int i = 0; i < p; i++){
        for (int j = 0; j < p; j++){
            cor(i, j) = std::pow(base_cor, std::abs(i - j));
        }
    }
    return cor;
}
