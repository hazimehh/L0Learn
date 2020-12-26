#include "BetaVector.h"

/*
 * arma::vec implementation
 */

std::vector<std::size_t> nnzIndicies(const arma::vec& B){
    // Returns a vector of the Non Zero Indicies of B
    const arma::ucolvec nnzs_indicies = arma::find(B);
    return arma::conv_to<std::vector<std::size_t>>::from(nnzs_indicies); 
}

std::vector<std::size_t> nnzIndicies(const arma::vec& B, const std::size_t low){
    // Returns a vector of the Non Zero Indicies of a slice of B starting at low
    const arma::vec B_slice = B.subvec(low, B.n_rows-1);
    const arma::ucolvec nnzs_indicies = arma::find(B_slice);
    return arma::conv_to<std::vector<std::size_t>>::from(nnzs_indicies); 
}


std::size_t n_nonzero(const arma::vec& B){
    const arma::vec nnzs = arma::nonzeros(B);
    return nnzs.n_rows;
    
}

bool has_same_support(const arma::vec& B1, const arma::vec& B2){
    if (B1.size() != B2.size()){
        return false;
    }
    std::size_t n = B1.n_rows;
    
    bool same_support = true;
    for (std::size_t i = 0; i < n; i++){
        same_support = same_support && ((B1.at(i) != 0) == (B2.at(i) != 0));
    }
    return same_support;
}

