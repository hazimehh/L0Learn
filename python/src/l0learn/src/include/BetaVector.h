#ifndef BETA_VECTOR_H
#define BETA_VECTOR_H
#include <vector>
#include <armadillo>

/*
 * arma::vec implementation
 */


using beta_vector = arma::vec;
//using beta_vector = arma::sp_mat;

inline std::vector<std::size_t> nnzIndicies(const arma::vec& B){
    // Returns a vector of the Non Zero Indicies of B
    const arma::ucolvec nnzs_indicies = arma::find(B);
    return arma::conv_to<std::vector<std::size_t>>::from(nnzs_indicies);
}

// std::vector<std::size_t> nnzIndicies(const arma::sp_mat& B){
//     // Returns a vector of the Non Zero Indicies of B
//     std::vector<std::size_t> S;
//     arma::sp_mat::const_iterator it;
//     const arma::sp_mat::const_iterator it_end = B.end();
//     for(it = B.begin(); it != it_end; ++it)
//     {
//         S.push_back(it.row());
//     }
//     return S;
// }

inline std::vector<std::size_t> nnzIndicies(const arma::vec& B, const std::size_t low){
    // Returns a vector of the Non Zero Indicies of a slice of B starting at low
    // This is for NoSelectK situations
    const arma::vec B_slice = B.subvec(low, B.n_rows-1);
    const arma::ucolvec nnzs_indicies = arma::find(B_slice);
    return arma::conv_to<std::vector<std::size_t>>::from(nnzs_indicies);
}

// std::vector<std::size_t> nnzIndicies(const arma::sp_mat& B, const std::size_t low){
//     // Returns a vector of the Non Zero Indicies of B
//     std::vector<std::size_t> S;
//
//
//     arma::sp_mat::const_iterator it;
//     const arma::sp_mat::const_iterator it_end = B.end();
//
//
//     for(it = B.begin(); it != it_end; ++it)
//     {
//         if (it.row() >= low){
//             S.push_back(it.row());
//         }
//     }
//     return S;
// }


inline std::size_t n_nonzero(const arma::vec& B){
    const arma::vec nnzs = arma::nonzeros(B);
    return nnzs.n_rows;

}

// std::size_t n_nonzero(const arma::sp_mat& B){
//     return B.n_nonzero;
//
// }

inline bool has_same_support(const arma::vec& B1, const arma::vec& B2){
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

// bool has_same_support(const arma::sp_mat& B1, const arma::sp_mat& B2){
//
//     if (B1.n_nonzero != B2.n_nonzero) {
//         return false;
//     } else {  // same number of nnz and Supp is sorted
//         arma::sp_mat::const_iterator i1, i2;
//         const arma::sp_mat::const_iterator i1_end = B1.end();
//
//
//         for(i1 = B1.begin(), i2 = B2.begin(); i1 != i1_end; ++i1, ++i2)
//         {
//             if(i1.row() != i2.row())
//             {
//                 return false;
//             }
//         }
//         return true;
//     }
// }

#endif // BETA_VECTOR_H
