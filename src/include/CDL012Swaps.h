#ifndef CDL012SWAPS_H
#define CDL012SWAPS_H
#include <map>
#include "RcppArmadillo.h"
#include "CD.h"
#include "CDL012.h"
#include "FitResult.h"
#include "Params.h"
#include "utils.h"

template <class T>
class CDL012Swaps : public CD<T> { 
    private:
        std::size_t MaxNumSwaps;
        Params<T> P;
        std::size_t NoSelectK;
    public:
        CDL012Swaps(const T& Xi, const arma::vec& yi, const Params<T>& Pi);

        FitResult<T> Fit() final;

        double Objective(arma::vec & r, arma::sp_mat & B) final;

};

template <class T>
inline double CDL012Swaps<T>::Objective(arma::vec & r, arma::sp_mat & B) {
    auto l2norm = arma::norm(B, 2);
    return 0.5 * arma::dot(r, r) + this->ModelParams[0] * B.n_nonzero + this->ModelParams[1] * arma::norm(B, 1) + this->ModelParams[2] * l2norm * l2norm;
}

#endif
