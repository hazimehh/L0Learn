#ifndef CDL012SWAPS_H
#define CDL012SWAPS_H
#include <map>
#include "RcppArmadillo.h"
#include "CD.h"
#include "CDSwaps.h"
#include "CDL012.h"
#include "FitResult.h"
#include "Params.h"
#include "utils.h"
#include "BetaVector.h"

template <class T>
class CDL012Swaps : public CDSwaps<T> { 
    public:
        CDL012Swaps(const T& Xi, const arma::vec& yi, const Params<T>& Pi);

        FitResult<T> _FitWithBounds() final;
        
        FitResult<T> _Fit() final;

        double Objective(const arma::vec & r, const beta_vector & B) final;
        
        double Objective() final;

};

template <class T>
inline double CDL012Swaps<T>::Objective(const arma::vec & r, const beta_vector & B) {
    auto l2norm = arma::norm(B, 2);
    return 0.5 * arma::dot(r, r) + this->lambda0 * n_nonzero(B) + this->lambda1 * arma::norm(B, 1) + this->lambda2 * l2norm * l2norm;
}

template <class T>
inline double CDL012Swaps<T>::Objective() {
    throw std::runtime_error("CDL012Swaps does not have this->r.");
}

#endif
