#ifndef CDL012LogisticSwaps_H
#define CDL012LogisticSwaps_H
#include "RcppArmadillo.h"
#include "CD.h"
#include "CDSwaps.h"
#include "CDL012Logistic.h"
#include "FitResult.h"
#include "Params.h"
#include "utils.h"

template <class T>
class CDL012LogisticSwaps : public CDSwaps<T> {
    private:
        const double LipschitzConst = 0.25;
        double twolambda2;
        double qp2lamda2;
        double lambda1ol;
        double stl0Lc;
        arma::vec ExpyXB;
        // std::vector<double> * Xtr;
        T * Xy;

    public:
        CDL012LogisticSwaps(const T& Xi, const arma::vec& yi, const Params<T>& P);

        FitResult<T> Fit() final;

        inline double Objective(const arma::vec & r, const arma::sp_mat & B) final;
        
        inline double Objective() final;

};


template <class T>
inline double CDL012LogisticSwaps<T>::Objective(const arma::vec & r, const arma::sp_mat & B) {
    auto l2norm = arma::norm(B, 2);
    return arma::sum(arma::log(1 + 1 / r)) + this->lambda0 * B.n_nonzero + this->lambda1 * arma::norm(B, 1) + this->lambda2 * l2norm * l2norm;
}

template <class T>
inline double CDL012LogisticSwaps<T>::Objective() {
    auto l2norm = arma::norm(this->B, 2);
    return arma::sum(arma::log(1 + 1 / ExpyXB)) + this->lambda0 * this->B.n_nonzero + this->lambda1 * arma::norm(this->B, 1) + this->lambda2 * l2norm * l2norm;
}

#endif
