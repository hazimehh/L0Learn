#ifndef CDL012LogisticSwaps_H
#define CDL012LogisticSwaps_H
// #include <chrono>
#include "RcppArmadillo.h"
#include "CD.h"
#include "FitResult.h"
#include "Params.h"
#include "CDL012Logistic.h"
#include "utils.h"

template <class T>
class CDL012LogisticSwaps : public CD<T> {
    private:
        const double LipschitzConst = 0.25;
        double thr;
        double twolambda2;
        double qp2lamda2;
        double lambda1;
        double lambda1ol;
        double b0;
        double stl0Lc;
        arma::vec ExpyXB;
        std::vector<double> * Xtr;
        T * Xy;
        std::size_t Iter;

        std::size_t MaxNumSwaps;
        Params<T> P;
        std::size_t NoSelectK;

    public:
        CDL012LogisticSwaps(const T& Xi, const arma::vec& yi, const Params<T>& P);

        FitResult<T> Fit() final;

        inline double Objective(arma::vec & r, arma::sp_mat & B) final;

};


template <class T>
inline double CDL012LogisticSwaps<T>::Objective(arma::vec & r, arma::sp_mat & B) {
    auto l2norm = arma::norm(B, 2);
    return arma::sum(arma::log(1 + 1 / r)) + this->ModelParams[0] * B.n_nonzero + this->ModelParams[1] * arma::norm(B, 1) + this->ModelParams[2] * l2norm * l2norm;
}

#endif
