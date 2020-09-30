#ifndef CDL012SquredHingeSwaps_H
#define CDL012SquredHingeSwaps_H
#include "RcppArmadillo.h"
#include "CD.h"
#include "CDL012SquaredHinge.h"
#include "Params.h"
#include "FitResult.h"
#include "utils.h"

template <class T>
class CDL012SquaredHingeSwaps : public CD<T> {
    private:
        const double LipschitzConst = 2;
        double thr;
        double twolambda2;
        double qp2lamda2;
        double lambda1;
        double lambda1ol;
        double b0;
        double stl0Lc;
        std::vector<double> * Xtr;
        std::size_t Iter;
        std::size_t NoSelectK;

        std::size_t MaxNumSwaps;
        Params<T> P;

    public:
        CDL012SquaredHingeSwaps(const T& Xi, const arma::vec& yi, const Params<T>& P);

        FitResult<T> Fit() final;

        inline double Objective(arma::vec & r, arma::sp_mat & B) final;


};

template <class T>
inline double CDL012SquaredHingeSwaps<T>::Objective(arma::vec & onemyxb, arma::sp_mat & B) {
    auto l2norm = arma::norm(B, 2);
    arma::uvec indices = arma::find(onemyxb > 0);
    return arma::sum(onemyxb.elem(indices) % onemyxb.elem(indices)) + this->ModelParams[0] * B.n_nonzero + this->ModelParams[1] * arma::norm(B, 1) + this->ModelParams[2] * l2norm * l2norm;
}


#endif
