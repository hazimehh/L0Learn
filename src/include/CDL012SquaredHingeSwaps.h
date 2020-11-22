#ifndef CDL012SquredHingeSwaps_H
#define CDL012SquredHingeSwaps_H
#include "RcppArmadillo.h"
#include "CD.h"
#include "CDSwaps.h"
#include "CDL012SquaredHinge.h"
#include "Params.h"
#include "FitResult.h"
#include "utils.h"

template <class T>
class CDL012SquaredHingeSwaps : public CDSwaps<T> {
    private:
        const double LipschitzConst = 2;
        double twolambda2;
        double qp2lamda2;
        double lambda1ol;
        double stl0Lc;
        // std::vector<double> * Xtr;
        
    public:
        CDL012SquaredHingeSwaps(const T& Xi, const arma::vec& yi, const Params<T>& P);

        FitResult<T> _FitWithBounds() final;
        
        FitResult<T> _Fit() final;

        inline double Objective(const arma::vec & r, const arma::sp_mat & B) final;
        
        inline double Objective() final;


};

template <class T>
inline double CDL012SquaredHingeSwaps<T>::Objective(const arma::vec & onemyxb, const arma::sp_mat & B) {
    auto l2norm = arma::norm(B, 2);
    arma::uvec indices = arma::find(onemyxb > 0);
    return arma::sum(onemyxb.elem(indices) % onemyxb.elem(indices)) + this->lambda0 * B.n_nonzero + this->lambda1 * arma::norm(B, 1) + this->lambda2 * l2norm * l2norm;
}

template <class T>
inline double CDL012SquaredHingeSwaps<T>::Objective() {
    throw std::runtime_error("CDL012SquaredHingeSwaps does not have this->onemyxb");
}


#endif
