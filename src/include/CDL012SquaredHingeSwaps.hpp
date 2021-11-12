#ifndef CDL012SquredHingeSwaps_H
#define CDL012SquredHingeSwaps_H
#include <armadillo>
#include "CD.hpp"
#include "CDSwaps.hpp"
#include "CDL012SquaredHinge.hpp"
#include "Params.hpp"
#include "FitResult.hpp"
#include "utils.hpp"
#include "BetaVector.hpp"

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

        inline double Objective(const arma::vec & r, const beta_vector & B) final;
        
        inline double Objective() final;


};

template <class T>
inline double CDL012SquaredHingeSwaps<T>::Objective(const arma::vec & onemyxb, const beta_vector & B) {
    auto l2norm = arma::norm(B, 2);
    arma::uvec indices = arma::find(onemyxb > 0);
    return arma::sum(onemyxb.elem(indices) % onemyxb.elem(indices)) + this->lambda0 * n_nonzero(B) + this->lambda1 * arma::norm(B, 1) + this->lambda2 * l2norm * l2norm;
}

template <class T>
inline double CDL012SquaredHingeSwaps<T>::Objective() {
    throw std::runtime_error("CDL012SquaredHingeSwaps does not have this->onemyxb");
}


#endif
