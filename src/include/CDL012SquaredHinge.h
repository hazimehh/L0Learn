#ifndef CDL012SquaredHinge_H
#define CDL012SquaredHinge_H
#include "RcppArmadillo.h"
#include "CD.h"
#include "FitResult.h"
#include "Params.h"
#include "utils.h"
// #include "Normalize.h" // Remove later
// #include "Grid.h" // Remove later

template <class T>
class CDL012SquaredHinge : public CD<T> {
    private:
        const double LipschitzConst = 2; // for f (without regularization)
        double thr;
        double twolambda2;
        double qp2lamda2;
        double lambda1;
        double lambda1ol;
        std::vector<double> * Xtr;
        std::size_t Iter;
        arma::vec onemyxb;
        std::size_t NoSelectK;
        T * Xy;
        std::size_t ScreenSize;
        std::vector<std::size_t> Range1p;


    public:
        CDL012SquaredHinge(const T& Xi, const arma::vec& yi, const Params<T>& P);
        //~CDL012SquaredHinge(){}
        //inline double Derivativei(std::size_t i);

        //inline double Derivativeb();

        FitResult<T> Fit() final;

        inline double Objective(arma::vec & r, arma::sp_mat & B) final;

        bool CWMinCheck();


};

template <class T>
inline double CDL012SquaredHinge<T>::Objective(arma::vec & r, arma::sp_mat & B) {
    
    auto l2norm = arma::norm(this->B, 2);
    //arma::vec onemyxb = 1 - *y % (*X * B + b0);
    arma::uvec indices = arma::find(onemyxb > 0);
    
    return arma::sum(onemyxb.elem(indices) % onemyxb.elem(indices)) + this->ModelParams[0] * B.n_nonzero + this->ModelParams[1] * arma::norm(B, 1) + this->ModelParams[2] * l2norm * l2norm;
}


#endif
