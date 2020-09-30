#ifndef CDL012Logistic_H
#define CDL012Logistic_H
#include "RcppArmadillo.h"
#include "CD.h"
#include "FitResult.h"
#include "Params.h"
#include "utils.h"

template <class T>
class CDL012Logistic : public CD<T> {
    private:
        const double LipschitzConst = 0.25;
        double thr;
        double twolambda2;
        double qp2lamda2;
        double lambda1;
        double lambda1ol;
        arma::vec ExpyXB;
        std::vector<double> * Xtr;
        T * Xy;
        std::size_t Iter;
        std::size_t NoSelectK;
        std::size_t ScreenSize;
        std::vector<std::size_t> Range1p;

    public:
        CDL012Logistic(const T& Xi, const arma::vec& yi, const Params<T>& P);
        //~CDL012Logistic(){}
        FitResult<T> Fit() final;

        inline double Objective(arma::vec & r, arma::sp_mat & B) final;

        bool CWMinCheck();

};

template <class T>
inline double CDL012Logistic<T>::Objective(arma::vec & r, arma::sp_mat & B) {  // hint inline
    const auto l2norm = arma::norm(B, 2);
    // arma::sum(arma::log(1 + 1 / ExpyXB)) is the negative log-likelihood
    return arma::sum(arma::log(1 + 1 / ExpyXB)) + this->ModelParams[0] * B.n_nonzero + this->ModelParams[1] * arma::norm(B, 1) + this->ModelParams[2] * l2norm * l2norm;
}


#endif
