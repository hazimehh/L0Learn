#ifndef CDL012_H
#define CDL012_H
#include "RcppArmadillo.h"
#include "CD.h"
#include "FitResult.h"
#include "utils.h"

template <class T>
class CDL012 : public CD<T> {
    private:
        double thr;
        double Onep2lamda2;
        double lambda1;
        std::vector<double> * Xtr;
        std::size_t Iter;
        std::size_t ScreenSize;
        std::vector<std::size_t> Range1p;
        std::size_t NoSelectK;
    public:
        CDL012(const T& Xi, const arma::vec& yi, const Params<T>& P);
        //~CDL012(){}

        FitResult<T> Fit() final;

        inline double Objective(arma::vec & r, arma::sp_mat & B) final;

        bool CWMinCheck();

};

template <class T>
inline double CDL012<T>::Objective(arma::vec & r, arma::sp_mat & B) {   // hint inline
    auto l2norm = arma::norm(B, 2);
    return 0.5 * arma::dot(r, r) + this->ModelParams[0] * B.n_nonzero + this->ModelParams[1] * arma::norm(B, 1) + this->ModelParams[2] * l2norm * l2norm;
}

#endif
