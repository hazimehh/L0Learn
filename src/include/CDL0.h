#ifndef CDL0_H
#define CDL0_H
#include <tuple>
#include "RcppArmadillo.h"
#include "CD.h"
#include "FitResult.h" // REMOVE: may need to include params??
#include "utils.h"


template <class T>
class CDL0: public CD<T> {
    private:
        double thr;
        std::vector<double> * Xtr;
        std::size_t Iter;
        std::size_t ScreenSize;
        std::vector<std::size_t> Range1p;
        std::size_t NoSelectK;
    public:
        CDL0(const T& Xi, const arma::vec& yi, const Params<T>& P);
        //~CDL0(){}

        FitResult<T> Fit() final;
    
        double Objective(arma::vec & r, arma::sp_mat & B) final;
        
        bool CWMinCheck();
};

template <class T>
inline double CDL0<T>::Objective(arma::vec & r, arma::sp_mat & B) {
    return 0.5 * arma::dot(r, r) + this->ModelParams[0] * B.n_nonzero;
}


#endif
