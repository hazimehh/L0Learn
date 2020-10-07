#ifndef CDL0_H
#define CDL0_H
#include <tuple>
#include "RcppArmadillo.h"
#include "CD.h"
#include "Params.h"
#include "FitResult.h"
#include "utils.h"


template <class T>
class CDL0: public CD<T> {
    private:
        
        arma::vec r; //vector of residuals
        
    public:
        CDL0(const T& Xi, const arma::vec& yi, const Params<T>& P);
        //~CDL0(){}

        FitResult<T> Fit() final;
    
        inline double Objective(const arma::vec &, const arma::sp_mat &) final;
        
        inline double Objective() final;
        
        inline double GetBiGrad(const std::size_t i) final;
        
        inline double GetBiValue(const double old_Bi, const double grd_Bi) final;
        
        inline double GetBiReg(const double nrb_Bi) final;
        
        // inline double GetBiDelta(const double reg_Bi) final;
        
        inline void ApplyNewBi(const std::size_t i, const double old_Bi, const double new_Bi) final;
        
        inline void ApplyNewBiCWMinCheck(const std::size_t i, const double old_Bi, const double new_Bi) final;
        
        bool CWMinCheck();
};

template <class T>
inline double CDL0<T>::GetBiGrad(const std::size_t i){
    return matrix_column_dot(*(this->X), i, this->r);
}

template <class T>
inline double CDL0<T>::GetBiValue(const double old_Bi, const double grd_Bi){
    return grd_Bi + old_Bi;
}

template <class T>
inline double CDL0<T>::GetBiReg(const double nrb_Bi){
    return nrb_Bi;
}

// template <class T>
// inline double CDL0<T>::GetBiDelta(const double Bi_reg){
//     return std::sqrt(Bi_reg*Bi_reg - 2*this->lambda0);
// }

template <class T>
inline void CDL0<T>::ApplyNewBi(const std::size_t i, const double old_Bi, const double new_Bi){
    this->r += matrix_column_mult(*(this->X), i, old_Bi - new_Bi);
    this->B[i] = new_Bi;
}

template <class T>
inline void CDL0<T>::ApplyNewBiCWMinCheck(const std::size_t i, const double old_Bi, const double new_Bi){
    this->r += matrix_column_mult(*(this->X), i, old_Bi - new_Bi);
    this->B[i] = new_Bi;
}

template <class T>
inline double CDL0<T>::Objective(const arma::vec & r, const arma::sp_mat & B) {
    return 0.5 * arma::dot(r, r) + this->lambda0 * B.n_nonzero;
}

template <class T>
inline double CDL0<T>::Objective() {
    return 0.5 * arma::dot(this->r, this->r) + this->lambda0 * this->B.n_nonzero;
}



#endif
