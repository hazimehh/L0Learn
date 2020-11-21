#ifndef CDL012_H
#define CDL012_H
#include "RcppArmadillo.h"
#include "CD.h"
#include "FitResult.h"
#include "utils.h"

template <class T>
class CDL012 : public CD<T> {
    private:
        double Onep2lamda2;
        arma::vec r; //vector of residuals
        
    public:
        CDL012(const T& Xi, const arma::vec& yi, const Params<T>& P);
        //~CDL012(){}

        FitResult<T> Fit() final;

        inline double Objective(const arma::vec & r, const arma::sp_mat & B) final;
        
        inline double Objective() final;

        inline double GetBiGrad(const std::size_t i) final;
        
        inline double GetBiValue(const double old_Bi, const double grd_Bi) final;
        
        inline double GetBiReg(const double nrb_Bi) final;
        
        // inline double GetBiDelta(const double reg_Bi) final;
        
        inline void ApplyNewBi(const std::size_t i, const double old_Bi, const double new_Bi) final; 
        
        inline void ApplyNewBiCWMinCheck(const std::size_t i, const double old_Bi, const double new_Bi) final;

};


template <class T>
inline double CDL012<T>::GetBiGrad(const std::size_t i){
    return matrix_column_dot(*(this->X), i, this->r);
}

template <class T>
inline double CDL012<T>::GetBiValue(const double old_Bi, const double grd_Bi){
    return grd_Bi + old_Bi;
}

template <class T>
inline double CDL012<T>::GetBiReg(const double nrb_Bi){
    // sign(nrb_Bi)*(|nrb_Bi| - lambda1)/(1 + 2*lambda2)
    return (std::abs(nrb_Bi) - this->lambda1) / Onep2lamda2;
}

// template <class T>
// inline double CDL012<T>::GetBiDelta(const double Bi_reg){
//     return std::sqrt(Bi_reg*Bi_reg - this->thr2);
// }

template <class T>
inline void CDL012<T>::ApplyNewBi(const std::size_t i, const double Bi_old, const double Bi_new){
    this->r += matrix_column_mult(*(this->X), i, Bi_old - Bi_new);
    this->B[i] = Bi_new;
}

template <class T>
inline void CDL012<T>::ApplyNewBiCWMinCheck(const std::size_t i, const double Bi_old, const double Bi_new){
    this->r += matrix_column_mult(*(this->X), i, Bi_old - Bi_new);
    this->B[i] = Bi_new;
}

template <class T>
inline double CDL012<T>::Objective(const arma::vec & r, const arma::sp_mat & B) { 
    auto l2norm = arma::norm(B, 2);
    return 0.5 * arma::dot(r, r) + this->lambda0 * B.n_nonzero + this->lambda1 * arma::norm(B, 1) + this->lambda2 * l2norm * l2norm;
}

template <class T>
inline double CDL012<T>::Objective() { 
    auto l2norm = arma::norm(this->B, 2);
    return 0.5 * arma::dot(this->r, this->r) + this->lambda0 * this->B.n_nonzero + this->lambda1 * arma::norm(this->B, 1) + this->lambda2 * l2norm * l2norm;
}

#endif
