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
        double twolambda2;
        double qp2lamda2;
        double lambda1ol;
        arma::vec ExpyXB;
        // std::vector<double> * Xtr;
        T * Xy;
        
    public:
        CDL012Logistic(const T& Xi, const arma::vec& yi, const Params<T>& P);
        //~CDL012Logistic(){}
        FitResult<T> Fit() final;

        inline double Objective(const arma::vec & r, const arma::sp_mat & B) final;
        
        inline double Objective() final;
        
        inline double GetBiGrad(const std::size_t i) final;
        
        inline double GetBiValue(const double old_Bi, const double grd_Bi) final;
        
        inline double GetBiReg(const double Bi_step) final;
        
        // inline double GetBiDelta(const double Bi_reg) final;
        
        inline void ApplyNewBi(const std::size_t i, const double Bi_old, const double Bi_new) final;
        
        inline void ApplyNewBiCWMinCheck(const std::size_t i, const double old_Bi, const double new_Bi) final;

        bool CWMinCheck();

};

template <class T>
inline double CDL012Logistic<T>::GetBiGrad(const std::size_t i){
    return -arma::sum( matrix_column_get(*(this->Xy), i) / (1 + ExpyXB) ) + twolambda2 * this->B[i];
}

template <class T>
inline double CDL012Logistic<T>::GetBiValue(const double old_Bi, const double grd_Bi){
    return old_Bi - grd_Bi/qp2lamda2;
}

template <class T>
inline double CDL012Logistic<T>::GetBiReg(const double Bi_step){
    return std::copysign(std::abs(Bi_step) - lambda1ol, Bi_step);
}

// template <class T>
// inline double CDL012Logistic<T>::GetBiDelta(const double Bi_reg){
//     return std::sqrt(Bi_reg*Bi_reg - 2*this->lambda0);
// }

template <class T>
inline void CDL012Logistic<T>::ApplyNewBi(const std::size_t i, const double old_Bi, const double new_Bi){
    ExpyXB %= arma::exp( (new_Bi - old_Bi) * matrix_column_get(*(this->Xy), i));
    this->B[i] = new_Bi;
}

template <class T>
inline void CDL012Logistic<T>::ApplyNewBiCWMinCheck(const std::size_t i,
                                                    const double old_Bi,
                                                    const double new_Bi){
    ExpyXB %= arma::exp( (new_Bi - old_Bi) * matrix_column_get(*(this->Xy), i));
    this->B[i] = new_Bi;
    this->Order.push_back(i);
}

template <class T>
inline double CDL012Logistic<T>::Objective(const arma::vec & expyXB, const arma::sp_mat & B) {  // hint inline
    const auto l2norm = arma::norm(B, 2);
    // arma::sum(arma::log(1 + 1 / expyXB)) is the negative log-likelihood
    return arma::sum(arma::log(1 + 1 / expyXB)) + this->lambda0 * B.n_nonzero + this->lambda1 * arma::norm(B, 1) + this->lambda2 * l2norm * l2norm;
}

template <class T>
inline double CDL012Logistic<T>::Objective() {  // hint inline
    const auto l2norm = arma::norm(this->B, 2);
    // arma::sum(arma::log(1 + 1 / ExpyXB)) is the negative log-likelihood
    return arma::sum(arma::log(1 + 1 / ExpyXB)) + this->lambda0 * this->B.n_nonzero + this->lambda1 * arma::norm(this->B, 1) + this->lambda2 * l2norm * l2norm;
}


#endif
