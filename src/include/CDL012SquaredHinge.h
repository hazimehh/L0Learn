#ifndef CDL012SquaredHinge_H
#define CDL012SquaredHinge_H
#include "RcppArmadillo.h"
#include "CD.h"
#include "FitResult.h"
#include "Params.h"
#include "utils.h"

template <class T>
class CDL012SquaredHinge : public CD<T> {
    private:
        const double LipschitzConst = 2; // for f (without regularization)
        double twolambda2;
        double qp2lamda2;
        double lambda1ol;
        // std::vector<double> * Xtr;
        arma::vec onemyxb;
        arma::uvec indices;
        T * Xy;


    public:
        CDL012SquaredHinge(const T& Xi, const arma::vec& yi, const Params<T>& P);
        
        //~CDL012SquaredHinge(){}

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
inline double CDL012SquaredHinge<T>::GetBiGrad(const std::size_t i){
    // Rcpp::Rcout << "Grad stuff: " << arma::sum(2 * onemyxb.elem(indices) % (- matrix_column_get(*Xy, i).elem(indices))  ) << "\n";
    return arma::sum(2 * onemyxb.elem(indices) % (- matrix_column_get(*Xy, i).elem(indices)) ) + twolambda2 * this->B[i];
}

template <class T>
inline double CDL012SquaredHinge<T>::GetBiValue(const double old_Bi, const double grd_Bi){
    return old_Bi - grd_Bi / qp2lamda2;
}

template <class T>
inline double CDL012SquaredHinge<T>::GetBiReg(const double Bi_step){
    return std::abs(Bi_step) - lambda1ol;
}

// template <class T>
// inline double CDL012SquaredHinge<T>::GetBiDelta(const double Bi_reg){
//     return std::sqrt(Bi_reg*Bi_reg - 2*this->lambda0);
// }

template <class T>
inline void CDL012SquaredHinge<T>::ApplyNewBi(const std::size_t i, const double Bi_old, const double Bi_new){
    onemyxb += (Bi_old - Bi_new) * matrix_column_get(*(this->Xy), i);
    this->B[i] = Bi_new;
    indices = arma::find(onemyxb > 0);
}

template <class T>
inline void CDL012SquaredHinge<T>::ApplyNewBiCWMinCheck(const std::size_t i, const double Bi_old, const double Bi_new){
    onemyxb += (Bi_old - Bi_new) * matrix_column_get(*(this->Xy), i);
    this->B[i] = Bi_new;
    indices = arma::find(onemyxb > 0);
    this->Order.push_back(i);
}

template <class T>
inline double CDL012SquaredHinge<T>::Objective(const arma::vec & onemyxb, const arma::sp_mat & B) {
    
    auto l2norm = arma::norm(B, 2);
    // arma::vec onemyxb = 1 - *y % (*X * B + b0);
    // arma::uvec indices = arma::find(onemyxb > 0);
    
    return arma::sum(onemyxb.elem(indices) % onemyxb.elem(indices)) + this->lambda0 * B.n_nonzero + this->lambda1 * arma::norm(B, 1) + this->lambda2 * l2norm * l2norm;
}


template <class T>
inline double CDL012SquaredHinge<T>::Objective() {
    
    auto l2norm = arma::norm(this->B, 2);
    //arma::vec onemyxb = 1 - *y % (*X * B + b0);
    // arma::uvec indices = arma::find(onemyxb > 0);
    
    return arma::sum(onemyxb.elem(indices) % onemyxb.elem(indices)) + this->lambda0 * this->B.n_nonzero + this->lambda1 * arma::norm(this->B, 1) + this->lambda2 * l2norm * l2norm;
}

#endif
