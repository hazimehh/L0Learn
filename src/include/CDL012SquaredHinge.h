#ifndef CDL012SquaredHinge_H
#define CDL012SquaredHinge_H
#include "RcppArmadillo.h"
#include "CD.h"
#include "FitResult.h"
#include "Params.h"
#include "utils.h"
#include "BetaVector.h"

template <class T>
class CDL012SquaredHinge : public CD<T, CDL012SquaredHinge<T>> {
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

        FitResult<T> _FitWithBounds() final;
        
        FitResult<T> _Fit() final;

        inline double Objective(const arma::vec & r, const beta_vector & B) final;
        
        inline double Objective() final;
        
        inline double GetBiGrad(const std::size_t i);
        
        inline double GetBiValue(const double old_Bi, const double grd_Bi);
        
        inline double GetBiReg(const double Bi_step);
        
        inline void ApplyNewBi(const std::size_t i, const double Bi_old, const double Bi_new);
        
        inline void ApplyNewBiCWMinCheck(const std::size_t i, const double old_Bi, const double new_Bi);
        
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
inline double CDL012SquaredHinge<T>::Objective(const arma::vec & onemyxb, const beta_vector & B) {
    
    auto l2norm = arma::norm(B, 2);
    arma::uvec indices = arma::find(onemyxb > 0);
    
    return arma::sum(onemyxb.elem(indices) % onemyxb.elem(indices)) + this->lambda0 * n_nonzero(B) + this->lambda1 * arma::norm(B, 1) + this->lambda2 * l2norm * l2norm;
}


template <class T>
inline double CDL012SquaredHinge<T>::Objective() {
    
    auto l2norm = arma::norm(this->B, 2);
    return arma::sum(onemyxb.elem(indices) % onemyxb.elem(indices)) + this->lambda0 * n_nonzero(this->B) + this->lambda1 * arma::norm(this->B, 1) + this->lambda2 * l2norm * l2norm;
}

template <class T>
CDL012SquaredHinge<T>::CDL012SquaredHinge(const T& Xi, const arma::vec& yi, const Params<T>& P) : CD<T, CDL012SquaredHinge<T>>(Xi, yi, P) {
    twolambda2 = 2 * this->lambda2;
    qp2lamda2 = (LipschitzConst + twolambda2); // this is the univariate lipschitz const of the differentiable objective
    this->thr2 = (2 * this->lambda0) / qp2lamda2;
    this->thr = std::sqrt(this->thr2);
    lambda1ol = this->lambda1 / qp2lamda2;
    
    // TODO: Review this line
    // TODO: Pass work from previous solution.
    onemyxb = 1 - *(this->y) % (*(this->X) * this->B + this->b0);
    
    // TODO: Add comment for purpose of 'indices'
    indices = arma::find(onemyxb > 0);
    Xy = P.Xy;
}

template <class T>
FitResult<T> CDL012SquaredHinge<T>::_Fit() {
    
    this->objective = Objective(); // Implicitly uses onemyx
    
    
    std::vector<std::size_t> FullOrder = this->Order; // never used in LR
    this->Order.resize(std::min((int) (n_nonzero(this->B) + this->ScreenSize + this->NoSelectK), (int)(this->p)));
    
    
    for (auto t = 0; t < this->MaxIters; ++t) {
        
        this->Bprev = this->B;
        
        // Update the intercept
        if (this->intercept) {
            const double b0old = this->b0;
            const double partial_b0 = arma::sum(2 * onemyxb.elem(indices) % (- (this->y)->elem(indices) ) );
            this->b0 -= partial_b0 / (this->n * LipschitzConst); // intercept is not regularized
            onemyxb += *(this->y) * (b0old - this->b0);
            indices = arma::find(onemyxb > 0);
        }
        
        for (auto& i : this->Order) {
            this->UpdateBi(i);
        }
        
        this->SupportStabilized();
        
        // only way to terminate is by (i) converging on active set and (ii) CWMinCheck
        if ((this->Converged()) && this->CWMinCheck()) {
            break;
        }
    }
    
    this->result.Objective = this->objective;
    this->result.B = this->B;
    this->result.Model = this;
    this->result.b0 = this->b0;
    this->result.IterNum = this->CurrentIters;
    this->result.onemyxb = this->onemyxb;
    return this->result;
}


template <class T>
FitResult<T> CDL012SquaredHinge<T>::_FitWithBounds() {
    
    this->B = clamp_by_vector(this->B, this->Lows, this->Highs);
    
    this->objective = Objective(); // Implicitly uses onemyx
    
    
    std::vector<std::size_t> FullOrder = this->Order; // never used in LR
    this->Order.resize(std::min((int) (n_nonzero(this->B) + this->ScreenSize + this->NoSelectK), (int)(this->p)));
    
    
    for (auto t = 0; t < this->MaxIters; ++t) {
        
        this->Bprev = this->B;
        
        // Update the intercept
        if (this->intercept) {
            const double b0old = this->b0;
            const double partial_b0 = arma::sum(2 * onemyxb.elem(indices) % (- (this->y)->elem(indices) ) );
            this->b0 -= partial_b0 / (this->n * LipschitzConst); // intercept is not regularized
            onemyxb += *(this->y) * (b0old - this->b0);
            indices = arma::find(onemyxb > 0);
        }
        
        for (auto& i : this->Order) {
            this->UpdateBiWithBounds(i);
        }
        
        this->SupportStabilized();
        
        // only way to terminate is by (i) converging on active set and (ii) CWMinCheck
        if (this->Converged()) {
            if (this->CWMinCheckWithBounds()) {
                break;
            }
        }
    }
    
    this->result.Objective = this->objective;
    this->result.B = this->B;
    this->result.Model = this;
    this->result.b0 = this->b0;
    this->result.IterNum = this->CurrentIters;
    this->result.onemyxb = this->onemyxb;
    return this->result;
}

template class CDL012SquaredHinge<arma::mat>;
template class CDL012SquaredHinge<arma::sp_mat>;

#endif
