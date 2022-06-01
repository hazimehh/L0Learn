#ifndef CDL012_H
#define CDL012_H
#include "arma_includes.h"
#include "CD.h"
#include "FitResult.h"
#include "utils.h"
#include "BetaVector.h"

template <class T>
class CDL012 : public CD<T, CDL012<T>>{
    private:
        double Onep2lamda2;
        arma::vec r; //vector of residuals
        
    public:
        CDL012(const T& Xi, const arma::vec& yi, const Params<T>& P);
        //~CDL012(){}

        FitResult<T> _FitWithBounds() final;
        
        FitResult<T> _Fit() final;

        inline double Objective(const arma::vec & r, const beta_vector & B) final;
        
        inline double Objective() final;

        inline double GetBiGrad(const std::size_t i);
        
        inline double GetBiValue(const double old_Bi, const double grd_Bi);
        
        inline double GetBiReg(const double nrb_Bi);
        
        inline void ApplyNewBi(const std::size_t i, const double old_Bi, const double new_Bi); 
        
        inline void ApplyNewBiCWMinCheck(const std::size_t i, const double old_Bi, const double new_Bi);

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

template <class T>
inline void CDL012<T>::ApplyNewBi(const std::size_t i, const double Bi_old, const double Bi_new){
    this->r += matrix_column_mult(*(this->X), i, Bi_old - Bi_new);
    this->B[i] = Bi_new;
}

template <class T>
inline void CDL012<T>::ApplyNewBiCWMinCheck(const std::size_t i, const double Bi_old, const double Bi_new){
    this->r += matrix_column_mult(*(this->X), i, Bi_old - Bi_new);
    this->B[i] = Bi_new;
    this->Order.push_back(i);
}

template <class T>
inline double CDL012<T>::Objective(const arma::vec & r, const beta_vector & B) { 
    auto l2norm = arma::norm(B, 2);
    return 0.5 * arma::dot(r, r) + this->lambda0 * n_nonzero(this->B) + this->lambda1 * arma::norm(B, 1) + this->lambda2 * l2norm * l2norm;
}

template <class T>
inline double CDL012<T>::Objective() { 
    auto l2norm = arma::norm(this->B, 2);
    return 0.5 * arma::dot(this->r, this->r) + this->lambda0 * n_nonzero(this->B) + this->lambda1 * arma::norm(this->B, 1) + this->lambda2 * l2norm * l2norm;
}

template <class T>
CDL012<T>::CDL012(const T& Xi, const arma::vec& yi, const Params<T>& P) : CD<T, CDL012<T>>(Xi, yi, P) {
    Onep2lamda2 = 1 + 2 * this->lambda2; 
    
    this->thr2 = 2 * this->lambda0 / Onep2lamda2;
    this->thr = std::sqrt(this->thr2);
    this->r = *P.r;
    this->result.r = P.r;
}

template <class T>
FitResult<T> CDL012<T>::_Fit() {
    
    this->objective = Objective(this->r, this->B);
    
    std::vector<std::size_t> FullOrder = this->Order;

    if (this->ActiveSet) {
        this->Order.resize(std::min((int) (n_nonzero(this->B) + this->ScreenSize + this->NoSelectK), (int)(this->p)));
    }
    
    for (std::size_t t = 0; t < this->MaxIters; ++t) {
        this->Bprev = this->B;
        
        if (this->isSparse && this->intercept){
            this->UpdateSparse_b0(this->r);
        }
        
        for (auto& i : this->Order) {
            this->UpdateBi(i);
        }
        
        this->RestrictSupport();
        
        if (this->isConverged() && this->CWMinCheck()) {
            break;
        }
    }
    
    if (this->isSparse && this->intercept){
        this->UpdateSparse_b0(this->r);
    }
    
    this->result.Objective = this->objective;
    this->result.B = this->B;
    *(this->result.r) = this->r; // change to pointer later
    this->result.IterNum = this->CurrentIters;
    this->result.b0 = this->b0;
    return this->result;
}

template <class T>
FitResult<T> CDL012<T>::_FitWithBounds() {
    
    clamp_by_vector(this->B, this->Lows, this->Highs);
    
    this->objective = Objective(this->r, this->B);
    
    std::vector<std::size_t> FullOrder = this->Order;

    if (this->ActiveSet) {
        this->Order.resize(std::min((int) (n_nonzero(this->B) + this->ScreenSize + this->NoSelectK), (int)(this->p)));
    }
    
    for (std::size_t t = 0; t < this->MaxIters; ++t) {
        this->Bprev = this->B;
        
        if (this->isSparse && this->intercept){
            this->UpdateSparse_b0(this->r);
        }
        
        for (auto& i : this->Order) {
            this->UpdateBiWithBounds(i);
        }
        
        this->RestrictSupport();
        
        //B.print();
        if (this->isConverged() && this->CWMinCheckWithBounds()) {
            break;
        }
    
    }
    
    if (this->isSparse && this->intercept){
        this->UpdateSparse_b0(this->r);
    }
    
    this->result.Objective = this->objective;
    this->result.B = this->B;
    *(this->result.r) = this->r; // change to pointer later
    this->result.IterNum = this->CurrentIters;
    this->result.b0 = this->b0;
    return this->result;
}

template class CDL012<arma::mat>;
template class CDL012<arma::sp_mat>;

#endif
