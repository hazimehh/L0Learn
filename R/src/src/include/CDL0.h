#ifndef CDL0_H
#define CDL0_H
#include <tuple>
#include "arma_includes.h"
#include "CD.h"
#include "Params.h"
#include "FitResult.h"
#include "utils.h"
#include "BetaVector.h"


template <class T>
class CDL0: public CD<T, CDL0<T>>{
    private:
        
        arma::vec r; //vector of residuals
        
    public:
        CDL0(const T& Xi, const arma::vec& yi, const Params<T>& P);
        //~CDL0(){}

        FitResult<T> _FitWithBounds() final;
        
        FitResult<T> _Fit() final;
    
        inline double Objective(const arma::vec &, const beta_vector &) final;
        
        inline double Objective() final;
        
        inline double GetBiGrad(const std::size_t i);
        
        inline double GetBiValue(const double old_Bi, const double grd_Bi);
        
        inline double GetBiReg(const double nrb_Bi);
        
        inline void ApplyNewBi(const std::size_t i, const double old_Bi, const double new_Bi);
        
        inline void ApplyNewBiCWMinCheck(const std::size_t i, const double old_Bi, const double new_Bi);
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
    return std::abs(nrb_Bi);
}

template <class T>
inline void CDL0<T>::ApplyNewBi(const std::size_t i, const double old_Bi, const double new_Bi){
    this->r += matrix_column_mult(*(this->X), i, old_Bi - new_Bi);
    this->B[i] = new_Bi;
}

template <class T>
inline void CDL0<T>::ApplyNewBiCWMinCheck(const std::size_t i, const double old_Bi, const double new_Bi){
    this->r += matrix_column_mult(*(this->X), i, old_Bi - new_Bi);
    this->B[i] = new_Bi;
    this->Order.push_back(i);
}

template <class T>
inline double CDL0<T>::Objective(const arma::vec & r, const beta_vector & B) {
    return 0.5 * arma::dot(r, r) + this->lambda0 * n_nonzero(B);
}

template <class T>
inline double CDL0<T>::Objective() {
    return 0.5 * arma::dot(this->r, this->r) + this->lambda0 * n_nonzero(this->B);
}

template <class T>
CDL0<T>::CDL0(const T& Xi, const arma::vec& yi, const Params<T>& P) : CD<T, CDL0<T>>(Xi, yi, P){
    this->thr2 = 2 * this->lambda0;
    this->thr = sqrt(this->thr2); 
    this->r = *P.r; 
    this->result.r = P.r;
}

template <class T>
FitResult<T> CDL0<T>::_Fit() {
    // Rcpp::Rcout << "CDL0 Fit:";
    this->objective = Objective(this->r, this->B);
    
    std::vector<std::size_t> FullOrder = this->Order;
    
    if (this->ActiveSet) {
        this->Order.resize(std::min((int) (n_nonzero(this->B) + this->ScreenSize + this->NoSelectK), (int)(this->p))); // std::min(1000,Order.size())
    }
    
    for (std::size_t t = 0; t < this->MaxIters; ++t) {
        this->Bprev = this->B;
        
        if (this->isSparse && this->intercept){
            this->UpdateSparse_b0(this->r);
        }
        
        //Rcpp::Rcout << "{" << this->Order.size() << "}";
        for (auto& i : this->Order) {
            this->UpdateBi(i);
        }
        
        this->RestrictSupport();
        
        if (this->isConverged() && this->CWMinCheck()) {
            //Rcpp::Rcout << " |Converged on iter:" << t << "CWMinCheck \n";
            break;
        }
    }
                
    
    // Re-optimize b0 after convergence.
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
FitResult<T> CDL0<T>::_FitWithBounds() {
    // Rcpp::Rcout << "CDL0 Fit: ";
    clamp_by_vector(this->B, this->Lows, this->Highs);
    
    this->objective = Objective(this->r, this->B);
    
    std::vector<std::size_t> FullOrder = this->Order;
    
    if (this->ActiveSet) {
        this->Order.resize(std::min((int) (n_nonzero(this->B) + this->ScreenSize + this->NoSelectK), (int)(this->p))); // std::min(1000,Order.size())
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
        
        if (this->isConverged() && this->CWMinCheckWithBounds()) {
            // Rcpp::Rcout << "Converged on iter:" << t << "\n";
            break;
        }
    }
    
    // Re-optimize b0 after convergence.
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

template class CDL0<arma::mat>;
template class CDL0<arma::sp_mat>;

#endif
