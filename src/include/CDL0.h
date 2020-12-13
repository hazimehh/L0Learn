#ifndef CDL0_H
#define CDL0_H
#include <tuple>
#include "RcppArmadillo.h"
#include "CD.h"
#include "Params.h"
#include "FitResult.h"
#include "utils.h"


template <class T>
class CDL0: public CD<T, CDL0<T>>{
    private:
        
        arma::vec r; //vector of residuals
        
    public:
        CDL0(const T& Xi, const arma::vec& yi, const Params<T>& P);
        //~CDL0(){}

        FitResult<T> _FitWithBounds() final;
        
        FitResult<T> _Fit() final;
    
        inline double Objective(const arma::vec &, const arma::sp_mat &) final;
        
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
}

template <class T>
inline double CDL0<T>::Objective(const arma::vec & r, const arma::sp_mat & B) {
    return 0.5 * arma::dot(r, r) + this->lambda0 * B.n_nonzero;
}

template <class T>
inline double CDL0<T>::Objective() {
    return 0.5 * arma::dot(this->r, this->r) + this->lambda0 * this->B.n_nonzero;
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
    this->objective = Objective(this->r, this->B);
    
    std::vector<std::size_t> FullOrder = this->Order;
    bool FirstRestrictedPass = true;
    
    if (this->ActiveSet) {
        this->Order.resize(std::min((int) (this->B.n_nonzero + this->ScreenSize + this->NoSelectK), (int)(this->p))); // std::min(1000,Order.size())
    }
    
    bool ActiveSetInitial = this->ActiveSet;
    
    for (std::size_t t = 0; t < this->MaxIters; ++t) {
        this->Bprev = this->B;
        
        if (this->isSparse && this->intercept){
            this->UpdateSparse_b0(this->r);
        }
        
        for (auto& i : this->Order) {
            this->UpdateBi(i);
        }
        
        if (this->Converged()) {
            if(FirstRestrictedPass && ActiveSetInitial) {
                if (this->CWMinCheck()) {
                    break;
                }
                FirstRestrictedPass = false;
                this->Stabilized = false;
                this->ActiveSet = true;
                this->Order = FullOrder;
                
            } else {
                if (this->Stabilized == true && ActiveSetInitial) { // && !SecondPass
                    if (this->CWMinCheck()) {
                        break;
                    }
                    this->Order = this->OldOrder; // Recycle over all coordinates to make sure the achieved point is a CW-min.
                    //SecondPass = true; // a 2nd pass will be performed
                    this->Stabilized = false;
                    this->ActiveSet = true;
                } else {
                    break;
                }
            }
        }
        
        if (this->ActiveSet) {
            this->SupportStabilized();
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
    
    this->B = clamp_by_vector(this->B, this->Lows, this->Highs);
    
    this->objective = Objective(this->r, this->B);
    
    std::vector<std::size_t> FullOrder = this->Order;
    bool FirstRestrictedPass = true;
    
    if (this->ActiveSet) {
        this->Order.resize(std::min((int) (this->B.n_nonzero + this->ScreenSize + this->NoSelectK), (int)(this->p))); // std::min(1000,Order.size())
    }
    
    bool ActiveSetInitial = this->ActiveSet;
    
    for (std::size_t t = 0; t < this->MaxIters; ++t) {
        this->Bprev = this->B;
        
        if (this->isSparse && this->intercept){
            this->UpdateSparse_b0(this->r);
        }
        
        for (auto& i : this->Order) {
            this->UpdateBiWithBounds(i);
        }
        
        if (this->Converged()) {
            if(FirstRestrictedPass && ActiveSetInitial) {
                if (this->CWMinCheckWithBounds()) {
                    break;
                }
                FirstRestrictedPass = false;
                this->Stabilized = false;
                this->ActiveSet = true;
                this->Order = FullOrder;
                
            } else {
                if (this->Stabilized == true && ActiveSetInitial) { // && !SecondPass
                    if (this->CWMinCheckWithBounds()) {
                        break;
                    }
                    this->Order = this->OldOrder; // Recycle over all coordinates to make sure the achieved point is a CW-min.
                    //SecondPass = true; // a 2nd pass will be performed
                    this->Stabilized = false;
                    this->ActiveSet = true;
                } else {
                    break;
                }
            }
        }
        
        if (this->ActiveSet) {
            this->SupportStabilized();
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
