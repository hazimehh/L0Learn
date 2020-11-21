#include "CDL0.h"


template <class T>
CDL0<T>::CDL0(const T& Xi, const arma::vec& yi, const Params<T>& P) : CD<T>(Xi, yi, P){
    this->thr2 = 2 * this->lambda0;
    this->thr = sqrt(this->thr2); 
    this->r = *P.r; 
    this->result.r = P.r;
}

template <class T>
FitResult<T> CDL0<T>::Fit() {
    
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


template class CDL0<arma::mat>;
template class CDL0<arma::sp_mat>;
