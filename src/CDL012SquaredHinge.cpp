#include "CDL012SquaredHinge.h"

template <class T>
CDL012SquaredHinge<T>::CDL012SquaredHinge(const T& Xi, const arma::vec& yi, const Params<T>& P) : CD<T>(Xi, yi, P) {
    twolambda2 = 2 * this->lambda2;
    qp2lamda2 = (LipschitzConst + twolambda2); // this is the univariate lipschitz const of the differentiable objective
    this->thr2 = (2 * this->lambda0) / qp2lamda2;
    this->thr = std::sqrt(this->thr2);
    lambda1ol = this->lambda1 / qp2lamda2;
    
    this->B = clamp_by_vector(this->B, this->Lows, this->Highs);
    
    // TODO: Review this line
    // TODO: Pass work from previous solution.
    onemyxb = 1 - *(this->y) % (*(this->X) * this->B + this->b0);
    
    // TODO: Add comment for purpose of 'indices'
    indices = arma::find(onemyxb > 0);
    Xy = P.Xy;
}


template <class T>
FitResult<T> CDL012SquaredHinge<T>::Fit() {
    
    this->objective = Objective(); // Implicitly uses onemyx
    
    
    std::vector<std::size_t> FullOrder = this->Order; // never used in LR
    this->Order.resize(std::min((int) (this->B.n_nonzero + this->ScreenSize + this->NoSelectK), (int)(this->p)));
    
    
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
        if (this->Converged()) {
            if (this->CWMinCheck()) {
                break;
            }
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


template class CDL012SquaredHinge<arma::mat>;
template class CDL012SquaredHinge<arma::sp_mat>;

