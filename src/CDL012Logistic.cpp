#include "CDL012Logistic.h"

template <class T>
CDL012Logistic<T>::CDL012Logistic(const T& Xi, const arma::vec& yi, const Params<T>& P) : CD<T>(Xi, yi, P) {
    twolambda2 = 2 * this->lambda2;
    qp2lamda2 = (LipschitzConst + twolambda2); // this is the univariate lipschitz const of the differentiable objective
    this->thr2 = (2 * this->lambda0) / qp2lamda2;
    this->thr = std::sqrt(this->thr2);
    lambda1ol = this->lambda1 / qp2lamda2;
    
    ExpyXB = arma::exp(*this->y % (*(this->X) * this->B + this->b0)); // Maintained throughout the algorithm
    Xy = P.Xy;
}

template <class T>
FitResult<T> CDL012Logistic<T>::Fit() { // always uses active sets
    
    this->B = clamp_by_vector(this->B, this->Lows, this->Highs);
    
    this->objective = Objective(); // Implicitly used ExpyXB
    
    std::vector<std::size_t> FullOrder = this->Order; // never used in LR
    this->Order.resize(std::min((int) (this->B.n_nonzero + this->ScreenSize + this->NoSelectK), (int)(this->p)));
    
    for (std::size_t t = 0; t < this->MaxIters; ++t) {
        this->Bprev = this->B;
        
        // Update the intercept
        if (this->intercept){
            const double b0old = this->b0;
            const double partial_b0 = - arma::sum( *(this->y) / (1 + ExpyXB) );
            this->b0 -= partial_b0 / (this->n * LipschitzConst); // intercept is not regularized
            ExpyXB %= arma::exp( (this->b0 - b0old) * *(this->y));
        }
        
        for (auto& i : this->Order) {
            this->UpdateBi(i);
        }
        
        this->SupportStabilized();
        
        // only way to terminate is by (i) converging on active set and (ii) CWMinCheck
        if (this->Converged() && CWMinCheck()) {
            break;
        }
    }
    
    this->result.Objective = this->objective;
    this->result.B = this->B;
    this->result.Model = this;
    this->result.b0 = this->b0;
    this->result.ExpyXB = ExpyXB;
    this->result.IterNum = this->CurrentIters;
    
    return this->result;
}

template <class T>
bool CDL012Logistic<T>::CWMinCheck() {
    // Checks for violations outside Supp and updates Order in case of violations
    std::vector<std::size_t> S;
    
    for (arma::sp_mat::const_iterator it = this->B.begin(); it != this->B.end(); ++it)
        S.push_back(it.row());
    
    std::vector<std::size_t> Sc;
    set_difference(
        this->Range1p.begin(),
        this->Range1p.end(),
        S.begin(),
        S.end(),
        back_inserter(Sc));
    
    bool Cwmin = true;
    
    for (auto& i : Sc)  {
        Cwmin = this->UpdateBiCWMinCheck(i, Cwmin);
    }
    return Cwmin;
}

template class CDL012Logistic<arma::mat>;
template class CDL012Logistic<arma::sp_mat>;

