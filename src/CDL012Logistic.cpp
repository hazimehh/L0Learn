#include "CDL012Logistic.h"

template <class T>
CDL012Logistic<T>::CDL012Logistic(const T& Xi, const arma::vec& yi, const Params<T>& P) : CD<T>(Xi, yi, P) {
    //LipschitzConst = 0.25; // for logistic loss function only
    twolambda2 = 2 * this->ModelParams[2];
    qp2lamda2 = (LipschitzConst + twolambda2); // this is the univariate lipschitz const of the differentiable objective
    thr = std::sqrt((2 * this->ModelParams[0]) / qp2lamda2);
    lambda1 = this->ModelParams[1];
    lambda1ol = lambda1 / qp2lamda2;
    ExpyXB = arma::exp(*this->y % (*(this->X) * this->B + this->b0)); // Maintained throughout the algorithm
    Xtr = P.Xtr; Iter = P.Iter; 
    this->result.ModelParams = P.ModelParams; 
    Xy = P.Xy;
    NoSelectK = P.NoSelectK;
    Range1p.resize(this->p);
    std::iota(std::begin(Range1p), std::end(Range1p), 0);
    ScreenSize = P.ScreenSize;
}

template <class T>
FitResult<T> CDL012Logistic<T>::Fit() { // always uses active sets
    // arma::mat Xy = X->each_col() % *y; // later
    
    this->B = clamp_by_vector(this->B, this->Lows, this->Highs);
    
    double objective = Objective(this->r, this->B);
    
    std::vector<std::size_t> FullOrder = this->Order; // never used in LR
    this->Order.resize(std::min((int) (this->B.n_nonzero + ScreenSize + NoSelectK), (int)(this->p)));
    
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
            
            // Calculate Partial_i
            const double Biold = this->B[i];
            const double partial_i = - arma::sum( matrix_column_get(*(this->Xy), i) / (1 + ExpyXB) ) + twolambda2 * Biold;
            (*Xtr)[i] = std::abs(partial_i); // abs value of grad
            
            // Ideal value of Bi_new assuming no L0, L1, L2 or bounds.
            const double x = Biold - partial_i / qp2lamda2; 
            
            // Bi with No Bounds (nb); accounting for L1, L2 penalties
            const double Bi_nb = std::copysign(std::abs(x) - lambda1/qp2lamda2, x);
            const double Bi_wb = clamp(Bi_nb, this->Lows[i], this->Highs[i]);  // Bi With Bounds (wb)
            
            // New value that Bi will take
            double new_Bi = Biold;
            
            if (i < NoSelectK){
                // Only penalize by l1 and l2 (NOT L0)
                if (abs(x) < lambda1){
                    new_Bi = 0;
                } else {
                    new_Bi = Bi_wb;
                }
            } else if (std::abs(Bi_nb) < thr){
                // Maximum value of Bi to small to pass L0 threshold => set to 0;
                new_Bi = 0;
            } else {
                // We know Bi_nb >= thr)
                const double delta = std::sqrt(std::pow(std::abs(x) - lambda1/qp2lamda2, 2) 
                                                   - 2*this->ModelParams[0]*qp2lamda2);
                if ((Bi_nb - delta <= Bi_wb) && (Bi_wb <= Bi_nb + delta)){
                    // Bi_wb exists in [Bi_nb - delta, Bi_nb+delta]
                    // Therefore accept Bi_wb
                    new_Bi = Bi_wb;
                } else {
                    new_Bi = 0;
                }
            }
            
            // B[i] changed from Biold to new_Bi, therefore update residual by change.
            if (Biold != new_Bi){
                ExpyXB %= arma::exp( (new_Bi - Biold) * matrix_column_get(*(this->Xy), i));
                this->B[i] = new_Bi;
            }
        }
        
        this->SupportStabilized();
        
        // only way to terminate is by (i) converging on active set and (ii) CWMinCheck
        if (this->Converged() && CWMinCheck()) {
            break;
        }
        
        
        
    }
    
    this->result.Objective = objective;
    this->result.B = this->B;
    this->result.Model = this;
    this->result.intercept = this->b0;
    this->result.ExpyXB = ExpyXB;
    this->result.IterNum = this->CurrentIters;
    
    return this->result;
}

template <class T>
bool CDL012Logistic<T>::CWMinCheck() {
    // Checks for violations outside Supp and updates Order in case of violations
    //std::cout<<"#################"<<std::endl;
    //std::cout<<"In CWMinCheck!!!"<<std::endl;
    // Get the Sc = FullOrder - Order
    std::vector<std::size_t> S;
    
    for (arma::sp_mat::const_iterator it = this->B.begin(); it != this->B.end(); ++it)
        S.push_back(it.row());
    
    std::vector<std::size_t> Sc;
    set_difference(
        Range1p.begin(),
        Range1p.end(),
        S.begin(),
        S.end(),
        back_inserter(Sc));
    
    bool Cwmin = true;
    
    for (auto& i : Sc)  {
        // Calculate Partial_i
        const double partial_i = - arma::sum( matrix_column_get(*(this->Xy), i) / (1 + ExpyXB) );
        (*Xtr)[i] = std::abs(partial_i); // abs value of grad
        
        // B[i] == 0 for all i in Sc.
        const double x = - partial_i / qp2lamda2;
        const double Bi_nb = std::copysign(std::abs(x) - lambda1/qp2lamda2, x);
        const double Bi_wb = clamp(Bi_nb, this->Lows[i], this->Highs[i]);  // Bi With Bounds (wb)
        
        if (std::abs(Bi_nb) >= thr) {
            // We know Bi_nb >= sqrt(thr)
            const double delta = std::sqrt(std::pow(std::abs(x) - lambda1/qp2lamda2, 2) 
                                               - 2*this->ModelParams[0]*qp2lamda2);
            
            if ((Bi_nb - delta <= Bi_wb) && (Bi_wb <= Bi_nb + delta)){
                // Bi_wb exists in [Bi_nb - delta, Bi_nb+delta]
                // Therefore accept Bi_wb
                this->B[i] = Bi_wb;
                this->Order.push_back(i);
                ExpyXB %= arma::exp( Bi_wb * matrix_column_get(*(this->Xy), i));
                Cwmin = false;
            }
        }
    }
    return Cwmin;
}

template class CDL012Logistic<arma::mat>;
template class CDL012Logistic<arma::sp_mat>;

