#ifndef CDL012_H
#define CDL012_H
#include "RcppArmadillo.h"
#include "CD.h"
#include "FitResult.h"
#include "utils.h"

template <typename T>
class CDL012 : public CD<T> {
    private:
        double thr;
        double Onep2lamda2;
        double lambda1;
        std::vector<double> * Xtr;
        std::size_t Iter;
        std::size_t ScreenSize;
        std::vector<std::size_t> Range1p;
        std::size_t NoSelectK;
    public:
        CDL012(const T& Xi, const arma::vec& yi, const Params<T>& P);
        //~CDL012(){}

        FitResult<T> Fit() final;

        inline double Objective(arma::vec & r, arma::sp_mat & B) final;

        bool CWMinCheck();

};

template <typename T>
CDL012<T>::CDL012(const T& Xi, const arma::vec& yi, const Params<T>& P) : CD<T>(Xi, yi, P) {
    Onep2lamda2 = 1 + 2 * this->ModelParams[2]; 
    thr = std::sqrt((2 * this->ModelParams[0]) / Onep2lamda2);
    lambda1 = this->ModelParams[1]; 
    Xtr = P.Xtr; 
    Iter = P.Iter; 
    this->result.ModelParams = P.ModelParams; 
    ScreenSize = P.ScreenSize; 
    this->r = *P.r;
    Range1p.resize(this->p); 
    std::iota(std::begin(Range1p), std::end(Range1p), 0); 
    NoSelectK = P.NoSelectK;
    this->result.r = P.r;
}

template <typename T>
FitResult<T> CDL012<T>::Fit() {
    
    this->B = clamp_by_vector(this->B, this->Lows, this->Highs);
    
    double objective = Objective(this->r, this->B);
    
    std::vector<std::size_t> FullOrder = this->Order;
    bool FirstRestrictedPass = true;
    if (this->ActiveSet) {
        this->Order.resize(std::min((int) (this->B.n_nonzero + ScreenSize + NoSelectK), (int)(this->p))); // std::min(1000,Order.size())
    }
    
    bool ActiveSetInitial = this->ActiveSet;
    
    for (std::size_t t = 0; t < this->MaxIters; ++t) {
        this->Bprev = this->B;
        
        if (this->isSparse && this->intercept){
            const double new_b0 = arma::mean(this->r);
            this->r += this->b0 - new_b0;
            this->b0 = new_b0;
        }
        
        for (auto& i : this->Order) {
            const double cor = matrix_column_dot(*(this->X), i, this->r);
            
            (*Xtr)[i] = std::abs(cor); // do abs here instead from when sorting
            
            const double Bi = this->B[i]; // old Bi to adjust residuals if Bi updates
            const double x = cor + Bi;
            const double Bi_nb = std::copysign((std::abs(x) - lambda1) / Onep2lamda2, x); // Bi with No Bounds (nb);
            const double Bi_wb = clamp(Bi_nb, this->Lows[i], this->Highs[i]);  // Bi With Bounds (wb)
            
            // New value that Bi will take
            double new_Bi = Bi;
            
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
                const double delta = std::sqrt(std::pow(std::abs(x) - lambda1, 2) 
                                                   - 2*this->ModelParams[0]*Onep2lamda2) / Onep2lamda2;
                
                if ((Bi_nb - delta <= Bi_wb) && (Bi_wb <= Bi_nb + delta)){
                    // Bi_wb exists in [Bi_nb - delta, Bi_nb+delta]
                    // Therefore accept Bi_wb
                    new_Bi = Bi_wb;
                } else {
                    new_Bi = 0;
                }
            }
            
            // B[i] changed from Bi to new_Bi, therefore update residual by change.
            if (Bi != new_Bi){
                this->r += matrix_column_mult(*(this->X), i, Bi - new_Bi);
                this->B[i] = new_Bi;
            }
        }
            
        //B.print();
        if (this->Converged()) {
            if(FirstRestrictedPass && ActiveSetInitial) {
                if (CWMinCheck()) {
                    break;
                }
                FirstRestrictedPass = false;
                this->Order = FullOrder;
                this->Stabilized = false;
                this->ActiveSet = true;
            } else {
                if (this->Stabilized && ActiveSetInitial) { // && !SecondPass
                    if (CWMinCheck()) {
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
    
    if (this->isSparse && this->intercept){
        const double new_b0 = arma::mean(this->r);
        this->r += this->b0 - new_b0;
        this->b0 = new_b0;
    }
    
    this->result.Objective = objective;
    this->result.B = this->B;
    *(this->result.r) = this->r; // change to pointer later
    this->result.IterNum = this->CurrentIters;
    this->result.intercept = this->b0;
    return this->result;
}

template <typename T>
bool CDL012<T>::CWMinCheck(){
    // Get the Sc = FullOrder - Order
    std::vector<std::size_t> S;
    for(arma::sp_mat::const_iterator it = this->B.begin(); it != this->B.end(); ++it) {
        S.push_back(it.row());
    }
    
    std::vector<std::size_t> Sc;
    set_difference(
        Range1p.begin(),
        Range1p.end(),
        S.begin(),
        S.end(),
        back_inserter(Sc));
    
    bool Cwmin = true;
    for (auto& i : Sc) {
        // B[i] == 0 for all i in Sc
    
        const double x = matrix_column_dot(*(this->X), i, this->r);
        const double absx = std::abs(x);
        (*Xtr)[i] = absx; // do abs here instead from when sorting
        
        const double Bi_nb = std::copysign((std::abs(x) - lambda1) / Onep2lamda2, x); // Bi with No Bounds (nb);
        const double Bi_wb = clamp(Bi_nb, this->Lows[i], this->Highs[i]);  // Bi With Bounds (wb)
        
        if (std::abs(Bi_nb) >= thr) {
            // We know Bi_nb >= sqrt(thr)
            const double delta = std::sqrt(std::pow(std::abs(x) - lambda1, 2) 
                                               - 2*this->ModelParams[0]*Onep2lamda2)/Onep2lamda2;
            
            if ((Bi_nb - delta <= Bi_wb) && (Bi_wb <= Bi_nb + delta)){
                // Bi_wb exists in [Bi_nb - delta, Bi_nb+delta]
                // Therefore accept Bi_wb
                this->B[i] = Bi_wb;
                this->r -= matrix_column_mult(*(this->X), i, Bi_wb);
                Cwmin = false;
            }
        }
    }
    return Cwmin;
    
}


template <typename T>
inline double CDL012<T>::Objective(arma::vec & r, arma::sp_mat & B) {   // hint inline
    auto l2norm = arma::norm(B, 2);
    return 0.5 * arma::dot(r, r) + this->ModelParams[0] * B.n_nonzero + this->ModelParams[1] * arma::norm(B, 1) + this->ModelParams[2] * l2norm * l2norm;
}

#endif
