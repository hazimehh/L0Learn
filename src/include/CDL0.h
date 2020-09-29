#ifndef CDL0_H
#define CDL0_H
#include "RcppArmadillo.h"
#include "CD.h"
#include "FitResult.h" // REMOVE: may need to include params??
#include "utils.h"


template <typename T>
class CDL0: public CD<T> {
    private:
        double thr;
        std::vector<double> * Xtr;
        std::size_t Iter;
        std::size_t ScreenSize;
        std::vector<std::size_t> Range1p;
        std::size_t NoSelectK;
    public:
        CDL0(const T& Xi, const arma::vec& yi, const Params<T>& P);
        //~CDL0(){}

        FitResult<T> Fit() final;

        double Objective(arma::vec & r, arma::sp_mat & B) final;

        bool CWMinCheck();
};

template <typename T>
CDL0<T>::CDL0(const T& Xi, const arma::vec& yi, const Params<T>& P) : CD<T>(Xi, yi, P){
    thr = sqrt(2 * this->ModelParams[0]); 
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
FitResult<T> CDL0<T>::Fit() {
    
    this->B = clamp_by_vector(this->B, this->Lows, this->Highs);
    
    //bool SecondPass = false;
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
            const double Bi_nb = cor + Bi; // Bi with No Bounds (nb);
            const double Bi_wb = clamp(Bi_nb, this->Lows[i], this->Highs[i]);  // Bi With Bounds (wb)
           
           
            // New value that Bi will take
            double new_Bi = Bi;
            
            /* 2 Cases:
             *     1. Set Bi to 0
             *     2. Set Bi to NNZ
             */
            
            if (i < NoSelectK){
                new_Bi = Bi_wb;
            } else if (std::abs(Bi_nb) < thr){
                // Maximum value of Bi to small to pass L0 threshold => set to 0;
                new_Bi = 0;
            } else {
                // We know Bi_nb >= sqrt(thr)
                const double delta = std::sqrt(Bi_nb*Bi_nb - thr*thr);
               
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
        
        if (this->Converged()) {
            if(FirstRestrictedPass && ActiveSetInitial) {
                if (CWMinCheck()) {
                    break;
                }
                FirstRestrictedPass = false;
                this->Stabilized = false;
                this->ActiveSet = true;
                this->Order = FullOrder;
            } else {
                if (this->Stabilized == true && ActiveSetInitial) { // && !SecondPass
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
    
    // Re-optimize b0 after convergence.
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
bool CDL0<T>::CWMinCheck() {
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
        
        const double Bi_nb = matrix_column_dot(*(this->X), i, this->r);
        const double Bi_wb = clamp(Bi_nb, this->Lows[i], this->Highs[i]);
            
        if (std::abs(Bi_nb) >= thr) {
            // We know Bi_nb >= sqrt(thr)
            const double delta = std::sqrt(Bi_wb*Bi_wb - thr*thr);
            
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
inline double CDL0<T>::Objective(arma::vec & r, arma::sp_mat & B)   // hint inline
{
    return 0.5 * arma::dot(r, r) + this->ModelParams[0] * B.n_nonzero;
}

#endif
