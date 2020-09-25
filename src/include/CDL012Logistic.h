#ifndef CDL012Logistic_H
#define CDL012Logistic_H
#include "RcppArmadillo.h"
#include "CD.h"
#include "FitResult.h"
#include "Normalize.h" // Remove later
#include "Grid.h" // Remove later
#include "utils.h"

template <typename T>
class CDL012Logistic : public CD<T> {
    private:
        const double LipschitzConst = 0.25;
        double thr;
        double twolambda2;
        double qp2lamda2;
        double lambda1;
        double lambda1ol;
        double b0 = 0;
        arma::vec ExpyXB;
        std::vector<double> * Xtr;
        T * Xy;
        std::size_t Iter;
        std::size_t NoSelectK;
        std::size_t ScreenSize;
        std::vector<std::size_t> Range1p;
        bool intercept;
    public:
        CDL012Logistic(const T& Xi, const arma::vec& yi, const Params<T>& P);
        //~CDL012Logistic(){}
        FitResult<T> Fit() final;

        inline double Objective(arma::vec & r, arma::sp_mat & B) final;

        bool CWMinCheck();

};

template <typename T>
CDL012Logistic<T>::CDL012Logistic(const T& Xi, const arma::vec& yi, const Params<T>& P) : CD<T>(Xi, yi, P) {
    //LipschitzConst = 0.25; // for logistic loss function only
    twolambda2 = 2 * this->ModelParams[2];
    qp2lamda2 = (LipschitzConst + twolambda2); // this is the univariate lipschitz const of the differentiable objective
    thr = std::sqrt((2 * this->ModelParams[0]) / qp2lamda2);
    lambda1 = this->ModelParams[1];
    lambda1ol = lambda1 / qp2lamda2;
    b0 = P.b0;
    ExpyXB = arma::exp(*this->y % (*(this->X) * this->B + b0)); // Maintained throughout the algorithm
    Xtr = P.Xtr; Iter = P.Iter; 
    this->result.ModelParams = P.ModelParams; 
    Xy = P.Xy;
    NoSelectK = P.NoSelectK;
    Range1p.resize(this->p);
    std::iota(std::begin(Range1p), std::end(Range1p), 0);
    ScreenSize = P.ScreenSize;
    intercept = P.intercept;
}

template <typename T>
inline double CDL012Logistic<T>::Objective(arma::vec & r, arma::sp_mat & B) {  // hint inline
    auto l2norm = arma::norm(B, 2);
    // arma::sum(arma::log(1 + 1 / ExpyXB)) is the negative log-likelihood
    return arma::sum(arma::log(1 + 1 / ExpyXB)) + this->ModelParams[0] * B.n_nonzero + this->ModelParams[1] * arma::norm(B, 1) + this->ModelParams[2] * l2norm * l2norm;
}

template <typename T>
FitResult<T> CDL012Logistic<T>::Fit() { // always uses active sets
    // arma::mat Xy = X->each_col() % *y; // later
    
    this->B = clamp_by_vector(this->B, this->Lows, this->Highs);
    
    double objective = Objective(this->r, this->B);
    
    std::vector<std::size_t> FullOrder = this->Order; // never used in LR
    this->Order.resize(std::min((int) (this->B.n_nonzero + ScreenSize + NoSelectK), (int)(this->p)));
    
    for (std::size_t t = 0; t < this->MaxIters; ++t) {
        this->Bprev = this->B;
        
        // Update the intercept
        if (intercept){
            double b0old = b0;
            double partial_b0 = - arma::sum( *(this->y) / (1 + ExpyXB) );
            b0 -= partial_b0 / (this->n * LipschitzConst); // intercept is not regularized or bounded 
            ExpyXB %= arma::exp( (b0 - b0old) * *(this->y));
        }
        
        for (auto& i : this->Order) {
            
            // Calculate Partial_i
            double Biold = this->B[i];
            double partial_i = - arma::sum( matrix_column_get(*(this->Xy), i) / (1 + ExpyXB) ) + twolambda2 * Biold;
            (*Xtr)[i] = std::abs(partial_i); // abs value of grad
            
            // Ideal value of Bi_new assuming no L0, L1, L2 or bounds.
            double x = Biold - partial_i / qp2lamda2; 
            
            // Bi with No Bounds (nb); accounting for L1, L2 penalties
            double Bi_nb = std::copysign(std::abs(x) - lambda1/qp2lamda2, x);
            double Bi_wb = clamp(Bi_nb, this->Lows[i], this->Highs[i]);  // Bi With Bounds (wb)
            double delta;
            
            if (i < NoSelectK){
                this->B[i] = Bi_wb;
            } else if (Bi_nb < thr){
                // Maximum value of Bi to small to pass L0 threshold => set to 0;
                this->B[i] = 0;
            } else {
                // We know Bi_nb >= thr)
                delta = std::sqrt(std::pow(std::abs(Bi_wb) - lambda1/qp2lamda2, 2) - 2*this->ModelParams[0]*qp2lamda2);
                if ((Bi_nb - delta <= Bi_wb) && (Bi_wb <= Bi_nb + delta)){
                    // Bi_wb exists in [Bi_nb - delta, Bi_nb+delta]
                    // Therefore accept Bi_wb
                    this->B[i] = Bi_wb;
                } else {
                    this->B[i] = 0;
                }
            }
            
            // B changed from Bi to this->B[i], therefore update residual by change.
            ExpyXB %= arma::exp( (this->B[i] - Biold) * matrix_column_get(*(this->Xy), i));
            }
        }
        
        this->SupportStabilized();
        
        // only way to terminate is by (i) converging on active set and (ii) CWMinCheck
        if (this->Converged()) {
            if (CWMinCheck()) {
                break;
            }
        }
        
        
    }
    
    this->result.Objective = objective;
    this->result.B = this->B;
    this->result.Model = this;
    this->result.intercept = b0;
    this->result.ExpyXB = ExpyXB;
    this->result.IterNum = this->CurrentIters;
    
    return this->result;
}

template <typename T>
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
        double partial_i = - arma::sum( matrix_column_get(*(this->Xy), i) / (1 + ExpyXB) );
        (*Xtr)[i] = std::abs(partial_i); // abs value of grad
        double x = - partial_i / qp2lamda2;
        double z = clamp(std::copysign(std::abs(x) - lambda1ol, x),
                         this->Lows[i], this->Highs[i]);
        
        
        if (z >= thr || z <= -thr) {	// often false so body is not costly
            double Bnew = z;
            this->B[i] = Bnew;
            ExpyXB %= arma::exp( Bnew * matrix_column_get(*(this->Xy), i));
            Cwmin = false;
            this->Order.push_back(i);
            //std::cout<<"Found Violation !!!"<<std::endl;
        }
    }
    //std::cout<<"#################"<<std::endl;
    return Cwmin;
}


#endif
