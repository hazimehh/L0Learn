#ifndef CDL012SquaredHinge_H
#define CDL012SquaredHinge_H
#include "RcppArmadillo.h"
#include "CD.h"
#include "FitResult.h"
#include "utils.h"
#include "Normalize.h" // Remove later
#include "Grid.h" // Remove later

template <typename T>
class CDL012SquaredHinge : public CD<T> {
    private:
        const double LipschitzConst = 2; // for f (without regularization)
        double thr;
        double twolambda2;
        double qp2lamda2;
        double lambda1;
        double lambda1ol;
        std::vector<double> * Xtr;
        std::size_t Iter;
        arma::vec onemyxb;
        std::size_t NoSelectK;
        T * Xy;
        std::size_t ScreenSize;
        std::vector<std::size_t> Range1p;


    public:
        CDL012SquaredHinge(const T& Xi, const arma::vec& yi, const Params<T>& P);
        //~CDL012SquaredHinge(){}
        //inline double Derivativei(std::size_t i);

        //inline double Derivativeb();

        FitResult<T> Fit() final;

        inline double Objective(arma::vec & r, arma::sp_mat & B) final;

        bool CWMinCheck();


};

template <typename T>
CDL012SquaredHinge<T>::CDL012SquaredHinge(const T& Xi, const arma::vec& yi, const Params<T>& P) : CD<T>(Xi, yi, P) {
    twolambda2 = 2 * this->ModelParams[2];
    qp2lamda2 = (LipschitzConst + twolambda2); // this is the univariate lipschitz const of the differentiable objective
    thr = std::sqrt((2 * this->ModelParams[0]) / qp2lamda2);
    lambda1 = this->ModelParams[1];
    lambda1ol = lambda1 / qp2lamda2;
    // b0 = P.b0;
    Xtr = P.Xtr;
    Iter = P.Iter;
    this->result.ModelParams = P.ModelParams;
    onemyxb = 1 - *(this->y) % (*(this->X) * this->B + this->b0); // can be passed from prev solution, but definitely not a bottleneck now
    NoSelectK = P.NoSelectK;
    Xy = P.Xy;
    Range1p.resize(this->p);
    std::iota(std::begin(Range1p), std::end(Range1p), 0);
    ScreenSize = P.ScreenSize;
    // intercept = P.intercept;
}


template <typename T>
FitResult<T> CDL012SquaredHinge<T>::Fit() {
    
    this->B = clamp_by_vector(this->B, this->Lows, this->Highs);
    
    arma::uvec indices = arma::find(onemyxb > 0); // maintained throughout
    
    double objective = Objective(this->r, this->B);
    
    
    std::vector<std::size_t> FullOrder = this->Order; // never used in LR
    this->Order.resize(std::min((int) (this->B.n_nonzero + ScreenSize + NoSelectK), (int)(this->p)));
    
    
    for (auto t = 0; t < this->MaxIters; ++t) {
        //std::cout<<"CDL012 Squared Hinge: "<< t << " " << objective <<std::endl;
        this->Bprev = this->B;
        
        // Update the intercept
        if (this->intercept) {
            double b0old = this->b0;
            double partial_b0 = arma::sum(2 * onemyxb.elem(indices) % (- (this->y)->elem(indices) ) );
            this->b0 -= partial_b0 / (this->n * LipschitzConst); // intercept is not regularized
            onemyxb += *(this->y) * (b0old - this->b0);
            indices = arma::find(onemyxb > 0);
            //std::cout<<"Intercept: "<<b0<<" Obj: "<<Objective(r,B)<<std::endl;
        }
        
        for (auto& i : this->Order) {
            // Calculate Partial_i
            double Biold = this->B[i];
            
            double partial_i = arma::sum(2 * onemyxb.elem(indices) % (- matrix_column_get(*Xy, i).elem(indices))  ) + twolambda2 * Biold;
            
            (*Xtr)[i] = std::abs(partial_i); // abs value of grad
            
            // Ideal value of Bi_new assuming no L0, L1, L2 or bounds.
            double x = Biold - partial_i / qp2lamda2;
            double Bi_nb = std::copysign(std::abs(x) - lambda1/qp2lamda2, x); // Bi with No Bounds (nb)
            double Bi_wb = clamp(Bi_nb, this->Lows[i], this->Highs[i]);  // Bi With Bounds (wb)
            
            if (i < NoSelectK){
                this->B[i] = Bi_wb;
            } else if (Bi_nb < thr){
                // Maximum value of Bi to small to pass L0 threshold => set to 0;
                this->B[i] = 0;
            } else {
                // We know Bi_nb >= thr)
                double delta = std::sqrt(std::pow(std::abs(x) - lambda1/qp2lamda2, 2) - 2*this->ModelParams[0]*qp2lamda2);
                if ((Bi_nb - delta <= Bi_wb) && (Bi_wb <= Bi_nb + delta)){
                    // Bi_wb exists in [Bi_nb - delta, Bi_nb+delta]
                    // Therefore accept Bi_wb
                    this->B[i] = Bi_wb;
                } else {
                    this->B[i] = 0;
                }
            }
            
            onemyxb += (Biold - this->B[i]) * matrix_column_get(*(this->Xy), i);
            indices = arma::find(onemyxb > 0);
            
        }
        
        this->SupportStabilized();
        
        // only way to terminate is by (i) converging on active set and (ii) CWMinCheck
        if (this->Converged()) {
            if (CWMinCheck()) {
                break;
            }
            break;
        }
    }
    
    this->result.Objective = objective;
    this->result.B = this->B;
    this->result.Model = this;
    this->result.intercept = this->b0;
    this->result.IterNum = this->CurrentIters;
    return this->result;
}

template <typename T>
inline double CDL012SquaredHinge<T>::Objective(arma::vec & r, arma::sp_mat & B) {
    
    auto l2norm = arma::norm(this->B, 2);
    //arma::vec onemyxb = 1 - *y % (*X * B + b0);
    arma::uvec indices = arma::find(onemyxb > 0);
    
    return arma::sum(onemyxb.elem(indices) % onemyxb.elem(indices)) + this->ModelParams[0] * B.n_nonzero + this->ModelParams[1] * arma::norm(B, 1) + this->ModelParams[2] * l2norm * l2norm;
}


template <typename T>
bool CDL012SquaredHinge<T>::CWMinCheck(){
    std::vector<std::size_t> S;
    for(arma::sp_mat::const_iterator it = this->B.begin(); it != this->B.end(); ++it)
        S.push_back(it.row());
    
    std::vector<std::size_t> Sc;
    set_difference(
        Range1p.begin(),
        Range1p.end(),
        S.begin(),
        S.end(),
        back_inserter(Sc));
    
    bool Cwmin = true;
    
    arma::uvec indices = arma::find(onemyxb > 0);
    
    for (auto& i : Sc) {
        
        double partial_i = arma::sum(2 * onemyxb.elem(indices) % (- matrix_column_get(*(this->Xy), i).elem(indices)));
        
        (*Xtr)[i] = std::abs(partial_i); // abs value of grad
        
        // Ideal value of Bi_new assuming no L0, L1, L2 or bounds.
        // B[i] == 0 for all i in Sc
        double x = - partial_i / qp2lamda2;
        double Bi_nb = std::copysign(std::abs(x) - lambda1/qp2lamda2, x); // Bi with No Bounds (nb)
        double Bi_wb = clamp(Bi_nb, this->Lows[i], this->Highs[i]);  // Bi With Bounds (wb)
        
        if (std::abs(Bi_nb) >= thr) {
            // We know Bi_nb >= sqrt(thr)
            double delta = std::sqrt(std::pow(std::abs(x) - lambda1/qp2lamda2, 2) - 2*this->ModelParams[0]*qp2lamda2);
            
            if ((Bi_nb - delta <= Bi_wb) && (Bi_wb <= Bi_nb + delta)){
                // Bi_wb exists in [Bi_nb - delta, Bi_nb+delta]
                // Therefore accept Bi_wb
                this->B[i] = Bi_wb;
                this->Order.push_back(i);
                onemyxb -= Bi_wb * matrix_column_get(*(this->Xy), i);
                indices = arma::find(onemyxb > 0);
                Cwmin = false;
            }
        }
    }
    return Cwmin;
}

#endif
