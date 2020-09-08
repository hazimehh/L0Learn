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
        double b0;
        std::vector<double> * Xtr;
        unsigned int Iter;
        arma::vec onemyxb;
        unsigned int NoSelectK;
        T * Xy;
        unsigned int ScreenSize;
        std::vector<unsigned int> Range1p;
        bool intercept;

    public:
        CDL012SquaredHinge(const T& Xi, const arma::vec& yi, const Params<T>& P);
        //~CDL012SquaredHinge(){}
        //inline double Derivativei(unsigned int i);

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
    b0 = P.b0;
    Xtr = P.Xtr;
    Iter = P.Iter;
    this->result.ModelParams = P.ModelParams;
    onemyxb = 1 - *(this->y) % (*(this->X) * this->B + b0); // can be passed from prev solution, but definitely not a bottleneck now
    NoSelectK = P.NoSelectK;
    Xy = P.Xy;
    Range1p.resize(this->p);
    std::iota(std::begin(Range1p), std::end(Range1p), 0);
    ScreenSize = P.ScreenSize;
    intercept = P.intercept;
}


template <typename T>
FitResult<T> CDL012SquaredHinge<T>::Fit() {
    
    this->B = clamp_by_vector(this->B, this->Lows, this->Highs);
    
    arma::uvec indices = arma::find(onemyxb > 0); // maintained throughout
    
    double objective = Objective(this->r, this->B);
    
    
    std::vector<unsigned int> FullOrder = this->Order; // never used in LR
    this->Order.resize(std::min((int) (this->B.n_nonzero + ScreenSize + NoSelectK), (int)(this->p)));
    
    
    for (auto t = 0; t < this->MaxIters; ++t) {
        //std::cout<<"CDL012 Squared Hinge: "<< t << " " << objective <<std::endl;
        this->Bprev = this->B;
        
        // Update the intercept
        if (intercept) {
            double b0old = b0;
            double partial_b0 = arma::sum(2 * onemyxb.elem(indices) % (- (this->y)->elem(indices) ) );
            b0 -= partial_b0 / (this->n * LipschitzConst); // intercept is not regularized
            onemyxb += *(this->y) * (b0old - b0);
            indices = arma::find(onemyxb > 0);
            //std::cout<<"Intercept: "<<b0<<" Obj: "<<Objective(r,B)<<std::endl;
        }
        
        for (auto& i : this->Order) {
            // Calculate Partial_i
            double Biold = this->B[i];
            
            double partial_i = arma::sum(2 * onemyxb.elem(indices) % (- matrix_column_get(*Xy, i).elem(indices))  ) + twolambda2 * Biold;
            
            (*Xtr)[i] = std::abs(partial_i); // abs value of grad
            
            double x = Biold - partial_i / qp2lamda2;
            double z = std::abs(x) - lambda1ol;
            
            
            if (z >= thr || (i < NoSelectK && z>0)) { 	// often false so body is not costly
                double Bnew = clamp(std::copysign(z, x), this->Lows[i], this->Highs[i]);
                this->B[i] = Bnew;
                onemyxb += (Biold - Bnew) * matrix_column_get(*(this->Xy), i);
                indices = arma::find(onemyxb > 0);
                
                //std::cout<<"In. "<<Objective(r,B)<<std::endl;
            } else if (Biold != 0) { // do nothing if x=0 and B[i] = 0
                this->B[i] = 0;
                onemyxb += (Biold) * matrix_column_get(*(this->Xy), i);
                indices = arma::find(onemyxb > 0);
            }
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
    this->result.intercept = b0;
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
    std::vector<unsigned int> S;
    for(arma::sp_mat::const_iterator it = this->B.begin(); it != this->B.end(); ++it)
        S.push_back(it.row());
    
    std::vector<unsigned int> Sc;
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
        
        double x = - partial_i / qp2lamda2;
        double z = std::abs(x) - lambda1ol;
        
        if (z > thr) {	// often false so body is not costly
            double Bnew = std::copysign(z, x);
            this->B[i] = Bnew;
            onemyxb +=  - Bnew * matrix_column_get(*(this->Xy), i);
            indices = arma::find(onemyxb > 0);
            Cwmin = false;
            this->Order.push_back(i);
        }
    }
    return Cwmin;
}

#endif
