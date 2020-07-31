#ifndef CDL012Logistic_H
#define CDL012Logistic_H
#include "RcppArmadillo.h"
#include "CD.h"
#include "FitResult.h"
#include "Normalize.h" // Remove later
#include "Grid.h" // Remove later
#include "utils.h"

template <typename T>
class CDL012Logistic : public CD<T>
{
    private:
        const double LipschitzConst = 0.25;
        double thr;
        double twolambda2;
        double qp2lamda2;
        double lambda1;
        double lambda1ol;
        double b0;
        arma::vec ExpyXB;
        std::vector<double> * Xtr;
        T * Xy;
        unsigned int Iter;
        unsigned int NoSelectK;
        unsigned int ScreenSize;
        std::vector<unsigned int> Range1p;
        bool intercept;
    public:
        CDL012Logistic(const T& Xi, const arma::vec& yi, const Params<T>& P);
        //~CDL012Logistic(){}
        FitResult<T> Fit() final;

        inline double Objective(arma::vec & r, arma::sp_mat & B) final;

        bool CWMinCheck();

};

template <typename T>
inline double CDL012Logistic<T>::Objective(arma::vec & r, arma::sp_mat & B)   // hint inline
{
    auto l2norm = arma::norm(B, 2);
    // arma::sum(arma::log(1 + 1 / ExpyXB)) is the negative log-likelihood
    return arma::sum(arma::log(1 + 1 / ExpyXB)) + this->ModelParams[0] * B.n_nonzero + this->ModelParams[1] * arma::norm(B, 1) + this->ModelParams[2] * l2norm * l2norm;
}

template <typename T>
CDL012Logistic<T>::CDL012Logistic(const T& Xi, const arma::vec& yi, const Params<T>& P) : CD<T>(Xi, yi, P)
{
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
FitResult<T> CDL012Logistic<T>::Fit() // always uses active sets
{
    // arma::mat Xy = X->each_col() % *y; // later
    
    double objective = Objective(this->r, this->B);
    
    std::vector<unsigned int> FullOrder = this->Order; // never used in LR
    this->Order.resize(std::min((int) (this->B.n_nonzero + ScreenSize + NoSelectK), (int)(this->p)));
    
    for (unsigned int t = 0; t < this->MaxIters; ++t)
    {
        //std::cout<<"CDL012 Logistic: "<< t << " " << objective<<std::endl;
        this->Bprev = this->B;
        
        // Update the intercept
        if (intercept){
            double b0old = b0;
            double partial_b0 = - arma::sum( *(this->y) / (1 + ExpyXB) );
            b0 -= partial_b0 / (this->n * LipschitzConst); // intercept is not regularized
            ExpyXB %= arma::exp( (b0 - b0old) * *(this->y));
        }
        
        //std::cout<<"Intercept. "<<Objective(r,B)<<std::endl;
        
        
        for (auto& i : this->Order)
        {
            
            // Calculate Partial_i
            double Biold = this->B[i];
            double partial_i = - arma::sum( matrix_column_get(*(this->Xy), i) / (1 + ExpyXB) ) + twolambda2 * Biold;
            (*Xtr)[i] = std::abs(partial_i); // abs value of grad
            
            double x = Biold - partial_i / qp2lamda2;
            double z = std::abs(x) - lambda1ol;
            
            
            if (z >= thr || (i < NoSelectK && z>0) ) 	// often false so body is not costly
            {
                double Bnew = std::copysign(z, x);
                this->B[i] = Bnew;
                ExpyXB %= arma::exp( (Bnew - Biold) * matrix_column_get(*(this->Xy), i));
                //std::cout<<"In. "<<Objective(r,B)<<std::endl;
            }
            
            else if (Biold != 0)   // do nothing if x=0 and B[i] = 0
            {
                ExpyXB %= arma::exp( - Biold * matrix_column_get(*(this->Xy), i));
                this->B[i] = 0;
            }
        }
        
        this->SupportStabilized();
        
        // only way to terminate is by (i) converging on active set and (ii) CWMinCheck
        if (this->Converged())
        {
            if (CWMinCheck()) {break;}
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
bool CDL012Logistic<T>::CWMinCheck()
    // Checks for violations outside Supp and updates Order in case of violations
{
    //std::cout<<"#################"<<std::endl;
    //std::cout<<"In CWMinCheck!!!"<<std::endl;
    // Get the Sc = FullOrder - Order
    std::vector<unsigned int> S;
    for(arma::sp_mat::const_iterator it = this->B.begin(); it != this->B.end(); ++it) {S.push_back(it.row());}
    
    std::vector<unsigned int> Sc;
    set_difference(
        Range1p.begin(),
        Range1p.end(),
        S.begin(),
        S.end(),
        back_inserter(Sc));
    
    bool Cwmin = true;
    
    for (auto& i : Sc)
    {
        // Calculate Partial_i
        double partial_i = - arma::sum( matrix_column_get(*(this->Xy), i) / (1 + ExpyXB) );
        (*Xtr)[i] = std::abs(partial_i); // abs value of grad
        double x = - partial_i / qp2lamda2;
        double z = std::abs(x) - lambda1ol;
        if (z > thr) 	// often false so body is not costly
        {
            double Bnew = std::copysign(z, x);
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




/*
 int main(){
 
 
 Params P;
 P.ModelType = "L012Logistic";
 P.ModelParams = std::vector<double>{18.36,0,0.01};
 P.ActiveSet = true;
 P.ActiveSetNum = 6;
 P.Init = 'z';
 P.MaxIters = 200;
 //P.RandomStartSize = 100;
 
 
 arma::mat X;
 X.load("X.csv");
 
 arma::vec y;
 y.load("y.csv");
 
 std::vector<double> * Xtr = new std::vector<double>(X.n_cols);
 P.Xtr = Xtr;
 
 arma::mat Xscaled;
 arma::vec yscaled;
 
 arma::vec BetaMultiplier;
 arma::vec meanX;
 double meany;
 
 std::tie(BetaMultiplier, meanX, meany) = Normalize(X,y, Xscaled, yscaled, false);
 
 
 auto model = CDL012Logistic(Xscaled, yscaled, P);
 auto result = model.Fit();
 
 arma::sp_mat B_unscaled;
 double intercept;
 std::tie(B_unscaled, intercept) = DeNormalize(result.B, BetaMultiplier, meanX, meany);
 
 result.B.print();
 B_unscaled.print();
 
 
 //std::cout<<result.intercept<<std::endl;
 //arma::sign(Xscaled*result.B + result.intercept).print();
 //std::cout<<"#############"<<std::endl;
 //arma::sign(X*B_unscaled + result.intercept).print();
 
 
 
 // GridParams PG;
 // PG.Type = "L0Logistic";
 // PG.NnzStopNum = 12;
 //
 // arma::mat X;
 // X.load("X.csv");
 //
 // arma::vec y;
 // y.load("y.csv");
 // auto g = Grid(X, y, PG);
 //
 // g.Fit();
 //
 // unsigned int i = g.NnzCount.size();
 // for (uint j=0;j<i;++j){
 // 	std::cout<<g.NnzCount[j]<<"   "<<g.Lambda0[j]<<std::endl;
 // }
 
 
 
 return 0;
 
 }
 */


#endif
