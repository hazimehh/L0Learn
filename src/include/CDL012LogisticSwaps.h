#ifndef CDL012LogisticSwaps_H
#define CDL012LogisticSwaps_H
#include <chrono>
#include "RcppArmadillo.h"
#include "CD.h"
#include "FitResult.h"
#include "CDL012Logistic.h"
#include "Normalize.h" // Remove later
#include "utils.h"

template <typename T>
class CDL012LogisticSwaps : public CD<T>
{
    private:
        const double LipschitzConst = 0.25;
        double thr;
        double twolambda2;
        double qp2lamda2;
        double lambda1;
        double lambda1ol;
        double b0;
        double stl0Lc;
        arma::vec ExpyXB;
        std::vector<double> * Xtr;
        arma::mat * Xy;
        unsigned int Iter;

        unsigned int MaxNumSwaps;
        Params<T> P;
        unsigned int NoSelectK;

    public:
        CDL012LogisticSwaps(const T& Xi, const arma::vec& yi, const Params<T>& P);

        FitResult<T> Fit() final;

        inline double Objective(arma::vec & r, arma::sp_mat & B) final;


};

template <typename T>
CDL012LogisticSwaps<T>::CDL012LogisticSwaps(const T& Xi, const arma::vec& yi, const Params<T>& Pi) : CD<T>(Xi, yi, Pi)
{
    MaxNumSwaps = Pi.MaxNumSwaps;
    P = Pi;
    twolambda2 = 2 * this->ModelParams[2];
    qp2lamda2 = (LipschitzConst + twolambda2); // this is the univariate lipschitz const of the differentiable objective
    thr = std::sqrt((2 * this->ModelParams[0]) / qp2lamda2);
    stl0Lc = std::sqrt((2 * this->ModelParams[0]) * qp2lamda2);
    lambda1 = this->ModelParams[1];
    lambda1ol = lambda1 / qp2lamda2;
    Xtr = P.Xtr; Iter = P.Iter;
    this->result.ModelParams = P.ModelParams;
    Xy = P.Xy;
    NoSelectK = P.NoSelectK;
}

template <typename T>
FitResult<T> CDL012LogisticSwaps<T>::Fit()
{
    auto result = CDL012Logistic<T>(*(this->X), *(this->y), P).Fit(); // result will be maintained till the end
    b0 = result.intercept; // Initialize from previous later....!
    this->B = result.B;
    ExpyXB = result.ExpyXB; // Maintained throughout the algorithm
    
    double objective = result.Objective;
    double Fmin = objective;
    unsigned int maxindex;
    double Bmaxindex;
    
    P.Init = 'u';
    
    
    
    bool foundbetter;
    for (unsigned int t = 0; t < MaxNumSwaps; ++t)
    {
        //std::cout<<"1Swaps Iteration: "<<t<<". "<<"Obj: "<<objective<<std::endl;
        //B.print();
        arma::sp_mat::const_iterator start = this->B.begin();
        arma::sp_mat::const_iterator end   = this->B.end();
        std::vector<unsigned int> NnzIndices;
        for(arma::sp_mat::const_iterator it = start; it != end; ++it)
        {
            if (it.row() >= NoSelectK){NnzIndices.push_back(it.row());}
        }
        // Can easily shuffle here...
        //std::shuffle(std::begin(Order), std::end(Order), engine);
        foundbetter = false;
        
        
        for (auto& j : NnzIndices)
        {
            
            // Remove j
            arma::vec ExpyXBnoj = ExpyXB % arma::exp( - this->B[j] * matrix_column_get(*(this->Xy), j));
            
            //auto start1 = std::chrono::high_resolution_clock::now();
            ///
            arma::rowvec gradient = - arma::sum( Xy->each_col() / (1 + ExpyXBnoj) , 0); // + twolambda2 * Biold // sum column-wise
            arma::uvec indices = arma::sort_index(arma::abs(gradient),"descend");
            bool foundbetteri = false;
            ///
            //auto end1 = std::chrono::high_resolution_clock::now();
            //std::cout<<"grad computation:  "<<std::chrono::duration_cast<std::chrono::milliseconds>(end1-start1).count() << " ms " << std::endl;
            
            
            //auto start2 = std::chrono::high_resolution_clock::now();
            // Later: make sure this scans at least 100 coordinates from outside supp (now it does not)
            for(unsigned int ll = 0; ll < std::min(100, (int) this->p); ++ll)
            {
                unsigned int i = indices(ll);
                if(this->B[i] == 0 && i >= NoSelectK)
                {
                    arma::vec ExpyXBnoji = ExpyXBnoj;
                    
                    double Biold = 0;
                    double Binew;
                    //double partial_i = - arma::sum( (Xy->unsafe_col(i)) / (1 + ExpyXBnoji) ); // + twolambda2 * Biold
                    double partial_i = gradient[i];
                    bool Converged = false;
                    
                    arma::sp_mat Btemp = this->B;
                    Btemp[j] = 0;
                    double ObjTemp = Objective(ExpyXBnoji, Btemp);
                    unsigned int innerindex = 0;
                    
                    double x = Biold - partial_i/qp2lamda2;
                    double z = std::abs(x) - lambda1ol;
                    Binew = std::copysign(z, x); // no need to check if >= sqrt(2lambda_0/Lc)
                    
                    while(!Converged && innerindex < 20  && ObjTemp >= Fmin) // ObjTemp >= Fmin
                    {
                        ExpyXBnoji %= arma::exp( (Binew - Biold) *  matrix_column_get(*Xy, i));
                        partial_i = - arma::sum( matrix_column_get(*Xy, i) / (1 + ExpyXBnoji) ) + twolambda2 * Binew;
                        
                        if (std::abs((Binew - Biold)/Biold) < 0.0001){
                            Converged = true;
                            //std::cout<<"swaps converged!!!"<<std::endl;
                            
                        }
                        
                        Biold = Binew;
                        x = Biold - partial_i/qp2lamda2;
                        z = std::abs(x) - lambda1ol;
                        Binew = std::copysign(z, x); // no need to check if >= sqrt(2lambda_0/Lc)
                        innerindex += 1;
                    }
                    
                    
                    Btemp[i] = Binew;
                    ObjTemp = Objective(ExpyXBnoji, Btemp);
                    
                    if (ObjTemp < Fmin)
                    {
                        Fmin = ObjTemp;
                        maxindex = i;
                        Bmaxindex = Binew;
                        foundbetteri = true;
                    }
                    
                    // Can be made much faster (later)
                    Btemp[i] = Binew;
                    
                    //}
                    
                    
                    
                }
                
                if (foundbetteri == true)
                {
                    this->B[j] = 0;
                    this->B[maxindex] = Bmaxindex;
                    P.InitialSol = &(this->B);
                    P.b0 = b0;
                    result = CDL012Logistic<T>(*(this->X), *(this->y), P).Fit();
                    ExpyXB = result.ExpyXB;
                    this->B = result.B;
                    b0 = result.intercept;
                    objective = result.Objective;
                    Fmin = objective;
                    foundbetter = true;
                    break;
                }
            }
            
            //auto end2 = std::chrono::high_resolution_clock::now();
            //std::cout<<"restricted:  "<<std::chrono::duration_cast<std::chrono::milliseconds>(end2-start2).count() << " ms " << std::endl;
            
            //if (foundbetter){break;}
            
        }
        
        if(!foundbetter)
        {
            //result.Model = this;
            return result;
        }
    }
    
    //result.Model = this;
    return result;
}

template <typename T>
inline double CDL012LogisticSwaps<T>::Objective(arma::vec & r, arma::sp_mat & B)   // hint inline
{
    auto l2norm = arma::norm(B, 2);
    return arma::sum(arma::log(1 + 1 / r)) + this->ModelParams[0] * B.n_nonzero + this->ModelParams[1] * arma::norm(B, 1) + this->ModelParams[2] * l2norm * l2norm;
}

/*
 int main(){
 
 
 Params P;
 P.ModelType = "L012LogisticSwaps";
 P.ModelParams = std::vector<double>{5,0,0.01};
 P.ActiveSet = true;
 P.ActiveSetNum = 6;
 P.Init = 'r';
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
 
 
 auto model = CDL012LogisticSwaps(Xscaled, yscaled, P);
 auto result = model.Fit();
 
 arma::sp_mat B_unscaled;
 double intercept;
 std::tie(B_unscaled, intercept) = DeNormalize(result.B, BetaMultiplier, meanX, meany);
 
 result.B.print();
 B_unscaled.print();
 
 
 return 0;
 
 }
 */




#endif
