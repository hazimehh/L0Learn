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
class CDL012LogisticSwaps : public CD<T> {
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
        T * Xy;
        std::size_t Iter;

        std::size_t MaxNumSwaps;
        Params<T> P;
        std::size_t NoSelectK;

    public:
        CDL012LogisticSwaps(const T& Xi, const arma::vec& yi, const Params<T>& P);

        FitResult<T> Fit() final;

        inline double Objective(arma::vec & r, arma::sp_mat & B) final;

};

template <typename T>
CDL012LogisticSwaps<T>::CDL012LogisticSwaps(const T& Xi, const arma::vec& yi, const Params<T>& Pi) : CD<T>(Xi, yi, Pi) {
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
FitResult<T> CDL012LogisticSwaps<T>::Fit() {
    auto result = CDL012Logistic<T>(*(this->X), *(this->y), P).Fit(); // result will be maintained till the end
    b0 = result.intercept; // Initialize from previous later....!
    this->B = result.B;
    ExpyXB = result.ExpyXB; // Maintained throughout the algorithm
    
    double objective = result.Objective;
    double Fmin = objective;
    std::size_t maxindex;
    double Bmaxindex;
    
    P.Init = 'u';
    
    bool foundbetter;
    bool foundbetter_i;
    
    for (std::size_t t = 0; t < MaxNumSwaps; ++t) {
        
        arma::sp_mat::const_iterator start = this->B.begin();
        arma::sp_mat::const_iterator end   = this->B.end();
        std::vector<std::size_t> NnzIndices;
        for(arma::sp_mat::const_iterator it = start; it != end; ++it) {
            if (it.row() >= NoSelectK) {
                NnzIndices.push_back(it.row());
            }
        }
        // Can easily shuffle here...
        //std::shuffle(std::begin(Order), std::end(Order), engine);
        foundbetter = false;
        
        for (auto& j : NnzIndices) {
        
            // Remove j
            arma::vec ExpyXBnoj = ExpyXB % arma::exp( - this->B[j] * matrix_column_get(*(this->Xy), j));
            
            //auto start1 = std::chrono::high_resolution_clock::now();
            ///
            //arma::rowvec gradient = - arma::sum( Xy->each_col() / (1 + ExpyXBnoj) , 0); // + twolambda2 * Biold // sum column-wise
            arma::vec temp_gradient = 1 + ExpyXBnoj;
            T divided_matrix = matrix_vector_divide(*Xy, temp_gradient);
            arma::rowvec gradient = - matrix_column_sums(divided_matrix);
            arma::uvec indices = arma::sort_index(arma::abs(gradient), "descend");
            foundbetter_i = false;
            ///
            //auto end1 = std::chrono::high_resolution_clock::now();
            //std::cout<<"grad computation:  "<<std::chrono::duration_cast<std::chrono::milliseconds>(end1-start1).count() << " ms " << std::endl;
            
            
            //auto start2 = std::chrono::high_resolution_clock::now();
            // Later: make sure this scans at least 100 coordinates from outside supp (now it does not)
            for(std::size_t ll = 0; ll < std::min(100, (int) this->p); ++ll) {
                std::size_t i = indices(ll);
                
                if(this->B[i] == 0 && i >= NoSelectK) {
                    arma::vec ExpyXBnoji = ExpyXBnoj;
                    
                    double Biold = 0;
                    double Binew;
                    //double partial_i = - arma::sum( (Xy->unsafe_col(i)) / (1 + ExpyXBnoji) ); // + twolambda2 * Biold
                    double partial_i = gradient[i];
                    bool converged = false;
                    
                    arma::sp_mat Btemp = this->B;
                    Btemp[j] = 0;
                    double ObjTemp = Objective(ExpyXBnoji, Btemp);
                    std::size_t innerindex = 0;
                    
                    double x = Biold - partial_i/qp2lamda2;
                    double z = std::abs(x) - lambda1ol;
                    Binew = clamp(std::copysign(z, x), this->Lows[i], this->Highs[i]); // no need to check if >= sqrt(2lambda_0/Lc)
                    
                    while(!converged && innerindex < 20  && ObjTemp >= Fmin) { // ObjTemp >= Fmin
                        ExpyXBnoji %= arma::exp( (Binew - Biold) *  matrix_column_get(*Xy, i));
                        partial_i = - arma::sum( matrix_column_get(*Xy, i) / (1 + ExpyXBnoji) ) + twolambda2 * Binew;
                        
                        if (std::abs((Binew - Biold)/Biold) < 0.0001) {
                            converged = true;
                            //std::cout<<"swaps converged!!!"<<std::endl;
                        }
                        
                        Biold = Binew;
                        x = Biold - partial_i/qp2lamda2;
                        z = std::abs(x) - lambda1ol;
                        Binew = clamp(std::copysign(z, x), this->Lows[i], this->Highs[i]); // no need to check if >= sqrt(2lambda_0/Lc)
                        innerindex += 1;
                    }
                    
                    
                    Btemp[i] = Binew;
                    ObjTemp = Objective(ExpyXBnoji, Btemp);
                    
                    if (ObjTemp < Fmin) {
                        Fmin = ObjTemp;
                        maxindex = i;
                        Bmaxindex = Binew;
                        foundbetter_i = true;
                    }
                    
                    // Can be made much faster (later)
                    Btemp[i] = Binew;
                    
                }
                
                if (foundbetter_i) {
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
        
        if(!foundbetter) {
            // Early exit to prevent looping
            return result;
        }
    }
    
    //result.Model = this;
    return result;
}

template <typename T>
inline double CDL012LogisticSwaps<T>::Objective(arma::vec & r, arma::sp_mat & B) {
    auto l2norm = arma::norm(B, 2);
    return arma::sum(arma::log(1 + 1 / r)) + this->ModelParams[0] * B.n_nonzero + this->ModelParams[1] * arma::norm(B, 1) + this->ModelParams[2] * l2norm * l2norm;
}

#endif
