#ifndef CDL012SquredHingeSwaps_H
#define CDL012SquredHingeSwaps_H
#include "RcppArmadillo.h"
#include "CD.h"
#include "FitResult.h"
#include "CDL012SquaredHinge.h"
#include "Normalize.h" // Remove later
#include "utils.h"

template <typename T>
class CDL012SquaredHingeSwaps : public CD<T> {
    private:
        const double LipschitzConst = 2;
        double thr;
        double twolambda2;
        double qp2lamda2;
        double lambda1;
        double lambda1ol;
        double b0;
        double stl0Lc;
        std::vector<double> * Xtr;
        std::size_t Iter;
        std::size_t NoSelectK;

        std::size_t MaxNumSwaps;
        Params<T> P;

    public:
        CDL012SquaredHingeSwaps(const T& Xi, const arma::vec& yi, const Params<T>& P);

        FitResult<T> Fit() final;

        inline double Objective(arma::vec & r, arma::sp_mat & B) final;


};

template <typename T>
CDL012SquaredHingeSwaps<T>::CDL012SquaredHingeSwaps(const T& Xi, const arma::vec& yi, const Params<T>& Pi) : CD<T>(Xi, yi, Pi) {
    MaxNumSwaps = Pi.MaxNumSwaps;
    P = Pi;
    twolambda2 = 2 * this->ModelParams[2];
    qp2lamda2 = (LipschitzConst + twolambda2); // this is the univariate lipschitz const of the differentiable objective
    thr = std::sqrt((2 * this->ModelParams[0]) / qp2lamda2);
    stl0Lc = std::sqrt((2 * this->ModelParams[0]) * qp2lamda2);
    lambda1 = this->ModelParams[1];
    lambda1ol = lambda1 / qp2lamda2;
    Xtr = P.Xtr; 
    Iter = P.Iter; 
    this->result.ModelParams = P.ModelParams;
    
}

template <typename T>
FitResult<T> CDL012SquaredHingeSwaps<T>::Fit() {
    auto result = CDL012SquaredHinge<T>(*(this->X), *(this->y), P).Fit(); // result will be maintained till the end
    b0 = result.intercept; // Initialize from previous later....!
    this->B = result.B;
    //ExpyXB = result.ExpyXB; // Maintained throughout the algorithm
    arma::vec onemyxb = 1 - *(this->y) % (*(this->X) * this->B + b0);
    
    double objective = result.Objective;
    double Fmin = objective;
    std::size_t maxindex;
    double Bmaxindex;
    
    P.Init = 'u';
    
    bool foundbetter;
    for (std::size_t t = 0; t < MaxNumSwaps; ++t) {
        arma::sp_mat::const_iterator start = this->B.begin();
        arma::sp_mat::const_iterator end   = this->B.end();
        std::vector<std::size_t> NnzIndices;
        for(arma::sp_mat::const_iterator it = start; it != end; ++it) {
            if (it.row() >= NoSelectK)
                NnzIndices.push_back(it.row());
        }
        // Can easily shuffle here...
        //std::shuffle(std::begin(Order), std::end(Order), engine);
        foundbetter = false;
        
        for (auto& j : NnzIndices) {
            // Remove j
            //arma::vec ExpyXBnoj = ExpyXB % arma::exp( - B[j] *  *y % X->unsafe_col(j));
            arma::vec onemyxbnoj = onemyxb + this->B[j] * *(this->y) % matrix_column_get(*(this->X), j);
            arma::uvec indices = arma::find(onemyxbnoj > 0);
            
            
            for(std::size_t i = 0; i < this->p; ++i) {
                if(this->B[i] == 0 && i>=NoSelectK) {
                    
                    double Biold = 0;
                    double Binew;
                    
                    
                    double partial_i = arma::sum(2 * onemyxbnoj.elem(indices) % (- (this->y)->elem(indices) % matrix_column_get(*(this->X), i).elem(indices)));
                    
                    bool converged = false;
                    if (std::abs(partial_i) >= lambda1 + stl0Lc) {
                        
                        //std::cout<<"Adding: "<<i<< std::endl;
                        arma::vec onemyxbnoji = onemyxbnoj;
                        
                        std::size_t l = 0;
                        arma::sp_mat Btemp = this->B;
                        Btemp[j] = 0;
                        //double ObjTemp = Objective(onemyxbnoj,Btemp);
                        //double Biolddescent = 0;
                        while(!converged) {
                            double x = Biold - partial_i / qp2lamda2;
                            double z = std::abs(x) - lambda1ol;
                            
                            Binew = std::copysign(z, x); // no need to check if >= sqrt(2lambda_0/Lc)
                            onemyxbnoji += (Biold - Binew) * *(this->y) % matrix_column_get(*(this->X), i);
                            
                            arma::uvec indicesi = arma::find(onemyxbnoji > 0);
                            partial_i = arma::sum(2 * onemyxbnoji.elem(indicesi) % (- (this->y)->elem(indicesi) % matrix_column_get(*(this->X), i).elem(indicesi)));
                            
                            if (std::abs((Binew - Biold) / Biold) < 0.0001) 
                                converged = true;
                        
                            Biold = Binew;
                            l += 1;
                            
                        }
                        
                        // Can be made much faster (later)
                        Btemp[i] = Binew;
                        //auto l2norm = arma::norm(Btemp,2);
                        //double Fnew = arma::sum(arma::log(1 + 1/ExpyXBnoji)) + ModelParams[0]*Btemp.n_nonzero + ModelParams[1]*arma::norm(Btemp,1) + ModelParams[2]*l2norm*l2norm;
                        double Fnew = Objective(onemyxbnoji, Btemp);
                        //std::cout<<"Fnew: "<<Fnew<<"Index: "<<i<<std::endl;
                        if (Fnew < Fmin) {
                            Fmin = Fnew;
                            maxindex = i;
                            Bmaxindex = Binew;
                        }
                    }
                }
            }
            
            if (Fmin < objective) {
                this->B[j] = 0;
                this->B[maxindex] = Bmaxindex;
                P.InitialSol = &(this->B);
                P.b0 = b0;
                
                result = CDL012SquaredHinge<T>(*(this->X), *(this->y), P).Fit();
                //ExpyXB = result.ExpyXB;
                this->B = result.B;
                b0 = result.intercept;
                onemyxb = 1 - *(this->y) % (*(this->X) * this->B + b0); // no need to calc. - change later.
                objective = result.Objective;
                Fmin = objective;
                foundbetter = true;
                break;
            }
        }
        
        if(!foundbetter) {
            //result.Model = this;
            return result;
        }
    }
    
    
    //std::cout<<"Did not achieve CW Swap min" << std::endl;
    //result.Model = this;
    return result;
}

template <typename T>
inline double CDL012SquaredHingeSwaps<T>::Objective(arma::vec & onemyxb, arma::sp_mat & B) {
    auto l2norm = arma::norm(B, 2);
    arma::uvec indices = arma::find(onemyxb > 0);
    return arma::sum(onemyxb.elem(indices) % onemyxb.elem(indices)) + this->ModelParams[0] * B.n_nonzero + this->ModelParams[1] * arma::norm(B, 1) + this->ModelParams[2] * l2norm * l2norm;
}


#endif
