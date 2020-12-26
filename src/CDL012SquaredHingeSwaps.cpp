#include "CDL012SquaredHingeSwaps.h"

template <class T>
CDL012SquaredHingeSwaps<T>::CDL012SquaredHingeSwaps(const T& Xi, const arma::vec& yi, const Params<T>& Pi) : CDSwaps<T>(Xi, yi, Pi) {
    twolambda2 = 2 * this->lambda2;
    qp2lamda2 = (LipschitzConst + twolambda2); // this is the univariate lipschitz constant of the differentiable objective
    this->thr2 = (2 * this->lambda0) / qp2lamda2;
    this->thr = std::sqrt(this->thr2);
    stl0Lc = std::sqrt((2 * this->lambda0) * qp2lamda2);
    lambda1ol = this->lambda1 / qp2lamda2;
}

template <class T>
FitResult<T> CDL012SquaredHingeSwaps<T>::_FitWithBounds() {
    throw "This Error should not happen. Please report it as an issue to https://github.com/hazimehh/L0Learn ";
}

template <class T>
FitResult<T> CDL012SquaredHingeSwaps<T>::_Fit() {
    auto result = CDL012SquaredHinge<T>(*(this->X), *(this->y), this->P).Fit(); // result will be maintained till the end
    this->b0 = result.b0; // Initialize from previous later....!
    this->B = result.B;
    
    arma::vec onemyxb = result.onemyxb;
    
    this->objective = result.Objective;
    double Fmin = this->objective;
    std::size_t maxindex;
    double Bmaxindex;
    
    this->P.Init = 'u';
    
    bool foundbetter;
    
    for (std::size_t t = 0; t < this->MaxNumSwaps; ++t) {
        // Rcpp::Rcout << "Swap Number: " << t << "|mean(onemyxb): " << arma::mean(onemyxb) << "\n";
        
        std::vector<std::size_t> NnzIndices = nnzIndicies(this->B, this->NoSelectK);
        
        // TODO: Implement shuffle of NnzIndices Indicies

        foundbetter = false;
        
        for (auto& j : NnzIndices) {
           
            arma::vec onemyxbnoj = onemyxb + this->B[j] * *(this->y) % matrix_column_get(*(this->X), j);
            arma::uvec indices = arma::find(onemyxbnoj > 0);
            
            
            for(std::size_t i = 0; i < this->p; ++i) {
                if(this->B[i] == 0 && i>=this->NoSelectK) {
                    
                    double Biold = 0;
                    double Binew;
                    
                    
                    double partial_i = arma::sum(2 * onemyxbnoj.elem(indices) % (- (this->y)->elem(indices) % matrix_column_get(*(this->X), i).elem(indices)));
                    
                    bool converged = false;
                    if (std::abs(partial_i) >= this->lambda1 + stl0Lc){
                        
                        //std::cout<<"Adding: "<<i<< std::endl;
                        arma::vec onemyxbnoji = onemyxbnoj;
                        
                        std::size_t l = 0;
                        beta_vector Btemp = this->B;
                        Btemp[j] = 0;
                        //double ObjTemp = Objective(onemyxbnoj,Btemp);
                        //double Biolddescent = 0;
                        while(!converged) {
                            
                            double x = Biold - partial_i / qp2lamda2;
                            double z = std::abs(x) - lambda1ol;
                            Binew = std::copysign(z, x);
                            
                            // Binew = clamp(std::copysign(z, x), this->Lows[i], this->Highs[i]); // no need to check if >= sqrt(2lambda_0/Lc)
                            onemyxbnoji += (Biold - Binew) * *(this->y) % matrix_column_get(*(this->X), i);
                            
                            arma::uvec indicesi = arma::find(onemyxbnoji > 0);
                            partial_i = arma::sum(2 * onemyxbnoji.elem(indicesi) % (- (this->y)->elem(indicesi) % matrix_column_get(*(this->X), i).elem(indicesi)));
                            
                            if (std::abs((Binew - Biold) / Biold) < 0.0001){
                                converged = true;   
                            }
                            
                            Biold = Binew;
                            l += 1;
                            
                        }
                        
                        Btemp[i] = Binew;
                        double Fnew = Objective(onemyxbnoji, Btemp);
                        
                        if (Fnew < Fmin) {
                            Fmin = Fnew;
                            maxindex = i;
                            Bmaxindex = Binew;
                        }
                    }
                }
            }
            
            if (Fmin < this->objective) {
                this->B[j] = 0;
                this->B[maxindex] = Bmaxindex;
                
                this->P.InitialSol = &(this->B);
                
                // TODO: Check if this line is needed. P should already have b0.
                this->P.b0 = this->b0;
                
                result = CDL012SquaredHinge<T>(*(this->X), *(this->y), this->P).Fit();
                
                this->B = result.B;
                this->b0 = result.b0;
                
                onemyxb = result.onemyxb;
                this->objective = result.Objective;
                Fmin = this->objective;
                foundbetter = true;
                break;
            }
        }
        
        if(!foundbetter) {
            return result;
        }
    }
    
    return result;
}

template class CDL012SquaredHingeSwaps<arma::mat>;
template class CDL012SquaredHingeSwaps<arma::sp_mat>;
