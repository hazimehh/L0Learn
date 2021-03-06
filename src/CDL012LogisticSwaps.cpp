#include "CDL012LogisticSwaps.h"

template <class T>
CDL012LogisticSwaps<T>::CDL012LogisticSwaps(const T& Xi, const arma::vec& yi, const Params<T>& Pi) : CDSwaps<T>(Xi, yi, Pi) {
    twolambda2 = 2 * this->lambda2;
    qp2lamda2 = (LipschitzConst + twolambda2); // this is the univariate lipschitz const of the differentiable objective
    this->thr2 = (2 * this->lambda0) / qp2lamda2;
    this->thr = std::sqrt(this->thr2);
    stl0Lc = std::sqrt((2 * this->lambda0) * qp2lamda2);
    lambda1ol = this->lambda1 / qp2lamda2;
    Xy = Pi.Xy;
}

template <class T>
FitResult<T> CDL012LogisticSwaps<T>::_FitWithBounds() {
    throw "This Error should not happen. Please report it as an issue to https://github.com/hazimehh/L0Learn ";
}

template <class T>
FitResult<T> CDL012LogisticSwaps<T>::_Fit() {
    auto result = CDL012Logistic<T>(*(this->X), *(this->y), this->P).Fit(); // result will be maintained till the end
    this->b0 = result.b0; // Initialize from previous later....!
    this->B = result.B;
    ExpyXB = result.ExpyXB; // Maintained throughout the algorithm
    
    double objective = result.Objective;
    double Fmin = objective;
    std::size_t maxindex;
    double Bmaxindex;
    
    this->P.Init = 'u';
    
    bool foundbetter = false;
    bool foundbetter_i = false;
    
    for (std::size_t t = 0; t < this->MaxNumSwaps; ++t) {
        
        std::vector<std::size_t> NnzIndices = nnzIndicies(this->B, this->NoSelectK);
        
        // TODO: Add shuffle of Order
        //std::shuffle(std::begin(Order), std::end(Order), engine);
        
        foundbetter = false;
        
        // TODO: Check if this should be Templated Operation
        arma::mat ExpyXBnojs = arma::zeros(this->n, NnzIndices.size());
        
        int j_index = -1;
        for (auto& j : NnzIndices)
        {
            // Remove NnzIndices[j]
            ++j_index;
            ExpyXBnojs.col(j_index) = ExpyXB % arma::exp( - this->B.at(j) * matrix_column_get(*(this->Xy), j));

        }
        arma::mat gradients = - 1/(1 + ExpyXBnojs).t() * *Xy;
        arma::mat abs_gradients = arma::abs(gradients);
        
        
        j_index = -1;
        for (auto& j : NnzIndices) {
            // Set B[j] = 0
            ++j_index;
            arma::vec ExpyXBnoj = ExpyXBnojs.col(j_index);
            arma::rowvec gradient = gradients.row(j_index);
            arma::rowvec abs_gradient = abs_gradients.row(j_index);
            
            arma::uvec indices = arma::sort_index(arma::abs(gradient), "descend");
            foundbetter_i = false;
            
            // TODO: make sure this scans at least 100 coordinates from outside supp (now it does not)
            for(std::size_t ll = 0; ll < std::min(50, (int) this->p); ++ll) {
                std::size_t i = indices(ll);
                
                if(this->B[i] == 0 && i >= this->NoSelectK) {
                    // Do not swap B[i] if i between 0 and NoSelectK;
                    
                    arma::vec ExpyXBnoji = ExpyXBnoj;
                    
                    double Biold = 0;
                    double partial_i = gradient[i];
                    bool converged = false;
                    
                    beta_vector Btemp = this->B;
                    Btemp[j] = 0;
                    double ObjTemp = Objective(ExpyXBnoji, Btemp);
                    std::size_t innerindex = 0;
                    
                    double x = Biold - partial_i/qp2lamda2;
                    double z = std::abs(x) - lambda1ol;
                    double Binew = std::copysign(z, x);
                    // double Binew = clamp(std::copysign(z, x), this->Lows[i], this->Highs[i]); // no need to check if >= sqrt(2lambda_0/Lc)
                    
                    while(!converged && innerindex < 10  && ObjTemp >= Fmin) { // ObjTemp >= Fmin
                        ExpyXBnoji %= arma::exp( (Binew - Biold) *  matrix_column_get(*Xy, i));
                        //partial_i = - arma::sum( matrix_column_get(*Xy, i) / (1 + ExpyXBnoji) ) + twolambda2 * Binew;
                        partial_i = - arma::dot( matrix_column_get(*Xy, i), 1/(1 + ExpyXBnoji) ) + twolambda2 * Binew;
                        
                        if (std::abs((Binew - Biold)/Biold) < 0.0001) {
                            converged = true;
                            //std::cout<<"swaps converged!!!"<<std::endl;
                        }
                        
                        Biold = Binew;
                        x = Biold - partial_i/qp2lamda2;
                        z = std::abs(x) - lambda1ol;
                        Binew = std::copysign(z, x);
                        // Binew = clamp(std::copysign(z, x), this->Lows[i], this->Highs[i]); // no need to check if >= sqrt(2lambda_0/Lc)
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
                    this->P.InitialSol = &(this->B);
                    
                    // TODO: Check if this line is necessary. P should already have b0.
                    this->P.b0 = this->b0;
                    
                    result = CDL012Logistic<T>(*(this->X), *(this->y), this->P).Fit();
                    
                    ExpyXB = result.ExpyXB;
                    this->B = result.B;
                    this->b0 = result.b0;
                    objective = result.Objective;
                    Fmin = objective;
                    foundbetter = true;
                    break;
                }
            }
            
            //auto end2 = std::chrono::high_resolution_clock::now();
            //std::cout<<"restricted:  "<<std::chrono::duration_cast<std::chrono::milliseconds>(end2-start2).count() << " ms " << std::endl;
            
            if (foundbetter){
                break;
            }
            
        }
        
        if(!foundbetter) {
            // Early exit to prevent looping
            return result;
        }
    }
    
    //result.Model = this;
    return result;
}

template class CDL012LogisticSwaps<arma::mat>;
template class CDL012LogisticSwaps<arma::sp_mat>;
