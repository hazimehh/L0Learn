#ifndef CDL012SWAPS_H
#define CDL012SWAPS_H
#include <map>
#include "RcppArmadillo.h"
#include "CD.h"
#include "CDL012.h"
#include "utils.h"

template <typename T>
class CDL012Swaps : public CD<T> { 
    private:
        std::size_t MaxNumSwaps;
        Params<T> P;
        std::size_t NoSelectK;
    public:
        CDL012Swaps(const T& Xi, const arma::vec& yi, const Params<T>& Pi);

        FitResult<T> Fit() final;

        double Objective(arma::vec & r, arma::sp_mat & B) final;

};

template <typename T>
CDL012Swaps<T>::CDL012Swaps(const T& Xi, const arma::vec& yi, const Params<T>& Pi) : CD<T>(Xi, yi, Pi) {
    MaxNumSwaps = Pi.MaxNumSwaps;
    P = Pi;
    NoSelectK=P.NoSelectK;
}

template <typename T>
FitResult<T> CDL012Swaps<T>::Fit() {
    auto result = CDL012<T>(*(this->X), *(this->y), P).Fit(); // result will be maintained till the end
    this->B = result.B;
    double objective = result.Objective;
    P.Init = 'u';
    
    bool foundbetter = false;
    
    for (std::size_t t = 0; t < this->MaxNumSwaps; ++t) {
        arma::sp_mat::const_iterator start = this->B.begin();
        arma::sp_mat::const_iterator end   = this->B.end();
        std::vector<std::size_t> NnzIndices;
        for(arma::sp_mat::const_iterator it = start; it != end; ++it) {
            if (it.row() >= NoSelectK){
                NnzIndices.push_back(it.row());
            }
        }
        
        // TODO: shuffle NNz Indices to prevent bias.
        //std::shuffle(std::begin(Order), std::end(Order), engine);
        
        // TODO: This calculation is already preformed in a previous step
        // Can be pulled/stored 
        arma::vec r = *(this->y) - *(this->X) * this->B; 
        
        for (auto& i : NnzIndices) {
            arma::rowvec riX = (r + this->B[i] * matrix_column_get(*(this->X), i)).t() * *(this->X); 
            
            double maxcorr = -1;
            std::size_t maxindex = -1;
            
            for(std::size_t j = NoSelectK; j < this->p; ++j) {
                // TODO: Account for bounds when determining best swap
                // Loops through each column and finds the column with the highest correlation to residuals
                // In non-constrained cases, the highest correlation will always be the best option
                // However, if bounds restrict the value of B[j], it is possible that swapping column 'i'
                // and column 'j' might be rejected as B[j], when constrained, is not able to take a value
                // with sufficient magnitude to utilie the correleation.
                // Therefore, we must ensure that 'j' was not already rejected.
                if (std::fabs(riX[j]) > maxcorr && this->B[j] == 0) {
                    maxcorr = std::fabs(riX[j]);
                    maxindex = j;
                }
            }
            
            // Check if the correlation is sufficiently large to make up for regularization
            if(maxcorr > (1 + 2 * this->ModelParams[2])*std::fabs(this->B[i]) + this->ModelParams[1]) {
                Rcpp::Rcout << t << ": Proposing Swap " << i << " => NNZ and " << maxindex << " => 0 \n";
                // Proposed new Swap
                // Value (without considering bounds are solvable in closed form)
                // Must be clamped to bounds
                
                T old_B = T(this->B); // Copy B if swap needs to be "undone"
                // arma::vec old_r = P.r;
                this->B[i] = 0;
                
                // Bi with No Bounds (nb);
                double Bi_nb = (riX[maxindex]  - std::copysign(this->ModelParams[1],riX[maxindex])) / (1 + 2 * this->ModelParams[2]);
                double Bi_wb = clamp(Bi_nb, this->Lows[maxindex], this->Highs[maxindex]);  // Bi With Bounds (wb)
                this->B[maxindex] = Bi_wb;
                
                // Change initial solution to Swapped value to seed standard CD algorithm.
                P.InitialSol = &(this->B);
                *P.r = *(this->y) - *(this->X) * (this->B);
                result = CDL012<T>(*(this->X), *(this->y), P).Fit();
                
                Rcpp::Rcout << "Swap Objective  " <<  result.Objective << " \n";
                Rcpp::Rcout << "Old Objective  " <<  objective << " \n";
                if (result.Objective <= objective){
                    // Accept Swap
                    this->B = result.B;
                    objective = result.Objective;
                    foundbetter = true;
                    break;
                } else {
                    // Reject Swap
                    this->B = old_B;
                    //*P.r = old_r;
                }
            }
        }
        
        if(!foundbetter) {
            // Early exit to prevent looping
            return result;
        }
    }
    
    return result;
}

template <typename T>
inline double CDL012Swaps<T>::Objective(arma::vec & r, arma::sp_mat & B) {
    auto l2norm = arma::norm(B, 2);
    return 0.5 * arma::dot(r, r) + this->ModelParams[0] * B.n_nonzero + this->ModelParams[1] * arma::norm(B, 1) + this->ModelParams[2] * l2norm * l2norm;
}

#endif
