#ifndef CDL012SWAPS_H
#define CDL012SWAPS_H
#include "RcppArmadillo.h"
#include "CD.h"
#include "CDL012.h"
#include "utils.h"

template <typename T>
class CDL012Swaps : public CD<T>   // Calls CDL012 irrespective of P.ModelType
{
    private:
        unsigned int MaxNumSwaps;
        Params<T> P;
        unsigned int NoSelectK;
    public:
        CDL012Swaps(const T& Xi, const arma::vec& yi, const Params<T>& Pi);

        FitResult<T> Fit() final;

        double Objective(arma::vec & r, arma::sp_mat & B) final;

};

template <typename T>
CDL012Swaps<T>::CDL012Swaps(const T& Xi, const arma::vec& yi, const Params<T>& Pi) : CD<T>(Xi, yi, Pi) {
    MaxNumSwaps = Pi.MaxNumSwaps; P = Pi; NoSelectK=P.NoSelectK;
}

template <typename T>
FitResult<T> CDL012Swaps<T>::Fit()
{
    auto result = CDL012<T>(*(this->X), *(this->y), P).Fit(); // result will be maintained till the end
    this->B = result.B;
    double objective = this->result.Objective;
    P.Init = 'u';
    
    bool foundbetter;
    for (unsigned int t = 0; t < this->MaxNumSwaps; ++t)
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
        
        arma::vec r = *(this->y) - *(this->X) * this->B; // ToDO: take from CD !! No need for this computation.
        
        
        for (auto& i : NnzIndices)
        {
            
            
            arma::rowvec riX = (r + this->B[i] * matrix_column_get(*(this->X), i)).t() * *(this->X); // expensive computation ## call the new function here, inlined? ##
            
            double maxcorr = -1;
            unsigned int maxindex;
            for(unsigned int j = NoSelectK; j < this->p; ++j) // Can be made much faster..
            {
                if (std::fabs(riX[j]) > maxcorr && this->B[j] == 0)
                {
                    maxcorr = std::fabs(riX[j]);
                    maxindex = j;
                }
            }
            
            if(maxcorr > (1 + 2 * this->ModelParams[2])*std::fabs(this->B[i]) + this->ModelParams[1] )
            {
                this->B[i] = 0;
                this->B[maxindex] = (riX[maxindex] - std::copysign(this->ModelParams[1], riX[maxindex])) / (1 + 2 * this->ModelParams[2]);
                P.InitialSol = &(this->B);
                *P.r = *(this->y) - *(this->X) * (this->B);
                result = CDL012<T>(*(this->X), *(this->y), P).Fit();
                this->B = this->result.B;
                objective = this->result.Objective;
                foundbetter = true;
                break;
                
            }
        }
        
        if(!foundbetter)
        {
            //result.Model = this;
            return result;
        }
    }
    
    
    //std::cout<<"Did not achieve CW Swap min" << std::endl;
    //result.Model = this;
    return result;
}

template <typename T>
inline double CDL012Swaps<T>::Objective(arma::vec & r, arma::sp_mat & B)   // hint inline
{
    auto l2norm = arma::norm(B, 2);
    return 0.5 * arma::dot(r, r) + this->ModelParams[0] * B.n_nonzero + this->ModelParams[1] * arma::norm(B, 1) + this->ModelParams[2] * l2norm * l2norm;
}
#endif
