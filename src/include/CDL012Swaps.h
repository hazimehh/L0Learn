#ifndef CDL012SWAPS_H
#define CDL012SWAPS_H
#include "CD.h"

class CDL012Swaps : public CD   // Calls CDL012 irrespective of P.ModelType
{
    private:
        unsigned int MaxNumSwaps;
        Params P;
    public:
        CDL012Swaps(const arma::mat& Xi, const arma::vec& yi, const Params& Pi);

        FitResult Fit() final;

        double Objective(arma::vec & r, arma::sp_mat & B) final;

};

#endif