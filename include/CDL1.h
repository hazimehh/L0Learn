#ifndef CDL1_H
#define CDL1_H
#include "CD.h"
#include "FitResult.h"

class CDL1 : public CD
{
    private:
        double thr;
        std::vector<double> * Xtr;
        FitResult result;

    public:
        CDL1(const arma::mat& Xi, const arma::vec& yi, const Params& P);

        FitResult Fit() final;

        double Objective(arma::vec & r, arma::sp_mat & B) final;

};

#endif
