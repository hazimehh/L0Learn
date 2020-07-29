#ifndef CDL0_H
#define CDL0_H
#include "RcppArmadillo.h"
#include "CD.h"
#include "FitResult.h" // REMOVE: may need to include params??
#include "utils.h"

class CDL0 : public CD
{
    private:
        double thr;
        std::vector<double> * Xtr;
        unsigned int Iter;
        unsigned int ScreenSize;
        std::vector<unsigned int> Range1p;
        unsigned int NoSelectK;
    public:
        CDL0(const arma::mat& Xi, const arma::vec& yi, const Params& P);
        //~CDL0(){}

        FitResult Fit() final;

        double Objective(arma::vec & r, arma::sp_mat & B) final;

        bool CWMinCheck();
};

#endif
