#ifndef CDL012SquredHingeSwaps_H
#define CDL012SquredHingeSwaps_H
#include "RcppArmadillo.h"
#include "CD.h"
#include "FitResult.h"
#include "CDL012SquaredHinge.h"
#include "Normalize.h" // Remove later


class CDL012SquaredHingeSwaps : public CD
{
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
        unsigned int Iter;
        unsigned int NoSelectK;

        unsigned int MaxNumSwaps;
        Params P;

    public:
        CDL012SquaredHingeSwaps(const arma::mat& Xi, const arma::vec& yi, const Params& P);

        FitResult Fit() final;

        inline double Objective(arma::vec & r, arma::sp_mat & B) final;


};

#endif
