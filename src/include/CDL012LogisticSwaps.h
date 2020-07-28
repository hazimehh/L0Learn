#ifndef CDL012LogisticSwaps_H
#define CDL012LogisticSwaps_H
#include <chrono>
#include "CD.h"
#include "FitResult.h"
#include "CDL012Logistic.h"

#include "Normalize.h" // Remove later

class CDL012LogisticSwaps : public CD
{
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
        arma::mat * Xy;
        unsigned int Iter;

        unsigned int MaxNumSwaps;
        Params P;
        unsigned int NoSelectK;

    public:
        CDL012LogisticSwaps(const arma::mat& Xi, const arma::vec& yi, const Params& P);

        FitResult Fit() final;

        inline double Objective(arma::vec & r, arma::sp_mat & B) final;


};

#endif
