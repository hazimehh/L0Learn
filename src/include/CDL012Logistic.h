#ifndef CDL012Logistic_H
#define CDL012Logistic_H
#include "CD.h"
#include "FitResult.h"

class CDL012Logistic : public CD
{
    private:
        const double LipschitzConst = 0.25;
        double thr;
        double twolambda2;
        double qp2lamda2;
        double lambda1;
        double lambda1ol;
        double b0;
        arma::vec ExpyXB;
        std::vector<double> * Xtr;
        arma::vec * Xty;
        arma::mat * Xy;
        unsigned int Iter;
        unsigned int NoSelectK;
        unsigned int ScreenSize;
        std::vector<unsigned int> Range1p;
    public:
        CDL012Logistic(const arma::mat& Xi, const arma::vec& yi, const Params& P);

        FitResult Fit() final;

        inline double Objective(arma::vec & r, arma::sp_mat & B) final;

        bool CWMinCheck();

};

#endif
