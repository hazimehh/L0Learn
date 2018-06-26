#ifndef CDL012SquaredHinge_H
#define CDL012SquaredHinge_H
#include "CD.h"
#include "FitResult.h"

class CDL012SquaredHinge : public CD
{
    private:
        const double LipschitzConst = 2; // for f (without regularization)
        double thr;
        double twolambda2;
        double qp2lamda2;
        double lambda1;
        double lambda1ol;
        double b0;
        std::vector<double> * Xtr;
        unsigned int Iter;
        arma::vec onemyxb;

    public:
        CDL012SquaredHinge(const arma::mat& Xi, const arma::vec& yi, const Params& P);

        inline double Derivativei(unsigned int i);

        inline double Derivativeb();

        FitResult Fit() final;

        inline double Objective(arma::vec & r, arma::sp_mat & B) final;

};

#endif
