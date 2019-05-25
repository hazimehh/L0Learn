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
        unsigned int NoSelectK;
        arma::mat * Xy;
        unsigned int ScreenSize;
        std::vector<unsigned int> Range1p;

    public:
        CDL012SquaredHinge(const arma::mat& Xi, const arma::vec& yi, const Params& P);
        //~CDL012SquaredHinge(){}
        //inline double Derivativei(unsigned int i);

        //inline double Derivativeb();

        FitResult Fit() final;

        inline double Objective(arma::vec & r, arma::sp_mat & B) final;

        bool CWMinCheck();


};

#endif
