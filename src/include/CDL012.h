#ifndef CDL012_H
#define CDL012_H
#include "RcppArmadillo.h"
#include "CD.h"
#include "FitResult.h"
#include "utils.h"

class CDL012 : public CD
{
    private:
        double thr;
        double Onep2lamda2;
        double lambda1;
        std::vector<double> * Xtr;
        unsigned int Iter;
        unsigned int ScreenSize;
        std::vector<unsigned int> Range1p;
        unsigned int NoSelectK;
    public:
        CDL012(const arma::mat& Xi, const arma::vec& yi, const Params& P);
        //~CDL012(){}

        FitResult Fit() final;

        inline double Objective(arma::vec & r, arma::sp_mat & B) final;

        bool CWMinCheck();

};

#endif
