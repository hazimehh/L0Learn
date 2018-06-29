#ifndef CDL012_H
#define CDL012_H
#include "CD.h"
#include "FitResult.h"

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

        FitResult Fit() final;

        inline double Objective(arma::vec & r, arma::sp_mat & B) final;

        bool CWMinCheck();

};

#endif
