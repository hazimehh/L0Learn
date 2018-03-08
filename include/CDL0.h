#ifndef CDL0_H
#define CDL0_H
#include "CD.h"
#include "FitResult.h" // REMOVE: may need to include params??

class CDL0 : public CD
{
    private:
        double thr;
        std::vector<double> * Xtr;
        arma::rowvec * ytX; // new imp
        std::map<unsigned int, arma::rowvec> * D; //new imp
        unsigned int Iter;
        unsigned int ScreenSize;
        std::vector<unsigned int> Range1p;
        FitResult result;
    public:
        CDL0(const arma::mat& Xi, const arma::vec& yi, const Params& P);

        FitResult Fit() final;

        double Objective(arma::vec & r, arma::sp_mat & B) final;

        bool CWMinCheck();
};

#endif
