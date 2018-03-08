#ifndef CDL012CONS_H
#define CDL012CONS_H
#include "CD.h"
#include "FitResult.h"

class CDL012Cons : public CD
{
    private:

        //arma::vec * Xtr;
        unsigned int k = 100;
        double thr;
        unsigned int Iter;
        FitResult result;
        double Onep2lamda2;

    public:
        CDL012Cons(const arma::mat& Xi, const arma::vec& yi, const Params& P);

        FitResult Fit() final;

        double Objective(arma::vec & r, arma::sp_mat & B) final;

};
#endif