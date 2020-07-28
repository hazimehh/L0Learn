#ifndef CD_H
#define CD_H
//#include <armadillo>
#include <algorithm>
#include "RcppArmadillo.h"
#include "FitResult.h"
#include "Params.h"


class CD
{
    protected:
        unsigned int n, p;
        arma::sp_mat B;
        arma::sp_mat Bprev;
        unsigned int SameSuppCounter = 0;
        double objective;
        arma::vec r; //vector of residuals
        std::vector<unsigned int> Order; // Cycling order
        std::vector<unsigned int> OldOrder; // Cycling order to be used after support stabilization + convergence.
        FitResult result;

    public:
        const arma::mat * X;
        const arma::vec * y;
        std::vector<double> ModelParams;

        char CyclingOrder;
        unsigned int MaxIters;
        unsigned int CurrentIters; // current number of iterations - maintained by Converged()
        double Tol;
        bool ActiveSet;
        unsigned int ActiveSetNum;
        bool Stabilized = false;


        CD(const arma::mat& Xi, const arma::vec& yi, const Params& P);

        virtual ~CD(){}

        virtual double Objective(arma::vec & r, arma::sp_mat & B) = 0;

        virtual FitResult Fit() = 0;

        bool Converged();

        void SupportStabilized();

        static CD * make_CD(const arma::mat& Xi, const arma::vec& yi, const Params& P);


};

#endif
