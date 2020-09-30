#ifndef CD_H
#define CD_H
#include <algorithm>
#include "RcppArmadillo.h"
#include "FitResult.h"
#include "Params.h"


template<class T> 
class CD {
    protected:
        std::size_t n, p;
        arma::sp_mat B;
        arma::sp_mat Bprev;
        std::size_t SameSuppCounter = 0;
        double objective;
        arma::vec r; //vector of residuals
        std::vector<std::size_t> Order; // Cycling order
        std::vector<std::size_t> OldOrder; // Cycling order to be used after support stabilization + convergence.
        FitResult<T> result;
        
        /* Intercept and b0 are used for:
         *  1. Classification as b0 is updated iteratively in the CD algorithm 
         *  2. Regression on Sparse Matrices as we cannot adjust the support of 
         *  the columns of X and thus b0 must be updated iteraveily from the 
         *  residuals 
         */ 
        bool intercept; 
        double b0 = 0; 
        bool isSparse;

    public:
        const T * X;
        const arma::vec * y;
        std::vector<double> ModelParams;

        char CyclingOrder;
        std::size_t MaxIters;
        std::size_t CurrentIters; // current number of iterations - maintained by Converged()
        double Tol;
        arma::vec Lows;
        arma::vec Highs;
        bool ActiveSet;
        std::size_t ActiveSetNum;
        bool Stabilized = false;


        CD(const T& Xi, const arma::vec& yi, const Params<T>& P);

        virtual ~CD(){}

        virtual double Objective(arma::vec & r, arma::sp_mat & B) = 0;

        virtual FitResult<T> Fit() = 0;

        bool Converged();

        void SupportStabilized();

        static CD * make_CD(const T& Xi, const arma::vec& yi, const Params<T>& P);

};

#endif
