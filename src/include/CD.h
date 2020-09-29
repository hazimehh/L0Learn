#ifndef CD_H
#define CD_H
//#include <armadillo>
#include <algorithm>
#include "RcppArmadillo.h"
#include "FitResult.h"
#include "Params.h"


template<typename T> 
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

template <typename T>
CD<T>::CD(const T& Xi, const arma::vec& yi, const Params<T>& P) :
    ModelParams{P.ModelParams}, CyclingOrder{P.CyclingOrder}, MaxIters{P.MaxIters},
    Tol{P.Tol}, ActiveSet{P.ActiveSet}, ActiveSetNum{P.ActiveSetNum} {
        
        isSparse = std::is_same<T,arma::sp_mat>::value;
        
        b0 = P.b0;
        intercept = P.intercept;
        X = &Xi;
        y = &yi;
        
        n = X->n_rows;
        p = X->n_cols;
        
        if (P.Init == 'u') {
            B = *(P.InitialSol);
        } else if (P.Init == 'r') {
            arma::urowvec row_indices = arma::randi<arma::urowvec>(P.RandomStartSize, arma::distr_param(0, p - 1));
            row_indices = arma::unique(row_indices);
            auto rsize = row_indices.n_cols;
            
            arma::umat indices(2, rsize);
            indices.row(0) = row_indices;
            indices.row(1) = arma::zeros<arma::urowvec>(rsize);
            
            arma::vec values = arma::randu<arma::vec>(rsize) * 2 - 1; // uniform(-1,1)
            
            B = arma::sp_mat(false, indices, values, p, 1, false, false);
        } else {
            B = arma::sp_mat(p, 1); // Initialized to zeros
        }
        
        if (CyclingOrder == 'u') {
            Order = P.Uorder;
        } else if (CyclingOrder == 'c') {
            std::vector<std::size_t> cyclic(p);
            std::iota(std::begin(cyclic), std::end(cyclic), 0);
            Order = cyclic;
        }
        
        this->Lows = P.Lows;
        this->Highs = P.Highs;
        
        CurrentIters = 0;
    }


template <typename T>
bool CD<T>::Converged() {
    CurrentIters += 1; // keeps track of the number of calls to Converged
    double objectiveold = objective;
    objective = this->Objective(r, B);
    if (std::abs(objectiveold - objective) / objectiveold <= Tol) {// handles obj <=0 for non-descent methods
        return true; 
    } else { 
        return false; 
    }
}

template <typename T>
void CD<T>::SupportStabilized() {
    
    bool SameSupp = true;
    
    if (Bprev.n_nonzero != B.n_nonzero) {
        SameSupp = false;
    } else {  // same number of nnz and Supp is sorted
        arma::sp_mat::const_iterator i1;
        arma::sp_mat::const_iterator i2;
        for(i1 = B.begin(), i2 = Bprev.begin(); i1 != B.end(); ++i1, ++i2) {
            if (i1.row() != i2.row()) {
                SameSupp = false;
                break;
            }
        }
    }
    
    if (SameSupp) {
        SameSuppCounter += 1;
        
        if (SameSuppCounter == ActiveSetNum - 1) {
            std::vector<std::size_t> NewOrder(B.n_nonzero);
            
            arma::sp_mat::const_iterator i;
            for(i = B.begin(); i != B.end(); ++i) {
                NewOrder.push_back(i.row());
            }
            
            std::sort(NewOrder.begin(), NewOrder.end(), [this](std::size_t i, std::size_t j) {return Order[i] <  Order[j] ;});
            
            OldOrder = Order;
            Order = NewOrder;
            ActiveSet = false;
            //std::cout<< "Stabilized" << std::endl;
            Stabilized = true;
            
        }
        
    } else {
        SameSuppCounter = 0;
    }
    
}

#endif
