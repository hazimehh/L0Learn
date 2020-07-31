#ifndef CD_H
#define CD_H
//#include <armadillo>
#include <algorithm>
#include "RcppArmadillo.h"
#include "FitResult.h"
#include "Params.h"


template<typename T> 
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
        FitResult<T> result;

    public:
        const T * X;
        const arma::vec * y;
        std::vector<double> ModelParams;

        char CyclingOrder;
        unsigned int MaxIters;
        unsigned int CurrentIters; // current number of iterations - maintained by Converged()
        double Tol;
        bool ActiveSet;
        unsigned int ActiveSetNum;
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
    Tol{P.Tol}, ActiveSet{P.ActiveSet}, ActiveSetNum{P.ActiveSetNum}
    {
        X = &Xi;
        y = &yi;
        
        n = X->n_rows;
        p = X->n_cols;
        
        if (P.Init == 'u')
        {
            B = *(P.InitialSol);
        }
        
        else if (P.Init == 'r')
        {
            arma::urowvec row_indices = arma::randi<arma::urowvec>(P.RandomStartSize, arma::distr_param(0, p - 1));
            row_indices = arma::unique(row_indices);
            auto rsize = row_indices.n_cols;
            
            arma::umat indices(2, rsize);
            indices.row(0) = row_indices;
            indices.row(1) = arma::zeros<arma::urowvec>(rsize);
            
            arma::vec values = arma::randu<arma::vec>(rsize) * 2 - 1; // uniform(-1,1)
            
            B = arma::sp_mat(false, indices, values, p, 1, false, false);
        }
        
        else
        {
            B = arma::sp_mat(p, 1);
        }
        
        if (CyclingOrder == 'u')
        {
            Order = P.Uorder;
        }
        
        else if (CyclingOrder == 'c')
        {
            std::vector<unsigned int> cyclic(p);
            std::iota(std::begin(cyclic), std::end(cyclic), 0);
            Order = cyclic;
        }
        
        CurrentIters = 0;
    }

template <typename T>
bool CD<T>::Converged()
{
    CurrentIters += 1; // keeps track of the number of calls to Converged
    double objectiveold = objective;
    objective = this->Objective(r, B);
    if (std::abs(objectiveold - objective) / objectiveold <= Tol) // handles obj <=0 for non-descent methods
    { return true; }
    else
    { return false; }
}

template <typename T>
void CD<T>::SupportStabilized()
{
    
    bool SameSupp = true;
    
    if (Bprev.n_nonzero != B.n_nonzero) {SameSupp = false;}
    else   // same number of nnz and Supp is sorted
    {
        arma::sp_mat::const_iterator i1;
        arma::sp_mat::const_iterator i2;
        for(i1 = B.begin(), i2 = Bprev.begin(); i1 != B.end(); ++i1, ++i2)
        {
            if(i1.row() != i2.row())
            {
                SameSupp = false;
                break;
            }
        }
    }
    
    if (SameSupp)
    {
        SameSuppCounter += 1;
        if (SameSuppCounter == ActiveSetNum - 1)
        {
            std::vector<unsigned int> NewOrder(B.n_nonzero);
            
            arma::sp_mat::const_iterator i;
            for(i = B.begin(); i != B.end(); ++i)
            {
                NewOrder.push_back(i.row());
            }
            
            std::sort(NewOrder.begin(), NewOrder.end(), [this](unsigned int i, unsigned int j) {return Order[i] <  Order[j] ;});
            
            OldOrder = Order;
            Order = NewOrder;
            ActiveSet = false;
            //std::cout<< "Stabilized" << std::endl;
            Stabilized = true;
            
        }
        
    }
    
    else {SameSuppCounter = 0;}
    
}

#endif
