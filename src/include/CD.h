#ifndef CD_H
#define CD_H
#include <algorithm>
#include "RcppArmadillo.h"
#include "FitResult.h"
#include "Params.h"
#include "Model.h"
#include "utils.h"


template<class T> 
class CDBase {
    protected:
        std::size_t NoSelectK;
        std::vector<double> * Xtr;
        std::size_t n, p;
        std::size_t Iter;
        
        arma::sp_mat B;
        arma::sp_mat Bprev;
        
        std::size_t SameSuppCounter = 0;
        double objective;
        std::vector<std::size_t> Order; // Cycling order
        std::vector<std::size_t> OldOrder; // Cycling order to be used after support stabilization + convergence.
        FitResult<T> result;
        
        /* Intercept and b0 are used for:
         *  1. Classification as b0 is updated iteratively in the CD algorithm 
         *  2. Regression on Sparse Matrices as we cannot adjust the support of 
         *  the columns of X and thus b0 must be updated iteraveily from the 
         *  residuals 
         */ 
        double b0 = 0; 
        double lambda1;
        double lambda0;
        double lambda2;
        double thr;
        double thr2; // threshold squared; 
        
        bool isSparse;
        bool intercept; 

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


        CDBase(const T& Xi, const arma::vec& yi, const Params<T>& P);

        virtual ~CDBase(){}

        virtual inline double Objective(const arma::vec &, const arma::sp_mat &)=0;
        
        virtual inline double Objective()=0;

        virtual FitResult<T> Fit() = 0;
        
        virtual inline double GetBiGrad(const std::size_t i)=0;
        
        virtual inline double GetBiValue(const double old_Bi, const double grd_Bi)=0;
        
        virtual inline double GetBiReg(const double nrb_Bi)=0;
        
        // virtual inline double GetBiDelta(const double reg_Bi)=0;
        
        virtual inline void ApplyNewBi(const std::size_t i, const double old_Bi, const double new_Bi)=0;

        static CDBase * make_CD(const T& Xi, const arma::vec& yi, const Params<T>& P);

};

template<class T>
class CD : public CDBase<T>{
    protected:
        std::size_t ScreenSize;
        std::vector<std::size_t> Range1p;
        
    public:
        CD(const T& Xi, const arma::vec& yi, const Params<T>& P);
        
        virtual ~CD(){};
        
        void UpdateBi(const std::size_t i);
        
        bool UpdateBiCWMinCheck(const std::size_t i, const bool Cwmin);
        
        void SupportStabilized();
        
        void UpdateSparse_b0(arma::vec &r);
        
        bool Converged();
        
};


template<class T> 
class CDSwaps : public CDBase<T> {
    
    protected:
        std::size_t MaxNumSwaps;
        Params<T> P;

    public:
        
        CDSwaps(const T& Xi, const arma::vec& yi, const Params<T>& P);
        
        virtual ~CDSwaps(){};
        
        inline double GetBiGrad(const std::size_t i) final;
        
        inline double GetBiValue(const double old_Bi, const double grd_Bi) final;
        
        inline double GetBiReg(const double Bi_step) final;
        
        // inline double GetBiDelta(const double Bi_reg) final;
        
        inline void ApplyNewBi(const std::size_t i, const double Bi_old, const double Bi_new) final;
    
};


template <class T>
inline double CDSwaps<T>::GetBiGrad(const std::size_t i){
    return -1;
}

template <class T>
inline double CDSwaps<T>::GetBiValue(const double old_Bi, const double grd_Bi){
    return -1;
}

template <class T>
inline double CDSwaps<T>::GetBiReg(const double Bi_step){
    return -1;
}

// template <class T>
// inline double CDSwaps<T>::GetBiDelta(const double Bi_reg){
//     return -1;
// }

template <class T>
inline void CDSwaps<T>::ApplyNewBi(const std::size_t i, const double Bi_old, const double Bi_new){
}



#endif
