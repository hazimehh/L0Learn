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
        bool withBounds;

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
        
        FitResult<T> Fit();

        virtual ~CDBase(){}

        virtual inline double Objective(const arma::vec &, const arma::sp_mat &)=0;
        
        virtual inline double Objective()=0;
        
        virtual FitResult<T> _FitWithBounds() = 0;
        
        virtual FitResult<T> _Fit() = 0;
        
        virtual inline double GetBiGrad(const std::size_t i)=0;
        
        virtual inline double GetBiValue(const double old_Bi, const double grd_Bi)=0;
        
        virtual inline double GetBiReg(const double nrb_Bi)=0;
        
        virtual inline void ApplyNewBi(const std::size_t i, const double old_Bi, const double new_Bi)=0;
        
        virtual inline void ApplyNewBiCWMinCheck(const std::size_t i, const double old_Bi, const double new_Bi)=0;

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
        
        void UpdateBiWithBounds(const std::size_t i);
        
        bool UpdateBiCWMinCheck(const std::size_t i, const bool Cwmin);
        
        bool UpdateBiCWMinCheckWithBounds(const std::size_t i, const bool Cwmin);
        
        void SupportStabilized();
        
        void UpdateSparse_b0(arma::vec &r);
        
        bool Converged();
        
        bool CWMinCheck();
        
        bool CWMinCheckWithBounds();
        
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
        
        inline void ApplyNewBi(const std::size_t i, const double Bi_old, const double Bi_new) final;
        
        inline void ApplyNewBiCWMinCheck(const std::size_t i, const double old_Bi, const double new_Bi) final;
    
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

template <class T>
inline void CDSwaps<T>::ApplyNewBi(const std::size_t i, const double Bi_old, const double Bi_new){
}


template <class T>
inline void CDSwaps<T>::ApplyNewBiCWMinCheck(const std::size_t i, const double Bi_old, const double Bi_new){
}


template <class T>
void CD<T>::UpdateBiWithBounds(const std::size_t i){
    // Update a single coefficient of B for various CD Settings
    // The following functions are virtual and must be defined for any CD implementation.
    //    GetBiValue
    //    GetBiValue
    //    GetBiReg
    //    ApplyNewBi
    //    ApplyNewBiCWMinCheck (found in UpdateBiCWMinCheck)
    
    
    const double grd_Bi = this->GetBiGrad(i); // Gradient of Loss wrt to Bi
    
    (*this->Xtr)[i] = std::abs(grd_Bi);  // Store absolute value of gradient for later steps
    
    const double old_Bi = this->B[i]; // copy of old Bi to adjust residuals if Bi updates
    
    const double nrb_Bi = this->GetBiValue(old_Bi, grd_Bi); 
    // Update Step for New No regularization No Bounds Bi:
    //                     n  r                 b     _Bi => nrb_Bi
    // Example
    // For CDL0: the update step is nrb_Bi = old_Bi + grd_Bi
    
    const double reg_Bi = this->GetBiReg(nrb_Bi); 
    // Ideal Bi with L1 and L2 regularization (no bounds)
    // Does not account for L0 regularziaton 
    // Example
    // For CDL0: reg_Bi = nrb_Bi as there is no L1, L2 parameters
    
    const double bnd_Bi = clamp(std::copysign(reg_Bi, nrb_Bi),
                                this->Lows[i], this->Highs[i]); 
    // Ideal Bi with regularization and bounds
    
    if (i < this->NoSelectK){
        // L0 penalty is not applied for NoSelectK Variables.
        // Only L1 and L2 (if either are used)
        if (std::abs(nrb_Bi) > this->lambda1){
            this->ApplyNewBi(i, old_Bi, bnd_Bi);
        } else {
            this->ApplyNewBi(i, old_Bi, 0);
        }
    } else if (reg_Bi < this->thr){
        // If ideal non-bounded reg_Bi is less than threshold, coefficient is not worth setting.
        if (old_Bi != 0){
            this->ApplyNewBi(i, old_Bi, 0);
        }
    } else { 
        // Thus reg_Bi >= this->thr 
        
        const double delta_tmp = std::sqrt(reg_Bi*reg_Bi - this->thr2);
        // Due to numerical precision delta_tmp might be nan
        // Turns nans to 0.
        const double delta = (delta_tmp == delta_tmp) ? delta_tmp : 0;
        
        const double range_Bi = std::copysign(reg_Bi, nrb_Bi);
        
        
        if ((range_Bi - delta < bnd_Bi) && (bnd_Bi < range_Bi + delta)){
            // bnd_Bi exists in [bnd_Bi - delta, bnd_Bi + delta]
            // Therefore accept bnd_Bi
            this->ApplyNewBi(i, old_Bi, bnd_Bi);
        } else {
            // Otherwise, reject bnd_Bi
            this->ApplyNewBi(i, old_Bi, 0);
        }
    }
}

template <class T>
void CD<T>::UpdateBi(const std::size_t i){
    // Update a single coefficient of B for various CD Settings
    // The following functions are virtual and must be defined for any CD implementation.
    //    GetBiValue
    //    GetBiValue
    //    GetBiReg
    //    ApplyNewBi
    //    ApplyNewBiCWMinCheck (found in UpdateBiCWMinCheck)
    
    
    const double grd_Bi = this->GetBiGrad(i); // Gradient of Loss wrt to Bi
    
    (*this->Xtr)[i] = std::abs(grd_Bi);  // Store absolute value of gradient for later steps
    
    const double old_Bi = this->B[i]; // copy of old Bi to adjust residuals if Bi updates
    
    const double nrb_Bi = this->GetBiValue(old_Bi, grd_Bi); 
    // Update Step for New No regularization No Bounds Bi:
    //                     n  r                 b     _Bi => nrb_Bi
    // Example
    // For CDL0: the update step is nrb_Bi = old_Bi + grd_Bi
    
    const double reg_Bi = std::copysign(this->GetBiReg(nrb_Bi), nrb_Bi); 
    // Ideal Bi with L1 and L2 regularization (no bounds)
    // Does not account for L0 regularization 
    // Example
    // For CDL0: reg_Bi = nrb_Bi as there is no L1, L2 parameters
    
    if (i < this->NoSelectK){
        // L0 penalty is not applied for NoSelectK Variables.
        // Only L1 and L2 (if either are used)
        if (std::abs(nrb_Bi) > this->lambda1){
            this->ApplyNewBi(i, old_Bi, reg_Bi);
        } else {
            this->ApplyNewBi(i, old_Bi, 0);
        }
    } else if (reg_Bi < this->thr){
        // If ideal non-bounded reg_Bi is less than threshold, coefficient is not worth setting.
        if (old_Bi != 0){
            this->ApplyNewBi(i, old_Bi, 0);
        }
    } else { 
        this->ApplyNewBi(i, old_Bi, reg_Bi);
    }
}

template <class T>
bool CD<T>::UpdateBiCWMinCheck(const std::size_t i, const bool Cwmin){
    // See CD<T>::UpdateBi for documentation
    const double grd_Bi = this->GetBiGrad(i); 
    const double old_Bi = 0;
    
    (*this->Xtr)[i] = std::abs(grd_Bi);  
    
    const double nrb_Bi = this->GetBiValue(old_Bi, grd_Bi); 
    const double reg_Bi = std::copysign(this->GetBiReg(nrb_Bi), nrb_Bi); 
    
    if (i < this->NoSelectK){
        if (std::abs(nrb_Bi) > this->lambda1){
            this->ApplyNewBiCWMinCheck(i, old_Bi, reg_Bi);
            return false;
        } else {
            return Cwmin;
        }
        
    } else if (reg_Bi < this->thr){
        return Cwmin;
    } else {
        this->ApplyNewBiCWMinCheck(i, old_Bi, reg_Bi);
        return false;
    }
}

template <class T>
bool CD<T>::UpdateBiCWMinCheckWithBounds(const std::size_t i, const bool Cwmin){
    // See CD<T>::UpdateBi for documentation
    const double grd_Bi = this->GetBiGrad(i); 
    const double old_Bi = 0;
    
    (*this->Xtr)[i] = std::abs(grd_Bi);  
    
    const double nrb_Bi = this->GetBiValue(old_Bi, grd_Bi); 
    const double reg_Bi = this->GetBiReg(nrb_Bi); 
    const double bnd_Bi = clamp(std::copysign(reg_Bi, nrb_Bi),
                                this->Lows[i], this->Highs[i]); 
    
    if (i < this->NoSelectK){
        if (std::abs(nrb_Bi) > this->lambda1){
            this->ApplyNewBiCWMinCheck(i, old_Bi, bnd_Bi);
            return false;
        } else {
            return Cwmin;
        }
        
    } else if (reg_Bi < this->thr){
        return Cwmin;
    } else {
        
        const double delta_tmp = std::sqrt(reg_Bi*reg_Bi - this->thr2);
        const double delta = (delta_tmp == delta_tmp) ? delta_tmp : 0;
        
        const double range_Bi = std::copysign(reg_Bi, nrb_Bi);
        if ((range_Bi - delta < bnd_Bi) && (bnd_Bi < range_Bi + delta)){
            this->ApplyNewBiCWMinCheck(i, old_Bi, bnd_Bi);
            return false;
        } else {
            return Cwmin;
        }
    }
}


#endif
