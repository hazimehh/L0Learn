#include "CD.h"

template <class T>
CDBase<T>::CDBase(const T& Xi, const arma::vec& yi, const Params<T>& P) :
    ModelParams{P.ModelParams}, CyclingOrder{P.CyclingOrder}, MaxIters{P.MaxIters},
    Tol{P.Tol}, ActiveSet{P.ActiveSet}, ActiveSetNum{P.ActiveSetNum} 
{
    
    this->lambda0 = P.ModelParams[0];
    this->lambda1 = P.ModelParams[1];
    this->lambda2 = P.ModelParams[2];
    
    this->result.ModelParams = P.ModelParams; 
    this->NoSelectK = P.NoSelectK;
    
    this->Xtr = P.Xtr; 
    this->Iter = P.Iter;
        
    this->isSparse = std::is_same<T,arma::sp_mat>::value;
    
    this->b0 = P.b0;
    this->intercept = P.intercept;
    this->X = &Xi;
    this->y = &yi;
    
    this->n = X->n_rows;
    this->p = X->n_cols;
    
    if (P.Init == 'u') {
        this->B = *(P.InitialSol);
    } else if (P.Init == 'r') {
        arma::urowvec row_indices = arma::randi<arma::urowvec>(P.RandomStartSize, arma::distr_param(0, p - 1));
        row_indices = arma::unique(row_indices);
        auto rsize = row_indices.n_cols;
        
        arma::umat indices(2, rsize);
        indices.row(0) = row_indices;
        indices.row(1) = arma::zeros<arma::urowvec>(rsize);
        
        arma::vec values = arma::randu<arma::vec>(rsize) * 2 - 1; // uniform(-1,1)
        
        this->B = arma::sp_mat(false, indices, values, p, 1, false, false);
    } else {
        this->B = arma::sp_mat(p, 1); // Initialized to zeros
    }
    
    if (CyclingOrder == 'u') {
        this->Order = P.Uorder;
    } else if (CyclingOrder == 'c') {
        std::vector<std::size_t> cyclic(p);
        std::iota(std::begin(cyclic), std::end(cyclic), 0);
        this->Order = cyclic;
    }
    
    this->Lows = P.Lows;
    this->Highs = P.Highs;
 
    this->CurrentIters = 0;
}

template <class T>
void CD<T>::UpdateSparse_b0(arma::vec& r){
    // Only run for regression when sparse and intercept is True.
    // r is this->r on outer scope;                                                           
    const double new_b0 = arma::mean(r);
    r -= new_b0;
    this->b0 += new_b0;
}


template <class T>
void CD<T>::UpdateBi(const std::size_t i){
    const double grd_Bi = this->GetBiGrad(i); // Gradient of Loss wrt to Bi
    // grd_Bi = <X_i, r> 
  
    // Rcpp::Rcout << "grd_Bi: "<< i <<" = " << grd_Bi <<  "\n";
    
    (*this->Xtr)[i] = std::abs(grd_Bi);  // Store absolute value of gradient for later steps
    
    const double old_Bi = this->B[i]; // old Bi to adjust residuals if Bi updates
    
    // Rcpp::Rcout << "old_Bi: "<< i <<" = " << old_Bi <<  "\n";
    
    const double nrb_Bi = this->GetBiValue(old_Bi, grd_Bi); 
    // CDL0: nrb_Bi = old_Bi + grd_Bi
    // new Bi with no regularization or bounds applied for later computations.
    // 'GetBiValue' must be implemented for each subclass of CD.
    
    // Rcpp::Rcout << "nrb_Bi: "<< i <<" = " << nrb_Bi <<  "\n";
    
    const double reg_Bi = this->GetBiReg(nrb_Bi); 
    // CDL0: reg_Bi = nrb_Bi as there is no L1, L2 parameters
    // Ideal Bi with regularization (no bounds)
    // 'GetBiReg' must be implemented for each subclass of CD.
    // Does not account for L0 
  
    // Rcpp::Rcout << "reg_Bi = "<< reg_Bi <<  "\n";
    
    const double bnd_Bi = clamp(std::copysign(reg_Bi, nrb_Bi),
                                this->Lows[i], this->Highs[i]); 
    // Ideal Bi with regularization and bounds
    
    // Rcpp::Rcout << "bnd_Bi = " << bnd_Bi <<  "\n";
    
    double new_Bi;
    
    if (i < this->NoSelectK){
        // L0 penalty is not accounted for. Only L1 and L2 (if either are used)
        // Rcpp::Rcout << "i: " << i << ", nrb_Bi:" << nrb_Bi << ", lambda1: " << this->lambda1 << "\n";
        if (std::abs(nrb_Bi) > this->lambda1){
            new_Bi = bnd_Bi;
        } else {
            new_Bi = 0;
        }
    } else if (reg_Bi < this->thr){
        // old_Bi = 0
        // grd_Bi = 1
        // nrb_Bi = 1 + 0 = 1
        // bnd_Bi = 1;
        // Lets lambda0 = 10
        
        new_Bi = 0; 
    } else { 
      // Thus reg_Bi >= this->thr 
      const double delta = std::sqrt(reg_Bi*reg_Bi - this->thr2);
      
      // Rcpp::Rcout << "delta: "<< i <<" = " << delta <<  "\n";
      const double range_Bi = std::copysign(reg_Bi, nrb_Bi);
      // Rcpp::Rcout << "Range [ "<< range_Bi - delta <<", " << range_Bi + delta  <<  "]\n";
      if ((range_Bi - delta <= bnd_Bi) && (bnd_Bi <= range_Bi + delta)){
          // Bi_wb exists in [Bi_nb - delta, Bi_nb+delta]
          // Therefore accept Bi_wb
          new_Bi = bnd_Bi;
          // Rcpp::Rcout << "NNZ Range "<< i << "\n";
      } else {
          new_Bi = 0;
      }
    }
    
    if (old_Bi != new_Bi){
        // Rcpp::Rcout << "NNZ ApplyNewBi "<< i << "\n";
        this->ApplyNewBi(i, old_Bi, new_Bi);
    }
}

template <class T>
bool CD<T>::UpdateBiCWMinCheck(const std::size_t i, const bool Cwmin){
  const double grd_Bi = this->GetBiGrad(i); // Gradient of Loss wrt to Bi
  (*this->Xtr)[i] = std::abs(grd_Bi);  // Store absolute value of gradient for later steps
  
  const double nrb_Bi = this->GetBiValue(0, grd_Bi); 
  // new Bi with no regularization or bounds applied for later computations.
  // 'GetBiValue' must be implemented for each subclass of CD.
  
  const double reg_Bi = this->GetBiReg(nrb_Bi); 
  // Ideal Bi with regularization (no bounds)
  // 'GetBiReg' must be implemented for each subclass of CD.
  // Does not account for L0 
  
  const double bnd_Bi = clamp(std::copysign(reg_Bi, nrb_Bi),
                              this->Lows[i], this->Highs[i]); 
  // Ideal Bi with regularization and bounds
  
  double new_Bi;
  
  if (i < this->NoSelectK){
    // L0 penalty is not accounted for. Only L1 and L2 (if either are used)
    if (std::abs(nrb_Bi) > this->lambda1){
      new_Bi = bnd_Bi;
    } else {
      new_Bi = 0;
    }
  } else if (reg_Bi < this->thr){
    new_Bi = 0; 
  } else{
    
    const double delta = std::sqrt(reg_Bi*reg_Bi - this->thr2);
    const double range_Bi = std::copysign(reg_Bi, nrb_Bi);
    
    if ((range_Bi - delta <= bnd_Bi) && (bnd_Bi <= range_Bi + delta)){
      // Bi_wb exists in [Bi_nb - delta, Bi_nb+delta]
      // Therefore accept Bi_wb
      new_Bi = bnd_Bi;
    } else {
      new_Bi = 0;
    }
  }
  
  if (new_Bi != 0){
    // Rcpp::Rcout << "NNZ ApplyNewBiCWMinCheck "<< i << "\n";
    this->ApplyNewBiCWMinCheck(i, 0, new_Bi);
    return false;
  } else{
    return Cwmin;
  }
}


template <class T>
bool CD<T>::Converged() {
    this->CurrentIters += 1; // keeps track of the number of calls to Converged
    double objectiveold = this->objective;
    this->objective = this->Objective();
    return std::abs(objectiveold - this->objective) <= this->Tol*objectiveold; 
}

template <class T>
void CD<T>::SupportStabilized() {
    
    bool SameSupp = true;
    
    if (this->Bprev.n_nonzero != this->B.n_nonzero) {
        SameSupp = false;
    } else {  // same number of nnz and Supp is sorted
        arma::sp_mat::const_iterator i1;
        arma::sp_mat::const_iterator i2;
        for(i1 = this->B.begin(), i2 = this->Bprev.begin(); i1 != this->B.end(); ++i1, ++i2) {
            if (i1.row() != i2.row()) {
                SameSupp = false;
                break;
            }
        }
    }
    
    if (SameSupp) {
        this->SameSuppCounter += 1;
        
        if (this->SameSuppCounter == this->ActiveSetNum - 1) {
            std::vector<std::size_t> NewOrder(this->B.n_nonzero);
            
            arma::sp_mat::const_iterator i;
            for(i = this->B.begin(); i != this->B.end(); ++i) {
                NewOrder.push_back(i.row());
            }
            
            std::sort(NewOrder.begin(), NewOrder.end(), [this](std::size_t i, std::size_t j) {return this->Order[i] <  this->Order[j] ;});
            
            this->OldOrder = this->Order;
            this->Order = NewOrder;
            this->ActiveSet = false;
            this->Stabilized = true;
            
        }
        
    } else {
        this->SameSuppCounter = 0;
    }
    
}

template class CDBase<arma::mat>;
template class CDBase<arma::sp_mat>;


template <class T>
CD<T>::CD(const T& Xi, const arma::vec& yi, const Params<T>& P) : CDBase<T>(Xi, yi, P){
    Range1p.resize(this->p);
    std::iota(std::begin(Range1p), std::end(Range1p), 0);
    ScreenSize = P.ScreenSize;
}

template class CD<arma::mat>;
template class CD<arma::sp_mat>;


template <class T>
CDSwaps<T>::CDSwaps(const T& Xi, const arma::vec& yi, const Params<T>& Pi) : CDBase<T>(Xi, yi, Pi){
    MaxNumSwaps = Pi.MaxNumSwaps;
    P = Pi;
}

template class CDSwaps<arma::mat>;
template class CDSwaps<arma::sp_mat>;
