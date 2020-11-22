#include "CD.h"

#include <chrono>
#include <thread>

/*
 * 
 *  CDBase
 * 
 */

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
    
    this->withBounds = P.withBounds;
    
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
FitResult<T> CDBase<T>::Fit(){
  if (this->withBounds){
    return this->_FitWithBounds();
  } else{
    return this->_Fit();
  }
}

template class CDBase<arma::mat>;
template class CDBase<arma::sp_mat>;

/*
 * 
 *  CD
 * 
 */

template <class T>
void CD<T>::UpdateSparse_b0(arma::vec& r){
    // Only run for regression when T is arma::sp_mat and intercept is True.
    // r is this->r on outer scope;                                                           
    const double new_b0 = arma::mean(r);
    r -= new_b0;
    this->b0 += new_b0;
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

template <class T>
bool CD<T>::CWMinCheckWithBounds() {
    std::vector<std::size_t> S;
    for(arma::sp_mat::const_iterator it = this->B.begin(); it != this->B.end(); ++it) {
        S.push_back(it.row());
    }
    
    std::vector<std::size_t> Sc;
    set_difference(
        this->Range1p.begin(),
        this->Range1p.end(),
        S.begin(),
        S.end(),
        back_inserter(Sc));
    
    bool Cwmin = true;
    for (auto& i : Sc) {
        Cwmin = this->UpdateBiCWMinCheckWithBounds(i, Cwmin);
    }
    return Cwmin;
}

template <class T>
bool CD<T>::CWMinCheck() {
  std::vector<std::size_t> S;
  for(arma::sp_mat::const_iterator it = this->B.begin(); it != this->B.end(); ++it) {
    S.push_back(it.row());
  }
  
  std::vector<std::size_t> Sc;
  set_difference(
    this->Range1p.begin(),
    this->Range1p.end(),
    S.begin(),
    S.end(),
    back_inserter(Sc));
  
  bool Cwmin = true;
  for (auto& i : Sc) {
    Rcpp::Rcout << "CW Iteration: " << i << "\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    Cwmin = this->UpdateBiCWMinCheck(i, Cwmin);
  }
  
  return Cwmin;
}
  

template <class T>
CD<T>::CD(const T& Xi, const arma::vec& yi, const Params<T>& P) : CDBase<T>(Xi, yi, P){
    Range1p.resize(this->p);
    std::iota(std::begin(Range1p), std::end(Range1p), 0);
    ScreenSize = P.ScreenSize;
}

template class CD<arma::mat>;
template class CD<arma::sp_mat>;


/*
 * 
 *  CDSwaps
 * 
 */

template <class T>
CDSwaps<T>::CDSwaps(const T& Xi, const arma::vec& yi, const Params<T>& Pi) : CDBase<T>(Xi, yi, Pi){
    MaxNumSwaps = Pi.MaxNumSwaps;
    P = Pi;
}

template class CDSwaps<arma::mat>;
template class CDSwaps<arma::sp_mat>;
