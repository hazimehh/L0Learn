#ifndef CDL012Logistic_H
#define CDL012Logistic_H
#include "BetaVector.h"
#include "CD.h"
#include "FitResult.h"
#include "Params.h"
#include "arma_includes.h"
#include "utils.h"

template <class T>
class CDL012Logistic : public CD<T, CDL012Logistic<T>> {
 private:
  const double LipschitzConst = 0.25;
  double twolambda2;
  double qp2lamda2;
  double lambda1ol;
  arma::vec ExpyXB;
  // std::vector<double> * Xtr;
  T *Xy;

 public:
  CDL012Logistic(const T &Xi, const arma::vec &yi, const Params<T> &P);
  //~CDL012Logistic(){}

  FitResult<T> _FitWithBounds() final;

  FitResult<T> _Fit() final;

  inline double Objective(const arma::vec &r, const beta_vector &B) final;

  inline double Objective() final;

  inline double GetBiGrad(const std::size_t i);

  inline double GetBiValue(const double old_Bi, const double grd_Bi);

  inline double GetBiReg(const double Bi_step);

  inline void ApplyNewBi(const std::size_t i, const double Bi_old,
                         const double Bi_new);

  inline void ApplyNewBiCWMinCheck(const std::size_t i, const double old_Bi,
                                   const double new_Bi);
};

template <class T>
inline double CDL012Logistic<T>::GetBiGrad(const std::size_t i) {
  /*
   * Notes:
   *      When called in CWMinCheck, we know that this->B[i] is 0.
   */
  return -arma::dot(matrix_column_get(*(this->Xy), i), 1 / (1 + ExpyXB)) +
         twolambda2 * this->B[i];
  // return -arma::sum( matrix_column_get(*(this->Xy), i) / (1 + ExpyXB) ) +
  // twolambda2 * this->B[i];
}

template <class T>
inline double CDL012Logistic<T>::GetBiValue(const double old_Bi,
                                            const double grd_Bi) {
  return old_Bi - grd_Bi / qp2lamda2;
}

template <class T>
inline double CDL012Logistic<T>::GetBiReg(const double Bi_step) {
  return std::abs(Bi_step) - lambda1ol;
}

template <class T>
inline void CDL012Logistic<T>::ApplyNewBi(const std::size_t i,
                                          const double old_Bi,
                                          const double new_Bi) {
  ExpyXB %= arma::exp((new_Bi - old_Bi) * matrix_column_get(*(this->Xy), i));
  this->B[i] = new_Bi;
}

template <class T>
inline void CDL012Logistic<T>::ApplyNewBiCWMinCheck(const std::size_t i,
                                                    const double old_Bi,
                                                    const double new_Bi) {
  ExpyXB %= arma::exp((new_Bi - old_Bi) * matrix_column_get(*(this->Xy), i));
  this->B[i] = new_Bi;
  this->Order.push_back(i);
}

template <class T>
inline double CDL012Logistic<T>::Objective(
    const arma::vec &expyXB,
    const beta_vector &B) {  // hint inline
  const auto l2norm = arma::norm(B, 2);
  // arma::sum(arma::log(1 + 1 / expyXB)) is the negative log-likelihood
  return arma::sum(arma::log(1 + 1 / expyXB)) + this->lambda0 * n_nonzero(B) +
         this->lambda1 * arma::norm(B, 1) + this->lambda2 * l2norm * l2norm;
}

template <class T>
inline double CDL012Logistic<T>::Objective() {
  return this->Objective(ExpyXB, this->B);
}

template <class T>
CDL012Logistic<T>::CDL012Logistic(const T &Xi, const arma::vec &yi,
                                  const Params<T> &P)
    : CD<T, CDL012Logistic<T>>(Xi, yi, P) {
  twolambda2 = 2 * this->lambda2;
  qp2lamda2 =
      (LipschitzConst + twolambda2);  // this is the univariate lipschitz const
                                      // of the differentiable objective
  this->thr2 = (2 * this->lambda0) / qp2lamda2;
  this->thr = std::sqrt(this->thr2);
  lambda1ol = this->lambda1 / qp2lamda2;

  ExpyXB =
      arma::exp(this->y % (*(this->X) * this->B +
                           this->b0));  // Maintained throughout the algorithm
  Xy = P.Xy;
}

template <class T>
FitResult<T> CDL012Logistic<T>::_Fit() {
  this->objective = Objective();  // Implicitly used ExpyXB

  std::vector<std::size_t> FullOrder = this->Order;  // never used in LR
  this->Order.resize(
      std::min((int)(n_nonzero(this->B) + this->ScreenSize + this->NoSelectK),
               (int)(this->p)));

  for (std::size_t t = 0; t < this->MaxIters; ++t) {
    this->Bprev = this->B;

    // Update the intercept
    if (this->intercept) {
      const double b0old = this->b0;
      // const double partial_b0 = - arma::sum( *(this->y) / (1 + ExpyXB) );
      const double partial_b0 = -arma::dot((this->y), 1 / (1 + ExpyXB));
      this->b0 -= partial_b0 /
                  (this->n * LipschitzConst);  // intercept is not regularized
      ExpyXB %= arma::exp((this->b0 - b0old) * (this->y));
    }

    for (auto &i : this->Order) {
      this->UpdateBi(i);
    }

    this->RestrictSupport();

    // only way to terminate is by (i) converging on active set and (ii)
    // CWMinCheck
    if (this->isConverged() && this->CWMinCheck()) {
      break;
    }
  }

  this->result.Objective = this->objective;
  this->result.B = this->B;
  this->result.Model = this;
  this->result.b0 = this->b0;
  this->result.ExpyXB = ExpyXB;
  this->result.IterNum = this->CurrentIters;

  return this->result;
}

template <class T>
FitResult<T> CDL012Logistic<T>::_FitWithBounds() {  // always uses active sets

  // arma::sp_mat B2 = this->B;
  clamp_by_vector(this->B, this->Lows, this->Highs);

  this->objective = Objective();  // Implicitly used ExpyXB

  std::vector<std::size_t> FullOrder = this->Order;  // never used in LR
  this->Order.resize(
      std::min((int)(n_nonzero(this->B) + this->ScreenSize + this->NoSelectK),
               (int)(this->p)));

  for (std::size_t t = 0; t < this->MaxIters; ++t) {
    this->Bprev = this->B;

    // Update the intercept
    if (this->intercept) {
      const double b0old = this->b0;
      // const double partial_b0 = - arma::sum( *(this->y) / (1 + ExpyXB) );
      const double partial_b0 = -arma::dot((this->y), 1 / (1 + ExpyXB));
      this->b0 -= partial_b0 /
                  (this->n * LipschitzConst);  // intercept is not regularized
      ExpyXB %= arma::exp((this->b0 - b0old) * (this->y));
    }

    for (auto &i : this->Order) {
      this->UpdateBiWithBounds(i);
    }

    this->RestrictSupport();

    // only way to terminate is by (i) converging on active set and (ii)
    // CWMinCheck
    if (this->isConverged() && this->CWMinCheckWithBounds()) {
      break;
    }
  }

  this->result.Objective = this->objective;
  this->result.B = this->B;
  this->result.Model = this;
  this->result.b0 = this->b0;
  this->result.ExpyXB = ExpyXB;
  this->result.IterNum = this->CurrentIters;

  return this->result;
}

template class CDL012Logistic<arma::mat>;
template class CDL012Logistic<arma::sp_mat>;

#endif
