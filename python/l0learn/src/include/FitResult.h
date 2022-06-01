#ifndef FITRESULT_H
#define FITRESULT_H
#include "BetaVector.h"
#include "arma_includes.h"

template <class T>  // Forward Reference to prevent circular dependencies
class CDBase;

template <typename T>
struct FitResult {
  double Objective;
  beta_vector B;
  CDBase<T> *Model;
  std::size_t IterNum;
  arma::vec *r;
  std::vector<double> ModelParams;
  double b0 = 0;  // used by classification models and sparse regression models
  arma::vec ExpyXB;   // Used by Logistic regression
  arma::vec onemyxb;  // Used by SquaredHinge regression
};

#endif
