#ifndef FITRESULT_H
#define FITRESULT_H
#include "RcppArmadillo.h"

template <class T> // Forward Reference to prevent circular dependencies
class CD;

template <typename T>
struct FitResult {
    double Objective;
    arma::sp_mat B;
    CD<T> * Model;
    std::size_t IterNum;
    arma::vec * r;
    std::vector<double> ModelParams;
    double intercept = 0; // used by classification models
    arma::vec ExpyXB; // Used by logistic regression
};

#endif
