#ifndef FITRESULT_H
#define FITRESULT_H
#include "RcppArmadillo.h"

template <typename T> // Forward Reference to prevent circular dependencies
class CD;

template <typename T>
struct FitResult {
    double Objective;
    arma::sp_mat B;
    CD<T> * Model;
    unsigned int IterNum;
    arma::vec * r;
    std::vector<double> ModelParams;
    double intercept = 0; // used by classification models
    arma::vec ExpyXB; // Used by logistic regression
};

#endif
