#ifndef GRIDPARAMS_H
#define GRIDPARAMS_H
#include "RcppArmadillo.h"
#include "Params.h"

template <typename T>
struct GridParams
{
    Params<T> P;
    std::size_t G_ncols = 100;
    std::size_t G_nrows = 10;
    bool LambdaU = false;
    std::size_t NnzStopNum = 200;
    double LambdaMinFactor = 0.01;
    arma::vec Lambdas;
    std::vector< std::vector<double> > LambdasGrid;
    double Lambda2Max = 0.1;
    double Lambda2Min = 0.001;
    std::string Type = "L0";
    bool PartialSort = true;
    bool Refine = false;
    bool XtrAvailable = false;
    double ytXmax;
    std::vector<double> * Xtr;
    double ScaleDownFactor = 0.8;
    bool intercept;
};

#endif
