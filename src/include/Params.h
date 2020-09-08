#ifndef PARAMS_H
#define PARAMS_H
#include <map>
#include "RcppArmadillo.h"
#include "Model.h"

template <typename T>
struct Params {

    Model Specs;
    //std::string ModelType = "L0";
    std::vector<double> ModelParams {0, 0, 0, 2};
    std::size_t MaxIters = 500;
    double Tol = 1e-8;
    char Init = 'z';
    std::size_t RandomStartSize = 10;
    arma::sp_mat * InitialSol;
    double b0 = 0; // intercept
    char CyclingOrder = 'c';
    std::vector<std::size_t> Uorder;
    bool ActiveSet = true;
    std::size_t ActiveSetNum = 6;
    std::size_t MaxNumSwaps = 200; // Used by CDSwaps
    std::vector<double> * Xtr;
    arma::rowvec * ytX;
    std::map<std::size_t, arma::rowvec> * D;
    std::size_t Iter = 0; // Current iteration number in the grid
    std::size_t ScreenSize = 1000;
    arma::vec * r;
    T * Xy; // used for classification.
    std::size_t NoSelectK = 0;
    bool intercept;
    // double Low;
    // double High;
    arma::vec Lows;
    arma::vec Highs;
};

#endif
