#ifndef FITRESULT_H
#define FITRESULT_H

class CD;

struct FitResult
{
    double Objective;
    arma::sp_mat B;
    CD * Model;
    unsigned int IterNum;
    arma::vec * r;
    std::vector<double> ModelParams;
    double intercept = 0; // used by classification models
    arma::vec ExpyXB; // Used by logistic regression
};

#endif
