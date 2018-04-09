#include "Grid2D.h"
#include "Grid1D.h"

Grid2D::Grid2D(const arma::mat& Xi, const arma::vec& yi, const GridParams& PGi)
{
    // automatically selects lambda_0 (but assumes other lambdas are given in PG.P.ModelParams)
    X = &Xi;
    y = &yi;
    p = Xi.n_cols;
    PG = PGi;
    G_nrows = PG.G_nrows;
    G_ncols = PG.G_ncols;
    G.reserve(G_nrows);
    Lambda2Max = PG.Lambda2Max;
    Lambda2Min = PG.Lambda2Min;
    LambdaMinFactor = PG.LambdaMinFactor;

    P = PG.P;
}

std::vector< std::vector<FitResult*> > Grid2D::Fit()
{

    arma::vec Xtrarma;
    if (PG.P.Specs.Logistic)
    {
        Xtrarma = 0.5 * arma::abs(y->t() * *X).t(); // = gradient of logistic loss at zero
    }

    else if (PG.P.Specs.SquaredHinge)
    {
        Xtrarma = 2 * arma::abs(y->t() * *X).t(); // = gradient of loss function at zero
    }

    else
    {
        Xtrarma = arma::abs(y->t() * *X).t();
    }

    double ytXmax = arma::max(Xtrarma);

    unsigned int index;
    if (PG.P.Specs.L0L1)
    {
        index = 1;
        if(G_nrows != 1)
        {
            Lambda2Max = ytXmax;
            Lambda2Min = Lambda2Max * LambdaMinFactor;
        }

    }
    else if (PG.P.Specs.L0L2) {index = 2;}

    arma::vec Lambdas2 = arma::logspace(std::log10(Lambda2Min), std::log10(Lambda2Max), G_nrows);
    Lambdas2 = arma::flipud(Lambdas2);

    std::vector<double> Xtrvec = arma::conv_to< std::vector<double> >::from(Xtrarma);

    Xtr = new std::vector<double>(X->n_cols); // needed! careful

    PG.XtrAvailable = true;

    for(unsigned int i=0; i<Lambdas2.size();++i) //auto &l : Lambdas2
    {
        *Xtr = Xtrvec;

        PG.Xtr = Xtr;
        PG.ytXmax = ytXmax;

        PG.P.ModelParams[index] = Lambdas2[i];
        PG.Lambdas = PG.LambdasGrid[i];
        auto Gl = Grid1D(*X, *y, PG).Fit();
        G.push_back(Gl);
    }

    return G;

}
