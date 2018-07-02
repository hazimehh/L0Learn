#ifndef GRID2D_H
#define GRID2D_H
#include "GridParams.h"
#include "FitResult.h"

class Grid2D
{
    private:
        unsigned int G_nrows;
        unsigned int G_ncols;
        GridParams PG;
        const arma::mat * X;
        const arma::vec * y;
        unsigned int p;
        std::vector< std::vector<FitResult*> > G; // each inner vector corresponds to a single lambda_1/lambda_2
        double Lambda2Max;
        double Lambda2Min;
        double LambdaMinFactor;
        std::vector<double> * Xtr;
        Params P;


    public:
        Grid2D(const arma::mat& Xi, const arma::vec& yi, const GridParams& PGi);
        ~Grid2D();
        std::vector< std::vector<FitResult*> > Fit();

};

#endif
