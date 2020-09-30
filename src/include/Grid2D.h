#ifndef GRID2D_H
#define GRID2D_H
#include <memory>
#include "RcppArmadillo.h"
#include "GridParams.h"
#include "Params.h"
#include "FitResult.h"
#include "Grid1D.h"
#include "utils.h"

template<class T>
class Grid2D
{
    private:
        std::size_t G_nrows;
        std::size_t G_ncols;
        GridParams<T> PG;
        const T * X;
        const arma::vec * y;
        std::size_t p;
        std::vector<std::vector<std::unique_ptr<FitResult<T>>>> G; 
        // each inner vector corresponds to a single lambda_1/lambda_2
        
        double Lambda2Max;
        double Lambda2Min;
        double LambdaMinFactor;
        std::vector<double> * Xtr;
        Params<T> P;


    public:
        Grid2D(const T& Xi, const arma::vec& yi, const GridParams<T>& PGi);
        ~Grid2D();
        std::vector<std::vector<std::unique_ptr<FitResult<T>>>> Fit();

};


#endif
