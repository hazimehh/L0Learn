#ifndef GRID_H
#define GRID_H
#include <tuple>
#include <set>
#include <memory>
#include <armadillo>
#include "GridParams.h"
#include "FitResult.h"
#include "Grid1D.h"
#include "Grid2D.h"
#include "Normalize.h"

template <class T>
class Grid {
    private:
        T Xscaled;
        arma::vec yscaled;
        arma::vec BetaMultiplier;
        arma::vec meanX;
        double meany;
        double scaley;

    public:

        GridParams<T> PG;

        std::vector< std::vector<double> > Lambda0;
        std::vector<double> Lambda12;
        std::vector< std::vector<std::size_t> > NnzCount;
        std::vector< std::vector<arma::sp_mat> > Solutions;
        std::vector< std::vector<double> >Intercepts;
        std::vector< std::vector<bool> > Converged;

        Grid(const T& X, const arma::vec& y, const GridParams<T>& PG);
        //~Grid();

        void Fit();

};

#endif
