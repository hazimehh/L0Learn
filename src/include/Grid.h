#ifndef GRID_H
#define GRID_H
#include <set>
#include <memory>
#include "RcppArmadillo.h"
#include "GridParams.h"
#include "FitResult.h"
#include "Grid1D.h"
#include "Grid2D.h"
#include "Normalize.h"

template <typename T>
class Grid {
    private:
        T Xscaled;
        arma::vec yscaled;
        arma::vec BetaMultiplier;
        arma::vec meanX;
        double meany;

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


// Assumes PG.P.Specs have been already set
template <typename T>
Grid<T>::Grid(const T& X, const arma::vec& y, const GridParams<T>& PGi) {
    PG = PGi;
    std::tie(Xscaled, BetaMultiplier, meanX, meany) = Normalize(X, 
             y, yscaled,!PG.P.Specs.Classification, PG.intercept);
    
    // Must rescale bounds by BetaMultiplier inorder for final result to conform to bounds
    PG.P.Lows /= BetaMultiplier;
    PG.P.Highs /= BetaMultiplier;
}

template <typename T>
void Grid<T>::Fit() {
    
    std::vector< std::vector<std::unique_ptr<FitResult<T>> > > G;
    if (PG.P.Specs.L0) {
        G.push_back(std::move(Grid1D<T>(Xscaled, yscaled, PG).Fit()));
        Lambda12.push_back(0);
    } else {
        G = std::move(Grid2D<T>(Xscaled, yscaled, PG).Fit());
    }
    
    // TODO: Scale InitialSol to be within bounds
    // B = arma::clamp(B, Low, High); 
    
    Lambda0 = std::vector< std::vector<double> >(G.size());
    NnzCount = std::vector< std::vector<std::size_t> >(G.size());
    Solutions = std::vector< std::vector<arma::sp_mat> >(G.size());
    Intercepts = std::vector< std::vector<double> >(G.size());
    Converged = std::vector< std::vector<bool> >(G.size());
    
    //for (auto &g : G)
    for (std::size_t i=0; i<G.size(); ++i) {
        if (PG.P.Specs.L0L1){ 
            Lambda12.push_back(G[i][0]->ModelParams[1]); 
        } else if (PG.P.Specs.L0L2) { 
            Lambda12.push_back(G[i][0]->ModelParams[2]); 
        }
        
        for (auto &g : G[i]) {
            
            Lambda0[i].push_back(g->ModelParams[0]);
            
            NnzCount[i].push_back(g->B.n_nonzero);
            
            if (g->IterNum != PG.P.MaxIters){
                Converged[i].push_back(true);
            } else {
                Converged[i].push_back(false);
            }
            
            arma::sp_mat B_unscaled;
            double intercept;
            
            
            if (PG.P.Specs.Classification) {
                std::tie(B_unscaled, intercept) = DeNormalize(g->B, BetaMultiplier, meanX, meany);
                Solutions[i].push_back(B_unscaled);
                Intercepts[i].push_back(g->intercept + intercept); // + needed
            } else {
                std::tie(B_unscaled, intercept) = DeNormalize(g->B, BetaMultiplier, meanX, meany);
                Solutions[i].push_back(B_unscaled);
                Intercepts[i].push_back(intercept);
            }
        }
    }
}

#endif
