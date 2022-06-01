#ifndef GRID_H
#define GRID_H
#include <memory>
#include <set>
#include <tuple>

#include "FitResult.h"
#include "Grid1D.h"
#include "Grid2D.h"
#include "GridParams.h"
#include "Normalize.h"
#include "arma_includes.h"

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

  std::vector<std::vector<double>> Lambda0;
  std::vector<double> Lambda12;
  std::vector<std::vector<std::size_t>> NnzCount;
  std::vector<std::vector<arma::sp_mat>> Solutions;
  std::vector<std::vector<double>> Intercepts;
  std::vector<std::vector<bool>> Converged;

  Grid(const T &X, const arma::vec &y, const GridParams<T> &PG);
  //~Grid();

  void Fit();
};

template <class T>
Grid<T>::Grid(const T &X, const arma::vec &y, const GridParams<T> &PGi) {
  PG = PGi;

  std::tie(BetaMultiplier, meanX, meany, scaley) = Normalize(
      X, y, Xscaled, yscaled, !PG.P.Specs.Classification, PG.intercept);

  // Must rescale bounds by BetaMultiplier in order for final result to conform
  // to bounds
  if (PG.P.withBounds) {
    PG.P.Lows /= BetaMultiplier;
    PG.P.Highs /= BetaMultiplier;
  }
}

template <class T>
void Grid<T>::Fit() {
  std::vector<std::vector<std::unique_ptr<FitResult<T>>>> G;

  if (PG.P.Specs.L0) {
    G.push_back(std::move(Grid1D<T>(Xscaled, yscaled, PG).Fit()));
    Lambda12.push_back(0);
  } else {
    G = std::move(Grid2D<T>(Xscaled, yscaled, PG).Fit());
  }

  Lambda0 = std::vector<std::vector<double>>(G.size());
  NnzCount = std::vector<std::vector<std::size_t>>(G.size());
  Solutions = std::vector<std::vector<arma::sp_mat>>(G.size());
  Intercepts = std::vector<std::vector<double>>(G.size());
  Converged = std::vector<std::vector<bool>>(G.size());

  for (std::size_t i = 0; i < G.size(); ++i) {
    if (PG.P.Specs.L0L1) {
      Lambda12.push_back(G[i][0]->ModelParams[1]);
    } else if (PG.P.Specs.L0L2) {
      Lambda12.push_back(G[i][0]->ModelParams[2]);
    }

    for (auto &g : G[i]) {
      Lambda0[i].push_back(g->ModelParams[0]);

      NnzCount[i].push_back(n_nonzero(g->B));

      Converged[i].push_back(g->IterNum != PG.P.MaxIters);

      beta_vector B_unscaled;
      double b0;

      std::tie(B_unscaled, b0) =
          DeNormalize(g->B, BetaMultiplier, meanX, meany);
      Solutions[i].push_back(arma::sp_mat(B_unscaled));
      /* scaley is 1 for classification problems.
       *  g->intercept is 0 unless specifically optimized for in:
       *       classification
       *       sparse regression and intercept = true
       */
      Intercepts[i].push_back(scaley * g->b0 + b0);
    }
  }
}

#endif
