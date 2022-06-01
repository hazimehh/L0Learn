#ifndef GRID2D_H
#define GRID2D_H
#include <memory>

#include "FitResult.h"
#include "Grid1D.h"
#include "GridParams.h"
#include "Params.h"
#include "arma_includes.h"
#include "utils.h"

template <class T>
class Grid2D {
 private:
  std::size_t G_nrows;
  std::size_t G_ncols;
  GridParams<T> PG;
  const T *X;
  const arma::vec *y;
  std::size_t p;
  std::vector<std::vector<std::unique_ptr<FitResult<T>>>> G;
  // each inner vector corresponds to a single lambda_1/lambda_2

  double Lambda2Max;
  double Lambda2Min;
  double LambdaMinFactor;
  std::vector<double> *Xtr;
  Params<T> P;

 public:
  Grid2D(const T &Xi, const arma::vec &yi, const GridParams<T> &PGi);
  ~Grid2D();
  std::vector<std::vector<std::unique_ptr<FitResult<T>>>> Fit();
};

template <class T>
Grid2D<T>::Grid2D(const T &Xi, const arma::vec &yi, const GridParams<T> &PGi) {
  // automatically selects lambda_0 (but assumes other lambdas are given in
  // PG.P.ModelParams)
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

template <class T>
Grid2D<T>::~Grid2D() {
  delete Xtr;
  if (PG.P.Specs.Logistic) delete PG.P.Xy;
  if (PG.P.Specs.SquaredHinge) delete PG.P.Xy;
}

template <class T>
std::vector<std::vector<std::unique_ptr<FitResult<T>>>> Grid2D<T>::Fit() {
  arma::vec Xtrarma;

  if (PG.P.Specs.Logistic) {
    auto n = X->n_rows;
    double b0 = 0;
    arma::vec ExpyXB = arma::ones<arma::vec>(n);
    if (PG.intercept) {
      for (std::size_t t = 0; t < 50; ++t) {
        double partial_b0 = -arma::sum(*y / (1 + ExpyXB));
        b0 -= partial_b0 / (n * 0.25);  // intercept is not regularized
        ExpyXB = arma::exp(b0 * *y);
      }
    }
    PG.P.b0 = b0;
    Xtrarma = arma::abs(-arma::trans(*y / (1 + ExpyXB)) * *X)
                  .t();  // = gradient of logistic loss at zero
    // Xtrarma = 0.5 * arma::abs(y->t() * *X).t(); // = gradient of logistic
    // loss at zero

    T Xy = matrix_vector_schur_product(*X, y);  // X->each_col() % *y;

    PG.P.Xy = new T;
    *PG.P.Xy = Xy;
  }

  else if (PG.P.Specs.SquaredHinge) {
    auto n = X->n_rows;
    double b0 = 0;
    arma::vec onemyxb = arma::ones<arma::vec>(n);
    arma::uvec indices = arma::find(onemyxb > 0);
    if (PG.intercept) {
      for (std::size_t t = 0; t < 50; ++t) {
        double partial_b0 =
            arma::sum(2 * onemyxb.elem(indices) % (-y->elem(indices)));
        b0 -= partial_b0 / (n * 2);  // intercept is not regularized
        onemyxb = 1 - (*y * b0);
        indices = arma::find(onemyxb > 0);
      }
    }
    PG.P.b0 = b0;
    T indices_rows = matrix_rows_get(*X, indices);
    Xtrarma =
        2 * arma::abs(arma::trans(y->elem(indices) % onemyxb.elem(indices)) *
                      indices_rows)
                .t();  // = gradient of loss function at zero
    // Xtrarma = 2 * arma::abs(y->t() * *X).t(); // = gradient of loss function
    // at zero
    T Xy = matrix_vector_schur_product(*X, y);  // X->each_col() % *y;
    PG.P.Xy = new T;
    *PG.P.Xy = Xy;
  } else {
    Xtrarma = arma::abs(y->t() * *X).t();
  }

  double ytXmax = arma::max(Xtrarma);

  std::size_t index;
  if (PG.P.Specs.L0L1) {
    index = 1;
    if (G_nrows != 1) {
      Lambda2Max = ytXmax;
      Lambda2Min = Lambda2Max * LambdaMinFactor;
    }
  } else if (PG.P.Specs.L0L2) {
    index = 2;
  }

  arma::vec Lambdas2 =
      arma::logspace(std::log10(Lambda2Min), std::log10(Lambda2Max), G_nrows);
  Lambdas2 = arma::flipud(Lambdas2);

  std::vector<double> Xtrvec =
      arma::conv_to<std::vector<double>>::from(Xtrarma);

  Xtr = new std::vector<double>(X->n_cols);  // needed! careful

  PG.XtrAvailable = true;
  // Rcpp::Rcout << "Grid2D Start\n";
  for (std::size_t i = 0; i < Lambdas2.size(); ++i) {  // auto &l : Lambdas2
    // Rcpp::Rcout << "Grid1D Start: " << i << "\n";
    *Xtr = Xtrvec;

    PG.Xtr = Xtr;
    PG.ytXmax = ytXmax;

    PG.P.ModelParams[index] = Lambdas2[i];

    if (PG.LambdaU == true) PG.Lambdas = PG.LambdasGrid[i];

    // std::vector<std::unique_ptr<FitResult>> Gl();
    // auto Gl = Grid1D(*X, *y, PG).Fit();
    //  Rcpp::Rcout << "Grid1D Start: " << i << "\n";
    G.push_back(std::move(Grid1D<T>(*X, *y, PG).Fit()));
  }

  return std::move(G);
}

#endif
