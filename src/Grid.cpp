#include "Grid.h"
#include "Grid1D.h"
#include "Grid2D.h"
#include "Normalize.h"
#include <memory>

// Assumes PG.P.Specs have been already set
Grid::Grid(const arma::mat& X, const arma::vec& y, const GridParams& PGi)
{
    PG = PGi;
    std::tie(BetaMultiplier, meanX, meany) = Normalize(X, y, Xscaled, yscaled, !PG.P.Specs.Classification, PG.intercept);
}

void Grid::Fit()
{

    std::vector< std::vector<std::unique_ptr<FitResult> > > G;
    if (PG.P.Specs.L0)
    {
        G.push_back(std::move(Grid1D(Xscaled, yscaled, PG).Fit()));
        Lambda12.push_back(0);
    }
    else
    {
        G = std::move(Grid2D(Xscaled, yscaled, PG).Fit());
    }

    Lambda0 = std::vector< std::vector<double> >(G.size());
    NnzCount = std::vector< std::vector<unsigned int> >(G.size());
    Solutions = std::vector< std::vector<arma::sp_mat> >(G.size());
    Intercepts = std::vector< std::vector<double> >(G.size());
    Converged = std::vector< std::vector<bool> >(G.size());



    //for (auto &g : G)
    for (unsigned int i=0; i<G.size(); ++i)
    {
        if (PG.P.Specs.L0L1)
        { Lambda12.push_back(G[i][0]->ModelParams[1]); }
        else if (PG.P.Specs.L0L2)
        { Lambda12.push_back(G[i][0]->ModelParams[2]); }

        for (auto &g : G[i])
        {

            Lambda0[i].push_back(g->ModelParams[0]);

            NnzCount[i].push_back(g->B.n_nonzero);

            if (g->IterNum != PG.P.MaxIters)
            {
                Converged[i].push_back(true);
            }
            else
            {
                Converged[i].push_back(false);
            }

            arma::sp_mat B_unscaled;
            double intercept;


            if (PG.P.Specs.Classification)
            {
                std::tie(B_unscaled, intercept) = DeNormalize(g->B, BetaMultiplier, meanX, meany);
                Solutions[i].push_back(B_unscaled);
                Intercepts[i].push_back(g->intercept + intercept); // + needed
            }

            else
            {
                std::tie(B_unscaled, intercept) = DeNormalize(g->B, BetaMultiplier, meanX, meany);
                Solutions[i].push_back(B_unscaled);
                Intercepts[i].push_back(intercept);
            }


        }

    }
}

/*
// Bypass for R interface
int main(){
	arma::mat X;
	X.load("/Users/hh/Desktop/Research/L0Learn/src/X_training.csv");
	arma::vec y;
	y.load("/Users/hh/Desktop/Research/L0Learn/src/y_training.csv");

  std::string Penalty = "L0";
  std::string Loss = "SquaredError";
  std::string Algorithm = "CD";

  auto p = X.n_cols;
  GridParams PG;
  PG.NnzStopNum = 100;
  PG.G_ncols = 100;
  PG.G_nrows = 1;
  PG.Lambda2Max = 0.001;
  PG.Lambda2Min = 0.001;
  PG.LambdaMinFactor = 0.001; //
  PG.PartialSort = true;
  PG.ScaleDownFactor = 0.8;
  PG.LambdaU = false;
  PG.intercept = false;
  Params P;
  P.MaxIters = 200;
  P.Tol = 1e-6;
  P.ActiveSet = true;
  P.ActiveSetNum = 3;
  P.MaxNumSwaps = 100;
  P.ScreenSize = 1000;
  P.NoSelectK = 0;
  PG.P = P;

  if (Loss == "SquaredError") {PG.P.Specs.SquaredError = true;}
  else if (Loss == "Logistic") {PG.P.Specs.Logistic = true; PG.P.Specs.Classification = true;}
  else if (Loss == "SquaredHinge") {PG.P.Specs.SquaredHinge = true; PG.P.Specs.Classification = true;}

  if (Algorithm == "CD") {PG.P.Specs.CD = true;}
  else if (Algorithm == "CDPSI") {PG.P.Specs.PSI = true;}

  if (Penalty == "L0") { PG.P.Specs.L0 = true;}
  else if (Penalty == "L0L2") { PG.P.Specs.L0L2 = true;}
  else if (Penalty == "L0L1") { PG.P.Specs.L0L1 = true;}

  Grid G(X, y, PG);
  G.Fit();

	return 0;
  }
  */
