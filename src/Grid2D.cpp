#include "Grid2D.h"
#include "Grid1D.h"

Grid2D::Grid2D(const arma::mat& Xi, const arma::vec& yi, const GridParams& PGi){
	// automatically selects lambda_0 (but assumes other lambdas are given in PG.P.ModelParams)
	X = &Xi;
	y = &yi;
	p = Xi.n_cols;
	PG = PGi;
	G_nrows = PG.G_nrows;
	G_ncols = PG.G_ncols;
	G.reserve(G_nrows*G_ncols);
	Lambda2Max = PG.Lambda2Max;
	Lambda2Min = PG.Lambda2Min;
}

std::vector<FitResult*> Grid2D::Fit(){

	arma::vec Lambdas2 = arma::logspace(std::log10(Lambda2Min), std::log10(Lambda2Max), G_nrows);
	Lambdas2 = arma::flipud(Lambdas2);

	for(auto &l: Lambdas2){
		if (PG.Type == "L0L1"){PG.P.ModelParams[1] = l;}
		else if (PG.Type == "L0L2") {PG.P.ModelParams[2] = l;}
		else if (PG.Type == "L1Relaxed") {PG.P.ModelParams[1] = l;}

		auto Gl = Grid1D(*X, *y, PG).Fit();
		G.insert(G.end(), Gl.begin(), Gl.end());
		
	}

	return G;

}
