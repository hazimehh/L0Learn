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

	P = PG.P;
	P.Xtr = new std::vector<double>(X->n_cols); // needed! careful
	Xtr = P.Xtr;

}

std::vector<FitResult*> Grid2D::Fit(){

	arma::vec Lambdas2 = arma::logspace(std::log10(Lambda2Min), std::log10(Lambda2Max), G_nrows);
	Lambdas2 = arma::flipud(Lambdas2);

	uint index;
	if (PG.Type == "L0L1" || PG.Type == "L0L1Logistic" || PG.Type == "L1Relaxed"){index = 1;}
	else if (PG.Type == "L0L2" || PG.Type == "L0L2Logistic") {index = 2;}


	arma::vec Xtrarma;
	if (PG.P.ModelType == "L012Logistic"){
		Xtrarma = 0.5*arma::abs(y->t() * *X).t(); // = gradient of logistic loss at zero
	}
	else{
		Xtrarma = arma::abs(y->t() * *X).t();
	}

	*Xtr = arma::conv_to< std::vector<double> >::from(Xtrarma);
	PG.ytXmax = arma::max(Xtrarma);
	PG.XtrAvailable = true;

	for(auto &l: Lambdas2){
		PG.P.ModelParams[index] = l;
		auto Gl = Grid1D(*X, *y, PG).Fit();
		G.insert(G.end(), Gl.begin(), Gl.end());
	}

	return G;

}
