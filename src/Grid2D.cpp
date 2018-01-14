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
}

std::vector<FitResult*> Grid2D::Fit(){

	arma::vec Lambdas2 = arma::logspace(std::log10(Lambda2Min), std::log10(Lambda2Max), G_nrows);
	Lambdas2 = arma::flipud(Lambdas2);

	unsigned int index;
	if (PG.Type == "L0L1" || PG.Type == "L0L1Logistic" || PG.Type == "L0L1Classification" || PG.Type == "L1Relaxed"){index = 1;}
	else if (PG.Type == "L0L2" || PG.Type == "L0L2Logistic" || PG.Type == "L0L2SquaredHinge") {index = 2;}


	arma::vec Xtrarma;
	if (PG.P.ModelType == "L012Logistic" || PG.P.ModelType == "L012LogisticSwaps"){
		Xtrarma = 0.5*arma::abs(y->t() * *X).t(); // = gradient of logistic loss at zero
	}

	else if (PG.P.ModelType == "L012SquaredHinge" || PG.P.ModelType == "L012SquaredHingeSwaps"){
		Xtrarma = 2*arma::abs(y->t() * *X).t(); // = gradient of loss function at zero
	}

	else{
		Xtrarma = arma::abs(y->t() * *X).t();
	}

	double ytXmax = arma::max(Xtrarma);

	std::vector<double> Xtrvec = arma::conv_to< std::vector<double> >::from(Xtrarma);

	Xtr = new std::vector<double>(X->n_cols); // needed! careful

	PG.XtrAvailable = true;

	for(auto &l: Lambdas2){
		*Xtr = Xtrvec;

		PG.Xtr = Xtr;
		PG.ytXmax = ytXmax;

		PG.P.ModelParams[index] = l;
		auto Gl = Grid1D(*X, *y, PG).Fit();
		G.insert(G.end(), Gl.begin(), Gl.end());
	}

	return G;

}
