#ifndef GRIDPARAMS_H
#define GRIDPARAMS_H
#include "Params.h"

struct GridParams
{
	Params P;
	uint G_ncols = 100;
	uint G_nrows = 10;
	bool LambdaU;
	uint NnzStopNum = 200;
	double LambdaMinFactor = 0.01;
	arma::vec Lambdas;
	double Lambda2Max = 0.1;
	double Lambda2Min = 0.001;
	std::string Type = "L0";
	bool PartialSort = true;
	bool Refine = false;
};

#endif
