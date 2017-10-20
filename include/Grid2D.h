#ifndef GRID2D_H
#define GRID2D_H
#include "GridParams.h"
#include "FitResult.h"

class Grid2D{
private:
	uint G_nrows;
	uint G_ncols;
	GridParams PG;
	const arma::mat * X;
	const arma::vec * y;
	uint p;
	std::vector<FitResult*> G;
	double Lambda2Max;
	double Lambda2Min;

public:
	Grid2D(const arma::mat& Xi, const arma::vec& yi, const GridParams& PGi);

	std::vector<FitResult*> Fit();

};

#endif