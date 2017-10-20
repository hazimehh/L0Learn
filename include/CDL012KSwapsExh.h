#ifndef CDLKSwapsExh_H
#define CDLKSwapsExh_H
#include "CD.h"


class CDL012KSwapsExh : public CD { // Calls CDL012 irrespective of P.ModelType
private:
	uint MaxNumSwaps;
	Params P;
	uint K;
public:
	CDL012KSwapsExh(const arma::mat& Xi, const arma::vec& yi, const Params& Pi);

	FitResult Fit() final ;

	double Objective(arma::vec & r, arma::sp_mat & B) final;

};

#endif