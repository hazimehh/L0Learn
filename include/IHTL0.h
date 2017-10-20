#ifndef IHTLO_H
#define IHTL0_H
#include "CD.h"

class IHTL0 : public CD {
private:
	double thr;
	double alpha;

public:
	IHTL0(const arma::mat& Xi, const arma::vec& yi, const Params& P);

	FitResult Fit() final;

	double Objective(arma::vec & r, arma::sp_mat & B) final;

};

#endif