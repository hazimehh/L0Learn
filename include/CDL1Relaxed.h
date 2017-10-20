#ifndef CDL1RELAXED_H
#define CDL1RELAXED_H
#include "CD.h"
#include "FitResult.h"

class CDL1Relaxed : public CD {
private:
	double thr;
	std::vector<double> * Xtr;
	FitResult result;

public:
	CDL1Relaxed(const arma::mat& Xi, const arma::vec& yi, const Params& P);

	FitResult Fit() final;

	double Objective(arma::vec & r, arma::sp_mat & B) final;

};

#endif