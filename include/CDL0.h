#ifndef CDL0_H
#define CDL0_H
#include "CD.h"
#include "FitResult.h" // REMOVE: may need to include params??

class CDL0 : public CD {
private:
	double thr;
	std::vector<double> * Xtr;
	unsigned int Iter;
	FitResult result;
public:
	CDL0(const arma::mat& Xi, const arma::vec& yi, const Params& P);

	FitResult Fit() final;

	double Objective(arma::vec & r, arma::sp_mat & B) final;
};

#endif
