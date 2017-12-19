#ifndef CDL012LogisticSwpas_H
#define CDL012LogisticSwaps_H
#include "CD.h"
#include "FitResult.h"

class CDL012LogisticSwaps : public CD {
private:
	const double LipschitzConst = 0.25;
	double thr;
	double twolambda2;
	double qp2lamda2;
	double lambda1;
	double lambda1ol;
	double b0;
	double stl0Lc;
	arma::vec ExpyXB;
	std::vector<double> * Xtr;
	unsigned int Iter;
	FitResult result;

	unsigned int MaxNumSwaps;
	Params P;

public:
	CDL012LogisticSwaps(const arma::mat& Xi, const arma::vec& yi, const Params& P);

	FitResult Fit() final;

	inline double Objective(arma::vec & r, arma::sp_mat & B) final;


};

#endif
