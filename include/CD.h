#ifndef CD_H
#define CD_H
#include <armadillo>
//#include "RcppArmadillo.h"
#include "FitResult.h"
#include "Params.h"


class CD {
protected:
	unsigned int n, p;
	arma::sp_mat B;
	arma::sp_mat Bprev;
	unsigned int SameSuppCounter = 0;
	double objective;
	arma::vec r; //vector of residuals
	std::vector<unsigned int> Order; // Cycling order
	std::vector<unsigned int> OldOrder; // Cycling order to be used after support stabilization + convergence.

public:
	const arma::mat * X;
	const arma::vec * y;
	std::string ModelType;
	std::vector<double> ModelParams;

	char CyclingOrder;
	unsigned int MaxIters;
	double Tol;
	bool ActiveSet;
	unsigned int ActiveSetNum;
	bool Stabilized = false;


	CD(const arma::mat& Xi, const arma::vec& yi, const Params& P);

	virtual double Objective(arma::vec & r, arma::sp_mat & B) =0;

	virtual FitResult Fit() =0;

	bool Converged();

	void SupportStabilized();

	static CD * make_CD(const arma::mat& Xi, const arma::vec& yi, const Params& P);


};

#endif
