#ifndef CD_H
#define CD_H
//#include <armadillo>
#include "RcppArmadillo.h"
#include "FitResult.h"
#include "Params.h"


class CD {
protected:
	uint n, p;
	arma::sp_mat B;
	arma::sp_mat Bprev;
	uint SameSuppCounter = 0; 
	double objective;
	arma::vec r; //vector of residuals
	std::vector<uint> Order; // Cycling order
	std::vector<uint> OldOrder; // Cycling order to be used after support stabilization + convergence.

public:
	const arma::mat * X;
	const arma::vec * y;
	std::string ModelType;
	std::vector<double> ModelParams;

	char CyclingOrder;
	uint MaxIters;
	double Tol;
	bool ActiveSet;
	uint ActiveSetNum;
	bool Stabilized = false;


	CD(const arma::mat& Xi, const arma::vec& yi, const Params& P);

	virtual double Objective(arma::vec & r, arma::sp_mat & B) =0;

	virtual FitResult Fit() =0;

	bool Converged();

	void SupportStabilized();

	static CD * make_CD(const arma::mat& Xi, const arma::vec& yi, const Params& P);


};

#endif
