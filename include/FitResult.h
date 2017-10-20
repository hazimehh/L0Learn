#ifndef FITRESULT_H
#define FITRESULT_H

class CD;

struct FitResult
{
	double Objective;
	arma::sp_mat B;
	CD * Model;
	uint IterNum;
	arma::vec r;
	std::vector<double> ModelParams;
};

#endif