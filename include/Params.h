#ifndef PARAMS_H
#define PARAMS_H
#include <armadillo>
//#include "RcppArmadillo.h"

struct Params{
	std::string ModelType = "L0";
	std::vector<double> ModelParams {0,0,0,2};
	unsigned int MaxIters = 500;
	double Tol = 1e-8;
	char Init = 'z';
	unsigned int RandomStartSize = 10;
	arma::sp_mat * InitialSol;
	char CyclingOrder = 'c';
	std::vector<unsigned int> Uorder;
	bool ActiveSet = true;
	unsigned int ActiveSetNum = 6;
	unsigned int MaxNumSwaps = 200; // Used by CDSwaps
	std::vector<double> * Xtr;
	unsigned int Iter =0; // Current iteration number in the grid
};

#endif
