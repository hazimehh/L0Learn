#ifndef PARAMS_H
#define PARAMS_H
//#include <armadillo>
#include "RcppArmadillo.h"

struct Params{
	std::string ModelType = "L0";
	std::vector<double> ModelParams {0,0,0,2};
	uint MaxIters = 500; 
	double Tol = 1e-8;
	char Init = 'z';
	uint RandomStartSize = 10;
	arma::sp_mat * InitialSol;
	char CyclingOrder = 'c';
	std::vector<uint> Uorder;
	bool ActiveSet = true;
	uint ActiveSetNum = 6;
	uint MaxNumSwaps = 200; // Used by CDSwaps
	std::vector<double> * Xtr;
	uint Iter =0; // Current iteration number in the grid
};

#endif
