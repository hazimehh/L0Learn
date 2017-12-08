#ifndef GRID_H
#define GRID_H
#include <set>
#include "GridParams.h"
#include "FitResult.h"


class Grid{
private:
	arma::mat Xscaled;
	arma::vec yscaled;
	arma::vec BetaMultiplier;
	arma::vec meanX;
	double meany;
	bool classification;

public:

	std::set<std::string> L0Models = {"L0", "L0Swaps", "L0KSwaps","L0Logistic"};
	std::set<std::string> L0L1Models = {"L0L1", "L0L1Swaps", "L0L1KSwaps","L0L1Logistic"};
	std::set<std::string> L0L2Models = {"L0L2", "L0L2Swaps", "L0L2KSwaps","L0L2Logistic"};
	std::set<std::string> L1Models = {"L1", "L1Relaxed"};
	std::set<std::string> ClassificationModels = {"L0L1Logistic", "L0L2Logistic", "L0Logistic"};


	GridParams PG;

    std::vector<double> Lambda0;
    std::vector<double> Lambda12;
    std::vector<unsigned int> NnzCount;
    std::vector<arma::sp_mat> Solutions;
    std::vector<double> Intercepts;

	Grid(const arma::mat& X, const arma::vec& y, const GridParams& PG);

	void Fit();

};

#endif
