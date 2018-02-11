#include "Grid.h"
#include "Grid1D.h"
#include "Grid2D.h"
#include "Normalize.h"

#include <chrono> // Remove

Grid::Grid(const arma::mat& X, const arma::vec& y, const GridParams& PGi){
	PG = PGi;

	std::string Type = PG.Type;
	if (ClassificationModels.find(Type) != ClassificationModels.end()){
		classification = true;
	}



	std::tie(BetaMultiplier, meanX, meany) = Normalize(X,y, Xscaled, yscaled, false); // Don't normalize y //!classification

}

void Grid::Fit()
{

    std::vector<FitResult*> G;

    std::string Type = PG.Type;

	if (Type == "L0"){
		PG.P.ModelType = "L0";
		PG.Type = "L0";
	}

	else if (Type == "L0Logistic"){
		PG.P.ModelType = "L012Logistic";
		PG.Type = "L0Logistic";
	}

	else if (Type == "L0SquaredHinge"){
		PG.P.ModelType = "L012SquaredHinge";
		PG.Type = "L0SquaredHinge";
	}

	else if (Type == "L0LogisticSwaps"){
		PG.P.ModelType = "L012LogisticSwaps";
		PG.Type = "L0Logistic";
	}

	else if (Type == "L0SquaredHingeSwaps"){
		PG.P.ModelType = "L012SquaredHingeSwaps";
		PG.Type = "L0SquaredHinge";
	}

	else if (Type == "L0L1" || Type == "L0L2" ){
		PG.P.ModelType = "L012";
		PG.Type = Type;
	}

	else if (Type == "L0L1LogisticSwaps" || Type == "L0L2LogisticSwaps" ){
		PG.P.ModelType = "L012LogisticSwaps";
		if(Type == "L0L1LogisticSwaps"){PG.Type = "L0L1Logistic";}
		else {PG.Type = "L0L2Logistic";}
	}

	else if (Type == "L0L1SquaredHingeSwaps" || Type == "L0L2SquaredHingeSwaps" ){
		PG.P.ModelType = "L012SquaredHingeSwaps";
		if(Type == "L0L1SquaredHingeSwaps"){PG.Type = "L0L1SquaredHinge";}
		else {PG.Type = "L0L2SquaredHinge";}
	}

	else if (Type == "L0L1Logistic" || Type == "L0L2Logistic" ){
		PG.P.ModelType = "L012Logistic";
		PG.Type = Type;
	}

	else if (Type == "L0L1SquaredHinge" || Type == "L0L2SquaredHinge" ){
		PG.P.ModelType = "L012SquaredHinge";
		PG.Type = Type;
	}

	else if (Type == "L0Swaps"){
		PG.P.ModelType = "L012Swaps";
		PG.Type = "L0";
	}

	else if (Type == "L0KSwaps"){
		PG.P.ModelType = "L012KSwaps";
		PG.Type = "L0";
	}

	else if (Type == "L0L1Swaps" || Type == "L0L2Swaps" ){
		PG.P.ModelType = "L012Swaps";
		if(Type == "L0L1Swaps"){PG.Type = "L0L1";}
		else {PG.Type = "L0L2";}
	}

	else if (Type == "L0L1KSwaps" || Type == "L0L2KSwaps" ){
		PG.P.ModelType = "L012KSwaps";
		if(Type == "L0L1KSwaps"){PG.Type = "L0L1";}
		else {PG.Type = "L0L2";}
	}

	else if(Type=="L1"){
		PG.P.ModelType = "L1";
		PG.Type = "L1";
	}
	else if(Type=="L1Relaxed"){
		PG.P.ModelType = "L1Relaxed";
		PG.Type = "L1Relaxed";
	}


	if (Type == "L0" || Type == "L0Swaps" || Type == "L1" || Type == "L0KSwaps" || Type == "IHT" || Type == "L0Logistic" || Type == "L0SquaredHinge" || Type =="L0LogisticSwaps"|| Type == "L0SquaredHingeSwaps" ){
		G = Grid1D(Xscaled, yscaled, PG).Fit();
	}
	else{
		G = Grid2D(Xscaled, yscaled, PG).Fit();
	}

    for (auto &g: G){
        Lambda0.push_back(g->ModelParams[0]);

        if (L0L1Models.find(Type) != L0L1Models.end() || Type=="L1Relaxed")
        	Lambda12.push_back(g->ModelParams[1]);
        else if (L0L2Models.find(Type) != L0L2Models.end())
        	Lambda12.push_back(g->ModelParams[2]);

        NnzCount.push_back(g->B.n_nonzero);

		arma::sp_mat B_unscaled;
		double intercept;

		if (classification){
			std::tie(B_unscaled, intercept) = DeNormalize(g->B, BetaMultiplier, meanX, meany);
					Solutions.push_back(B_unscaled);
					Intercepts.push_back(g->intercept + intercept); // + needed
		}

		else{
			std::tie(B_unscaled, intercept) = DeNormalize(g->B, BetaMultiplier, meanX, meany);
					Solutions.push_back(B_unscaled);
					Intercepts.push_back(intercept);
		}

  }




}

/*
// Bypass for R interface
int main(){
	std::cout<<"Enter Method"<<std::endl;
	std::string method;
	std::cin>>method;
	arma::mat X;
	X.load("X_training.csv");

	arma::vec y;
	y.load("y_training.csv");

	GridParams PG;
	PG.Lambda2Max = 1;
	PG.Lambda2Min = 1;
	PG.Type = method; //Classification
	PG.P.Tol = 1e-4;
	PG.P.ActiveSetNum = 2;
	PG.G_nrows = 1;
	PG.G_ncols = 86;

	auto start = std::chrono::steady_clock::now();

	auto g = Grid(X,y, PG);
	g.Fit();

	auto end = std::chrono::steady_clock::now();

	auto diff = end-start;

	std::cout << std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;

	//for(auto &sol:g.Solutions){sol.print();}


	return 0;

}
*/
