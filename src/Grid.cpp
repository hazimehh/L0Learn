#include "Grid.h"
#include "Grid1D.h"
#include "Grid2D.h"
#include "Normalize.h"

Grid::Grid(const arma::mat& X, const arma::vec& y, const GridParams& PGi){
	PG = PGi;

	std::string Type = PG.Type;
	if (ClassificationModels.find(Type) != ClassificationModels.end()){
		classification = true;
	}

	std::tie(BetaMultiplier, meanX, meany) = Normalize(X,y, Xscaled, yscaled,!classification); // Don't normalize y

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

	else if (Type == "L0L1" || Type == "L0L2" ){
		PG.P.ModelType = "L012";
		PG.Type = Type;
	}

	else if (Type == "L0L1Logistic" || Type == "L0L2Logistic" ){
		PG.P.ModelType = "L012Logistic";
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


	if (Type == "L0" || Type == "L0Swaps" || Type == "L1" || Type == "L0KSwaps" || Type == "IHT" || Type == "L0Logistic"){
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
			std::tie(B_unscaled, intercept) = DeNormalize(g->B, BetaMultiplier, meanX, meany,false);
					Solutions.push_back(B_unscaled);
					Intercepts.push_back(g->intercept);
		}

		else{
			std::tie(B_unscaled, intercept) = DeNormalize(g->B, BetaMultiplier, meanX, meany, true);
					Solutions.push_back(B_unscaled);
					Intercepts.push_back(intercept);
		}

  }
}
