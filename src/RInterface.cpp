#include "RcppArmadillo.h"
#include "Grid.h"
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
Rcpp::List L0LearnFit(const arma::mat& X, const arma::vec& y, const std::string Loss, const std::string Penalty,
	 										const std::string Algorithm, const unsigned int NnzStopNum, const unsigned int G_ncols, const unsigned int G_nrows,
											const double Lambda2Max, const double Lambda2Min, const bool PartialSort,
											const unsigned int MaxIters, const double Tol, const bool ActiveSet,
											const unsigned int ActiveSetNum, const unsigned int MaxNumSwaps, const double ScaleDownFactor, unsigned int ScreenSize){

    GridParams PG;
    PG.NnzStopNum = NnzStopNum;
    PG.G_ncols = G_ncols;
    PG.G_nrows = G_nrows;
    PG.Lambda2Max = Lambda2Max;
    PG.Lambda2Min = Lambda2Min;
    PG.LambdaMinFactor = Lambda2Min; //
    PG.PartialSort = PartialSort;
		PG.ScaleDownFactor = ScaleDownFactor;
    Params P;
    P.MaxIters = MaxIters;
    P.Tol = Tol;
    P.ActiveSet = ActiveSet;
    P.ActiveSetNum = ActiveSetNum;
    P.MaxNumSwaps = MaxNumSwaps;
		P.ScreenSize = ScreenSize;
    PG.P = P;

		if (Loss == "SquaredError"){PG.P.Specs.SquaredError = true;}
		else if (Loss == "Logistic"){PG.P.Specs.Logistic = true; PG.P.Specs.Classification = true;}
		else if (Loss == "SquaredHinge") {PG.P.Specs.SquaredHinge = true; PG.P.Specs.Classification = true;}

		if (Algorithm == "CD"){PG.P.Specs.CD = true;}
		else if (Algorithm == "CDPSI"){PG.P.Specs.PSI = true;}

		if (Penalty == "L0"){ PG.P.Specs.L0 = true;}
		else if (Penalty == "L0L2"){ PG.P.Specs.L0L2 = true;}
		else if (Penalty == "L0L1"){ PG.P.Specs.L0L1 = true;}
		//case "L1": PG.P.Specs.L1 = true;
		//case "L1Relaxed": PG.P.Specs.L1Relaxed = true;

    Grid G(X,y,PG);
    G.Fit();

    std::string FirstParameter = "Lambda";
    std::string SecondParameter = "-1";
    if (PG.P.Specs.L0L1)
    	SecondParameter = "Gamma";
    else if PG.P.Specs.L0L2)
    	SecondParameter = "Gamma";
		/*
    else if (PG.P.Specs.L1Relaxed)
    {
    	FirstParameter = "Lambda";
    	SecondParameter = "Gamma";
    }
    else if (PG.P.Specs.L1)
    	FirstParameter = "Lambda";
		*/

    std::vector<std::vector<unsigned int>> indices(G.Solutions.size());
    std::vector<std::vector<double>> values(G.Solutions.size());

    for(unsigned int i=0; i<G.Solutions.size(); ++i){
    	arma::sp_mat::const_iterator j;
		for(j = G.Solutions[i].begin(); j != G.Solutions[i].end(); ++j){
			indices[i].push_back(j.row() + 1); // +1 to account for R's 1-based indexing
			values[i].push_back(*j);
		}

    }

    if (PG.Type != "-1"){
	    return Rcpp::List::create(Rcpp::Named(FirstParameter)=G.Lambda0,
	    						  Rcpp::Named(SecondParameter)=G.Lambda12,
	    						  Rcpp::Named("SuppSize")=G.NnzCount,
	    						  Rcpp::Named("BetaIndices")=indices,
	    						  Rcpp::Named("BetaValues")=values,
	    						  Rcpp::Named("Intercept")=G.Intercepts);
    }

    else{
	    return Rcpp::List::create(Rcpp::Named(FirstParameter)=G.Lambda0,
	    						  Rcpp::Named("SuppSize")=G.NnzCount,
	    						  Rcpp::Named("BetaIndices")=indices,
	    						  Rcpp::Named("BetaValues")=values,
	    						  Rcpp::Named("Intercept")=G.Intercepts);
    }
}
