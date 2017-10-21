#include "RcppArmadillo.h"
#include "Grid.h"
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
Rcpp::List L0LearnFit(const arma::mat& X, const arma::vec& y, const std::string Type, const unsigned int NnzStopNum, const unsigned int G_ncols, const unsigned int G_nrows,
				const double Lambda2Max, const double Lambda2Min, const bool PartialSort, 
				const unsigned int MaxIters, const double Tol, const bool ActiveSet, const unsigned int ActiveSetNum, const unsigned int MaxNumSwaps){
    
    GridParams PG;
    PG.Type = Type;
    PG.NnzStopNum = NnzStopNum;
    PG.G_ncols = G_ncols;
    PG.G_nrows = G_nrows;
    PG.Lambda2Max = Lambda2Max;
    PG.Lambda2Min = Lambda2Min;
    PG.LambdaMinFactor = Lambda2Min; //
    PG.PartialSort = PartialSort;
    Params P;
    P.MaxIters = MaxIters;
    P.Tol = Tol;
    P.ActiveSet = ActiveSet;
    P.ActiveSetNum = ActiveSetNum;
    P.MaxNumSwaps = MaxNumSwaps;
    PG.P = P;


    Grid G(X,y,PG);
    G.Fit();

    std::string FirstParameter = "Lambda";
    std::string SecondParameter = "-1";
    if (G.L0L1Models.find(PG.Type) != G.L0L1Models.end())
    	SecondParameter = "Gamma";
    else if (G.L0L2Models.find(PG.Type) != G.L0L2Models.end())
    	SecondParameter = "Gamma";
    else if (PG.Type=="L1Relaxed")
    {
    	FirstParameter = "Lambda";
    	SecondParameter = "Gamma";
    }
    else if (PG.Type=="L1")
    	FirstParameter = "Lambda";

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