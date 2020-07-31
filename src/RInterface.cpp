#include "Rface.h" // somehow "RInterace.h" does not work"
// [[Rcpp::depends(RcppArmadillo)]]

template <typename T>
Rcpp::List _L0LearnFit(const T& X, const arma::vec& y, const std::string Loss, const std::string Penalty,
                      const std::string Algorithm, const unsigned int NnzStopNum, const unsigned int G_ncols, const unsigned int G_nrows,
                      const double Lambda2Max, const double Lambda2Min, const bool PartialSort,
                      const unsigned int MaxIters, const double Tol, const bool ActiveSet,
                      const unsigned int ActiveSetNum, const unsigned int MaxNumSwaps,
                      const double ScaleDownFactor, unsigned int ScreenSize, const bool LambdaU, const std::vector< std::vector<double> > Lambdas, const unsigned int ExcludeFirstK, const bool Intercept)
{
    auto p = X.n_cols;
    GridParams<T> PG;
    PG.NnzStopNum = NnzStopNum;
    PG.G_ncols = G_ncols;
    PG.G_nrows = G_nrows;
    PG.Lambda2Max = Lambda2Max;
    PG.Lambda2Min = Lambda2Min;
    PG.LambdaMinFactor = Lambda2Min; //
    PG.PartialSort = PartialSort;
    PG.ScaleDownFactor = ScaleDownFactor;
    PG.LambdaU = LambdaU;
    PG.LambdasGrid = Lambdas;
    PG.Lambdas = Lambdas[0]; // to handle the case of L0 (i.e., Grid1D)
    PG.intercept = Intercept;
    Params<T> P;
    P.MaxIters = MaxIters;
    P.Tol = Tol;
    P.ActiveSet = ActiveSet;
    P.ActiveSetNum = ActiveSetNum;
    P.MaxNumSwaps = MaxNumSwaps;
    P.ScreenSize = ScreenSize;
    P.NoSelectK = ExcludeFirstK;
    P.intercept = Intercept;
    PG.P = P;

    if (Loss == "SquaredError") {PG.P.Specs.SquaredError = true;}
    else if (Loss == "Logistic") {PG.P.Specs.Logistic = true; PG.P.Specs.Classification = true;}
    else if (Loss == "SquaredHinge") {PG.P.Specs.SquaredHinge = true; PG.P.Specs.Classification = true;}

    if (Algorithm == "CD") {PG.P.Specs.CD = true;}
    else if (Algorithm == "CDPSI") {PG.P.Specs.PSI = true;}

    if (Penalty == "L0") { PG.P.Specs.L0 = true;}
    else if (Penalty == "L0L2") { PG.P.Specs.L0L2 = true;}
    else if (Penalty == "L0L1") { PG.P.Specs.L0L1 = true;}
    //case "L1": PG.P.Specs.L1 = true;
    //case "L1Relaxed": PG.P.Specs.L1Relaxed = true;

    Grid<T> G(X, y, PG);
    G.Fit();

    std::string FirstParameter = "lambda";
    std::string SecondParameter = "gamma";
    /*
    else if (PG.P.Specs.L1Relaxed)
    {
    FirstParameter = "Lambda";
    SecondParameter = "Gamma";
    }
    else if (PG.P.Specs.L1)
    FirstParameter = "Lambda";
    */


    // Next Construct the list of Sparse Beta Matrices.
    //std::vector<arma::sp_mat> Bs;

    arma::field<arma::sp_mat> Bs(G.Lambda12.size());

    for (unsigned int i=0; i<G.Lambda12.size(); ++i)
    {
        // create the px(reg path size) sparse sparseMatrix
        arma::sp_mat B(p,G.Solutions[i].size());
        for (unsigned int j=0; j<G.Solutions[i].size(); ++j)
        {
            B.col(j) = G.Solutions[i][j];
        }

        // append the sparse matrix
        Bs[i] = B;
    }



    /*
    if (!PG.P.Specs.L0)
    {
        return Rcpp::List::create(Rcpp::Named(FirstParameter) = G.Lambda0,
                                  Rcpp::Named(SecondParameter) = G.Lambda12,
                                  Rcpp::Named("SuppSize") = G.NnzCount,
                                  Rcpp::Named("beta") = Bs,
                                  Rcpp::Named("a0") = G.Intercepts,
                                  Rcpp::Named("Converged") = G.Converged);
    }

    else
    {
        return Rcpp::List::create(Rcpp::Named(FirstParameter) = G.Lambda0[0],
                                  Rcpp::Named(SecondParameter) = 0,
                                  Rcpp::Named("SuppSize") = G.NnzCount[0],
                                  Rcpp::Named("beta") = Bs[0],
                                  Rcpp::Named("a0") = G.Intercepts[0],
                                  Rcpp::Named("Converged") = G.Converged[0]);

    }
    */
    Rcpp::List l = Rcpp::List::create(Rcpp::Named(FirstParameter) = G.Lambda0,
                              Rcpp::Named(SecondParameter) = G.Lambda12, // contains 0 in case of L0
                              Rcpp::Named("SuppSize") = G.NnzCount,
                              Rcpp::Named("beta") = Bs,
                              Rcpp::Named("a0") = G.Intercepts,
                              Rcpp::Named("Converged") = G.Converged);

    return l;

}

// [[Rcpp::export]]
Rcpp::List L0LearnFit(const SEXP& X, const arma::vec& y, const std::string Loss, const std::string Penalty,
                      const std::string Algorithm, const unsigned int NnzStopNum, const unsigned int G_ncols, const unsigned int G_nrows,
                      const double Lambda2Max, const double Lambda2Min, const bool PartialSort,
                      const unsigned int MaxIters, const double Tol, const bool ActiveSet,
                      const unsigned int ActiveSetNum, const unsigned int MaxNumSwaps,
                      const double ScaleDownFactor, unsigned int ScreenSize, const bool LambdaU, const std::vector< std::vector<double> > Lambdas, const unsigned int ExcludeFirstK, const bool Intercept){
if (Rf_isS4(X) && Rf_inherits(X, "dgCMatrix")){
    arma::sp_mat m = Rcpp::as<arma::sp_mat>(X);
    return _L0LearnFit(m, y, Loss, Penalty,
                       Algorithm, NnzStopNum, G_ncols, G_nrows,
                       Lambda2Max, Lambda2Min, PartialSort,
                       MaxIters, Tol, ActiveSet,
                       ActiveSetNum, MaxNumSwaps,
                       ScaleDownFactor, ScreenSize, LambdaU, 
                       Lambdas, ExcludeFirstK, Intercept);
  }
  else if (Rf_isS4(X) && Rf_inherits(X, "dgeMatrix"))
  {
      Rcpp::stop("Only supports Dense and dgCMatrix Matricies");
  }
  else
  {
    arma::mat m = Rcpp::as<arma::mat>(X);
    return _L0LearnFit(m, y, Loss, Penalty,
                       Algorithm, NnzStopNum, G_ncols, G_nrows,
                       Lambda2Max, Lambda2Min, PartialSort,
                       MaxIters, Tol, ActiveSet,
                       ActiveSetNum, MaxNumSwaps,
                       ScaleDownFactor, ScreenSize, LambdaU, 
                       Lambdas, ExcludeFirstK, Intercept);
  }
}
