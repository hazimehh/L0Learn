#include "RcppArmadillo.h"
#include "Grid.h"
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
Rcpp::List L0LearnCV(const arma::mat& X, const arma::vec& y, const std::string Loss, const std::string Penalty,
                      const std::string Algorithm, const unsigned int NnzStopNum, const unsigned int G_ncols, const unsigned int G_nrows,
                      const double Lambda2Max, const double Lambda2Min, const bool PartialSort,
                      const unsigned int MaxIters, const double Tol, const bool ActiveSet,
                      const unsigned int ActiveSetNum, const unsigned int MaxNumSwaps,
                      const double ScaleDownFactor, unsigned int ScreenSize, const bool LambdaU, const std::vector< std::vector<double> > Lambdas,
                      const unsigned int nfolds, const double seed)
{
    auto p = X.n_cols;
    auto n = X.n_rows;
    GridParams PG;
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
    Params P;
    P.MaxIters = MaxIters;
    P.Tol = Tol;
    P.ActiveSet = ActiveSet;
    P.ActiveSetNum = ActiveSetNum;
    P.MaxNumSwaps = MaxNumSwaps;
    P.ScreenSize = ScreenSize;
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

    Grid G(X, y, PG);
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


    // CV starts here

    // arma::arma_rng::set_seed(seed);

    //Solutions = std::vector< std::vector<arma::sp_mat> >(G.size());
    //Intercepts = std::vector< std::vector<double> >(G.size());

    unsigned int Ngamma = G.Lambda12.size();

    std::vector< arma::mat > CVError (G.Solutions.size());
    for(unsigned int i=0; i<G.Solutions.size(); ++i)
    {
        CVError[i] = arma::mat(G.Lambda0[i].size(),nfolds, arma::fill::zeros);
    }


    arma::uvec a = arma::linspace<arma::uvec>(0, X.n_rows-1, X.n_rows);
  	arma::uvec indices = arma::shuffle(a);

    int samplesperfold = std::ceil(n/double(nfolds));
  	int samplesinlastfold = samplesperfold - (samplesperfold*nfolds - n);

  	std::vector<unsigned int> fullindices(X.n_rows);
  	std::iota(fullindices.begin(), fullindices.end(), 0);


  	for (unsigned int j=0; j<nfolds;++j)
  	{
    		std::vector<unsigned int> validationindices;
    		if (j < nfolds-1)
    				validationindices.resize(samplesperfold);
    		else
    				validationindices.resize(samplesinlastfold);

    		std::iota(validationindices.begin(), validationindices.end(), samplesperfold*j);

    		std::vector<unsigned int> trainingindices;

    		std::set_difference(fullindices.begin(), fullindices.end(), validationindices.begin(), validationindices.end(),
                            std::inserter(trainingindices, trainingindices.begin()));


        // validationindicesarma contains the randomly permuted validation indices as a uvec
        arma::uvec validationindicesarma;
        arma::uvec validationindicestemp = arma::conv_to< arma::uvec >::from(validationindices);
    		validationindicesarma = indices.elem(validationindicestemp);

        // trainingindicesarma is similar to validationindicesarma but for training
        arma::uvec trainingindicesarma;

        arma::uvec trainingindicestemp = arma::conv_to< arma::uvec >::from(trainingindices);


    		trainingindicesarma = indices.elem(trainingindicestemp);


        arma::mat Xtraining = X.rows(trainingindicesarma);

        arma::mat ytraining = y.elem(trainingindicesarma);

        arma::mat Xvalidation = X.rows(validationindicesarma);

        arma::mat yvalidation = y.elem(validationindicesarma);

        PG.LambdaU = true;
        PG.XtrAvailable = false; // reset XtrAvailable since its changed upon every call
        PG.LambdasGrid = G.Lambda0;
        Grid Gtraining(Xtraining, ytraining, PG);
        Gtraining.Fit();

        for (unsigned int i=0; i<Ngamma; ++i)
        { // i indexes the gamma parameter
            for (unsigned int k=0; k<Gtraining.Lambda0[i].size(); ++k)
            { // k indexes the solutions for a specific gamma

                if (PG.P.Specs.SquaredError)
                {
                    arma::vec r = yvalidation - Xvalidation*Gtraining.Solutions[i][k] + Gtraining.Intercepts[i][k];
                    CVError[i][k,j] = arma::dot(r,r);
                }

                else if (PG.P.Specs.Logistic)
                {
                    arma::sp_mat B = Gtraining.Solutions[i][k];
                    double b0 = Gtraining.Intercepts[i][k];
                    arma::vec ExpyXB = arma::exp(yvalidation % (Xvalidation * B + b0));
                    CVError[i][k,j] = arma::sum(arma::log(1 + 1 / ExpyXB));
                    std::cout<<"i, j, k"<<i<<" "<<j<<" "<<k<<" CVError[i][k,j]: "<<CVError[i][k,j]<<std::endl;
                    if (CVError[i][k,j] < 0)
                    {
                        std::cout<<"Detected Negative CV Errors!!! "<<CVError[i][k,j]<<std::endl;
                        ExpyXB.print();
                        B.print();
                    }
                }

            }
        }
  	}


    arma::field<arma::vec> CVMeans(Ngamma); // PROBLEM HERE!!!
    arma::field<arma::vec> CVSDs(Ngamma);

    for(unsigned int i=0; i<Ngamma; ++i)
    {

        CVError[i].print();

        arma::mean(CVError[i],1).print();

        //arma::stddev(CVError[i],0,1).print();

        CVMeans[i] = arma::mean(CVError[i],1);

        CVSDs[i] = arma::stddev(CVError[i],0,1);
    }

    // CV ends here.

    if (!PG.P.Specs.L0)
    {
        return Rcpp::List::create(Rcpp::Named(FirstParameter) = G.Lambda0,
                                  Rcpp::Named(SecondParameter) = G.Lambda12,
                                  Rcpp::Named("SuppSize") = G.NnzCount,
                                  Rcpp::Named("beta") = Bs,
                                  Rcpp::Named("a0") = G.Intercepts,
                                  Rcpp::Named("Converged") = G.Converged,
                                  Rcpp::Named("CVMeans") = CVMeans,
                                  Rcpp::Named("CVSDs") = CVSDs );
    }

    else
    {
        return Rcpp::List::create(Rcpp::Named(FirstParameter) = G.Lambda0[0],
                                  Rcpp::Named(SecondParameter) = 0,
                                  Rcpp::Named("SuppSize") = G.NnzCount[0],
                                  Rcpp::Named("beta") = Bs[0],
                                  Rcpp::Named("a0") = G.Intercepts[0],
                                  Rcpp::Named("Converged") = G.Converged[0],
                                  Rcpp::Named("CVMeans") = CVMeans,
                                  Rcpp::Named("CVSDs") = CVSDs );
    }
}
