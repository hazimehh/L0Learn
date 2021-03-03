#include "Interface.h"
#include "GridParams.h"
#include "RcppArmadillo.h"
#include "Grid.h"
#include "utils.h"
// [[Rcpp::depends(RcppArmadillo)]]

template <typename T>
GridParams<T> makeGridParams(const std::string Loss, const std::string Penalty,
                             const std::string Algorithm, const std::size_t NnzStopNum, const std::size_t G_ncols,
                             const std::size_t G_nrows, const double Lambda2Max, const double Lambda2Min,
                             const bool PartialSort, const std::size_t MaxIters, const double rtol, 
                             const double atol, const bool ActiveSet, const std::size_t ActiveSetNum, 
                             const std::size_t MaxNumSwaps, const double ScaleDownFactor,
                             const std::size_t ScreenSize, const bool LambdaU, 
                             const std::vector< std::vector<double> > Lambdas,
                             const std::size_t ExcludeFirstK, const bool Intercept, 
                             const bool withBounds, const arma::vec &Lows, const arma::vec &Highs){ 
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
  PG.P = P;
  PG.P.MaxIters = MaxIters;
  PG.P.rtol = rtol;
  PG.P.atol = atol;
  PG.P.ActiveSet = ActiveSet;
  PG.P.ActiveSetNum = ActiveSetNum;
  PG.P.MaxNumSwaps = MaxNumSwaps;
  PG.P.ScreenSize = ScreenSize;
  PG.P.NoSelectK = ExcludeFirstK;
  PG.P.intercept = Intercept;
  PG.P.withBounds = withBounds;
  PG.P.Lows = Lows;
  PG.P.Highs = Highs;
  
  if (Loss == "SquaredError") {
    PG.P.Specs.SquaredError = true;
  } else if (Loss == "Logistic") {
    PG.P.Specs.Logistic = true;
    PG.P.Specs.Classification = true;
  } else if (Loss == "SquaredHinge") {
    PG.P.Specs.SquaredHinge = true;
    PG.P.Specs.Classification = true;
  }
  
  if (Algorithm == "CD") {
    PG.P.Specs.CD = true;
  } else if (Algorithm == "CDPSI") {
    PG.P.Specs.PSI = true;
  }
  
  if (Penalty == "L0") {
    PG.P.Specs.L0 = true;
  } else if (Penalty == "L0L2") {
    PG.P.Specs.L0L2 = true;
  } else if (Penalty == "L0L1") { 
    PG.P.Specs.L0L1 = true;
  }
  return PG;
}


template <typename T>
Rcpp::List _L0LearnFit(const T& X, const arma::vec& y, const std::string Loss, const std::string Penalty,
                       const std::string Algorithm, const std::size_t NnzStopNum, const std::size_t G_ncols,
                       const std::size_t G_nrows, const double Lambda2Max, const double Lambda2Min,
                       const bool PartialSort, const std::size_t MaxIters, const double rtol, const double atol,
                       const bool ActiveSet, const std::size_t ActiveSetNum, const std::size_t MaxNumSwaps, 
                       const double ScaleDownFactor, const std::size_t ScreenSize, const bool LambdaU, 
                       const std::vector< std::vector<double> > Lambdas, const std::size_t ExcludeFirstK, 
                       const bool Intercept,  const bool withBounds, const arma::vec &Lows, 
                       const arma::vec &Highs){
  
  GridParams<T> PG = makeGridParams<T>(Loss, Penalty, Algorithm, NnzStopNum, G_ncols, G_nrows, 
                      Lambda2Max, Lambda2Min, PartialSort, MaxIters, rtol, atol, ActiveSet,
                      ActiveSetNum, MaxNumSwaps, ScaleDownFactor, ScreenSize,
                      LambdaU, Lambdas, ExcludeFirstK, Intercept, withBounds, Lows, Highs);

  Grid<T> G(X, y, PG);
  G.Fit();
  
  std::string FirstParameter = "lambda";
  std::string SecondParameter = "gamma";
  
  // Next Construct the list of Sparse Beta Matrices.
  
  auto p = X.n_cols;
  arma::field<arma::sp_mat> Bs(G.Lambda12.size());
  
  for (std::size_t i=0; i<G.Lambda12.size(); ++i) {
    // create the px(reg path size) sparse sparseMatrix
    arma::sp_mat B(p,G.Solutions[i].size());
    for (unsigned int j=0; j<G.Solutions[i].size(); ++j) {
      B.col(j) = G.Solutions[i][j];
    }
    
    // append the sparse matrix
    Bs[i] = B;
  }
  
  Rcpp::List l = Rcpp::List::create(Rcpp::Named(FirstParameter) = G.Lambda0,
                                    Rcpp::Named(SecondParameter) = G.Lambda12, // contains 0 in case of L0
                                    Rcpp::Named("SuppSize") = G.NnzCount,
                                    Rcpp::Named("beta") = Bs,
                                    Rcpp::Named("a0") = G.Intercepts,
                                    Rcpp::Named("Converged") = G.Converged);
  
  return l;
  
}

template <typename T>
Rcpp::List _L0LearnCV(const T& X, const arma::vec& y, const std::string Loss, 
                      const std::string Penalty, const std::string Algorithm, 
                      const unsigned int NnzStopNum, const unsigned int G_ncols,
                      const unsigned int G_nrows, const double Lambda2Max, 
                      const double Lambda2Min, const bool PartialSort,
                      const unsigned int MaxIters, const double rtol, 
                      const double atol, const bool ActiveSet, 
                      const unsigned int ActiveSetNum,
                      const unsigned int MaxNumSwaps, const double ScaleDownFactor, 
                      const unsigned int ScreenSize, const bool LambdaU,
                      const std::vector< std::vector<double> > Lambdas, 
                      const unsigned int nfolds, const double seed,
                      const unsigned int ExcludeFirstK, const bool Intercept, 
                      const bool withBounds, const arma::vec &Lows, 
                      const arma::vec &Highs){
  
  GridParams<T> PG = makeGridParams<T>(Loss, Penalty, Algorithm, NnzStopNum, G_ncols, G_nrows, 
                                       Lambda2Max, Lambda2Min, PartialSort, MaxIters, rtol, atol,
                                       ActiveSet,ActiveSetNum, MaxNumSwaps, ScaleDownFactor, 
                                       ScreenSize,LambdaU, Lambdas, ExcludeFirstK, Intercept, 
                                       withBounds, Lows, Highs);
  
  Grid<T> G(X, y, PG);
  G.Fit();
  
  std::string FirstParameter = "lambda";
  std::string SecondParameter = "gamma";
  
  // Next Construct the list of Sparse Beta Matrices.
  
  auto p = X.n_cols;
  auto n = X.n_rows;
  arma::field<arma::sp_mat> Bs(G.Lambda12.size());
  
  for (std::size_t i=0; i<G.Lambda12.size(); ++i) {
    // create the px(reg path size) sparse sparseMatrix
    arma::sp_mat B(p, G.Solutions[i].size());
    for (std::size_t j=0; j<G.Solutions[i].size(); ++j)
      B.col(j) = G.Solutions[i][j];
    
    // append the sparse matrix
    Bs[i] = B;
  }
  
  
  // CV starts here
  
  // arma::arma_rng::set_seed(seed);
  
  //Solutions = std::vector< std::vector<arma::sp_mat> >(G.size());
  //Intercepts = std::vector< std::vector<double> >(G.size());
  
  std::size_t Ngamma = G.Lambda12.size();
  
  //std::vector< arma::mat > CVError (G.Solutions.size());
  arma::field< arma::mat > CVError (G.Solutions.size());
  
  for (std::size_t i=0; i<G.Solutions.size(); ++i)
    CVError[i] = arma::mat(G.Lambda0[i].size(),nfolds, arma::fill::zeros);
  
  arma::uvec a = arma::linspace<arma::uvec>(0, X.n_rows-1, X.n_rows);
  
  arma::uvec indices = arma::shuffle(a);
  
  int samplesperfold = std::ceil(n/double(nfolds));
  int samplesinlastfold = samplesperfold - (samplesperfold*nfolds - n);
  
  std::vector<std::size_t> fullindices(X.n_rows);
  std::iota(fullindices.begin(), fullindices.end(), 0);
  
  
  for (std::size_t j=0; j<nfolds;++j) {
    std::vector<std::size_t> validationindices;
    if (j < nfolds-1)
      validationindices.resize(samplesperfold);
    else
      validationindices.resize(samplesinlastfold);
    
    std::iota(validationindices.begin(), validationindices.end(), samplesperfold*j);
    
    std::vector<std::size_t> trainingindices;
    
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
    
    
    T Xtraining = matrix_rows_get(X, trainingindicesarma);
    
    arma::mat ytraining = y.elem(trainingindicesarma);
    
    T Xvalidation = matrix_rows_get(X, validationindicesarma);
    
    arma::mat yvalidation = y.elem(validationindicesarma);
    
    PG.LambdaU = true;
    PG.XtrAvailable = false; // reset XtrAvailable since its changed upon every call
    PG.LambdasGrid = G.Lambda0;
    PG.NnzStopNum = p+1; // remove any constraints on the supp size when fitting over the cv folds // +1 is imp to avoid =p edge case
    if (PG.P.Specs.L0 == true){
      PG.Lambdas = PG.LambdasGrid[0];
    }
    Grid<T> Gtraining(Xtraining, ytraining, PG);
    Gtraining.Fit();
    
    for (std::size_t i=0; i<Ngamma; ++i) { 
      // i indexes the gamma parameter
      for (std::size_t k=0; k<Gtraining.Lambda0[i].size(); ++k){ 
        // k indexes the solutions for a specific gamma
        
        if (PG.P.Specs.SquaredError) {
          arma::vec r = yvalidation - Xvalidation*Gtraining.Solutions[i][k] + Gtraining.Intercepts[i][k];
          CVError[i](k,j) = arma::dot(r,r) / yvalidation.n_rows;
        } else if (PG.P.Specs.Logistic) {
          arma::sp_mat B = Gtraining.Solutions[i][k];
          double b0 = Gtraining.Intercepts[i][k];
          arma::vec ExpyXB = arma::exp(yvalidation % (Xvalidation * B + b0));
          CVError[i](k,j) = arma::sum(arma::log(1 + 1 / ExpyXB)) / yvalidation.n_rows;
          //std::cout<<"i, j, k"<<i<<" "<<j<<" "<<k<<" CVError[i](k,j): "<<CVError[i](k,j)<<std::endl;
          //CVError[i].print();
          
        } else if (PG.P.Specs.SquaredHinge) {
          arma::sp_mat B = Gtraining.Solutions[i][k];
          double b0 = Gtraining.Intercepts[i][k];
          arma::vec onemyxb = 1 - yvalidation % (Xvalidation * B + b0);
          arma::uvec indices = arma::find(onemyxb > 0);
          CVError[i](k,j) = arma::sum(onemyxb.elem(indices) % onemyxb.elem(indices)) / yvalidation.n_rows;
        }
      }
    }
  }
  
  arma::field<arma::vec> CVMeans(Ngamma);
  arma::field<arma::vec> CVSDs(Ngamma);

  for (std::size_t i=0; i<Ngamma; ++i) {
    CVMeans[i] = arma::mean(CVError[i],1);
    CVSDs[i] = arma::stddev(CVError[i],0,1);
  }
  
  return Rcpp::List::create(Rcpp::Named(FirstParameter) = G.Lambda0,
                            Rcpp::Named(SecondParameter) = G.Lambda12, // contains 0 in case of L0
                            Rcpp::Named("SuppSize") = G.NnzCount,
                            Rcpp::Named("beta") = Bs,
                            Rcpp::Named("a0") = G.Intercepts,
                            Rcpp::Named("Converged") = G.Converged,
                            Rcpp::Named("CVMeans") = CVMeans,
                            Rcpp::Named("CVSDs") = CVSDs );
  
}


// [[Rcpp::export]]
Rcpp::List L0LearnFit_sparse(const arma::sp_mat& X, const arma::vec& y, const std::string Loss, const std::string Penalty,
                      const std::string Algorithm, const std::size_t NnzStopNum, const std::size_t G_ncols,
                      const std::size_t G_nrows, const double Lambda2Max, const double Lambda2Min,
                      const bool PartialSort, const std::size_t MaxIters, const double rtol, 
                      const double atol, const bool ActiveSet, const std::size_t ActiveSetNum, 
                      const std::size_t MaxNumSwaps, const double ScaleDownFactor, 
                      const std::size_t ScreenSize, const bool LambdaU,
                      const std::vector< std::vector<double> > Lambdas,
                      const std::size_t ExcludeFirstK, const bool Intercept,
                      const bool withBounds, const arma::vec &Lows, const arma::vec &Highs) {
  
  return _L0LearnFit(X, y, Loss, Penalty, Algorithm, NnzStopNum, G_ncols, G_nrows, Lambda2Max, Lambda2Min,
                     PartialSort, MaxIters, rtol, atol, ActiveSet, ActiveSetNum, MaxNumSwaps, ScaleDownFactor, ScreenSize, LambdaU,
                     Lambdas, ExcludeFirstK, Intercept, withBounds, Lows, Highs);
}


// [[Rcpp::export]]
Rcpp::List L0LearnFit_dense(const arma::mat& X, const arma::vec& y, const std::string Loss, const std::string Penalty,
                      const std::string Algorithm, const std::size_t NnzStopNum, const std::size_t G_ncols,
                      const std::size_t G_nrows, const double Lambda2Max, const double Lambda2Min,
                      const bool PartialSort, const std::size_t MaxIters, const double rtol, 
                      const double atol, const bool ActiveSet, const std::size_t ActiveSetNum, 
                      const std::size_t MaxNumSwaps, const double ScaleDownFactor, 
                      const std::size_t ScreenSize, const bool LambdaU,
                      const std::vector< std::vector<double> > Lambdas,
                      const std::size_t ExcludeFirstK, const bool Intercept,
                      const bool withBounds, const arma::vec &Lows, const arma::vec &Highs) {
  
      return _L0LearnFit(X, y, Loss, Penalty, Algorithm, NnzStopNum, G_ncols, G_nrows, Lambda2Max, Lambda2Min,
                       PartialSort, MaxIters, rtol, atol, ActiveSet, ActiveSetNum, MaxNumSwaps, ScaleDownFactor, ScreenSize, LambdaU,
                       Lambdas, ExcludeFirstK, Intercept, withBounds, Lows, Highs);
}
  

// [[Rcpp::export]]
Rcpp::List L0LearnCV_sparse(const arma::sp_mat& X, const arma::vec& y, const std::string Loss, const std::string Penalty,
                     const std::string Algorithm, const std::size_t NnzStopNum, const std::size_t G_ncols,
                     const std::size_t G_nrows, const double Lambda2Max, const double Lambda2Min,
                     const bool PartialSort, const std::size_t MaxIters, const double rtol, const double atol, 
                     const bool ActiveSet, const std::size_t ActiveSetNum, const std::size_t MaxNumSwaps,
                     const double ScaleDownFactor, const std::size_t ScreenSize, const bool LambdaU, 
                     const std::vector< std::vector<double> > Lambdas, const std::size_t nfolds, 
                     const double seed, const std::size_t ExcludeFirstK, const bool Intercept, 
                     const bool withBounds, const arma::vec &Lows, const arma::vec &Highs){
  
    return _L0LearnCV(X, y, Loss, Penalty,
                      Algorithm, NnzStopNum, G_ncols, G_nrows,
                      Lambda2Max, Lambda2Min, PartialSort,
                      MaxIters, rtol,atol, ActiveSet,
                      ActiveSetNum, MaxNumSwaps,
                      ScaleDownFactor, ScreenSize, LambdaU, Lambdas,
                      nfolds, seed, ExcludeFirstK, Intercept, withBounds, Lows, Highs);
}

// [[Rcpp::export]]
Rcpp::List L0LearnCV_dense(const arma::mat& X, const arma::vec& y, const std::string Loss, const std::string Penalty,
                    const std::string Algorithm, const std::size_t NnzStopNum, const std::size_t G_ncols,
                    const std::size_t G_nrows, const double Lambda2Max, const double Lambda2Min,
                    const bool PartialSort, const std::size_t MaxIters, const double rtol, const double atol, 
                    const bool ActiveSet, const std::size_t ActiveSetNum, const std::size_t MaxNumSwaps,
                    const double ScaleDownFactor, const std::size_t ScreenSize, const bool LambdaU, 
                    const std::vector< std::vector<double> > Lambdas, const std::size_t nfolds, 
                    const double seed, const std::size_t ExcludeFirstK, const bool Intercept, 
                    const bool withBounds, const arma::vec &Lows, const arma::vec &Highs){
  
  return _L0LearnCV(X, y, Loss, Penalty,
                    Algorithm, NnzStopNum, G_ncols, G_nrows,
                    Lambda2Max, Lambda2Min, PartialSort,
                    MaxIters, rtol,atol, ActiveSet,
                    ActiveSetNum, MaxNumSwaps,
                    ScaleDownFactor, ScreenSize, LambdaU, Lambdas,
                    nfolds, seed, ExcludeFirstK, Intercept, withBounds, Lows, Highs);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cor_matrix(const int p, const double base_cor) {
  Rcpp::NumericMatrix cor(p, p);
  for (int i = 0; i < p; i++){
    for (int j = 0; j < p; j++){
      cor(i, j) = std::pow(base_cor, std::abs(i - j));
    }
  }
  return cor;
}

