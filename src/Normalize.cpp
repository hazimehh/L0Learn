#include <Normalize.h>

arma::rowvec matrix_normalize(arma::mat& mat_norm){
    
    auto p = mat_norm.n_cols;
    arma::rowvec scaleX = arma::zeros<arma::rowvec>(p); // will contain the l2norm of every col
    
    for (auto col = 0; col < p; col++) {
        double l2norm = arma::norm(mat_norm.unsafe_col(col), 2);
        scaleX(col) = l2norm;
    }
    
    scaleX.replace(0, -1);
    mat_norm.each_row() /= scaleX;
    
    if (mat_norm.has_nan()){
        mat_norm.replace(arma::datum::nan, 0); // can handle numerical instabilities.   
    }
    
    return scaleX;
}

// scale, meanX, meany
std::tuple<arma::vec, arma::vec, double>  Normalize(const arma::mat& X, 
                                                    const arma::vec& y,
                                                    arma::mat & X_normalized,
                                                    arma::vec & y_normalized, 
                                                    bool Normalizey, 
                                                    bool intercept){
    // unsigned int n = X.n_rows;
    unsigned int p = X.n_cols;

    arma::rowvec meanX;
    
    if (intercept){
        meanX = arma::mean(X, 0);
    } else {
        meanX = arma::zeros<arma::rowvec>(p);
    }
    
    X_normalized = X.each_row() - meanX;
    
    arma::rowvec scaleX = matrix_normalize(X_normalized); // contains the l2norm of every col
  
    arma::vec BetaMultiplier;
    double meany = 0;
    if (Normalizey)
    {
        if (intercept){
            meany = arma::mean(y);
        }
        
        y_normalized = y - meany;

        double scaley = arma::norm(y_normalized, 2);
        
        if (scaley == 0){
            scaley =1;
        } // contains the l2norm of y
        
        y_normalized = y_normalized / scaley;
        BetaMultiplier = scaley / (scaleX.t()); // transpose scale to get a col vec
        // Multiplying the learned Beta by BetaMultiplier gives the optimal Beta on the original scale
    }
    else
    {
        y_normalized = y;
        BetaMultiplier = 1 / (scaleX.t()); // transpose scale to get a col vec
    }
    return std::make_tuple(BetaMultiplier, meanX.t(), meany);
}

std::tuple<arma::sp_mat, double> DeNormalize(arma::sp_mat & B_scaled, arma::vec & BetaMultiplier, arma::vec & meanX, double meany)
{
    arma::sp_mat B_unscaled = B_scaled % BetaMultiplier;
    double intercept = meany - arma::dot(B_unscaled, meanX);
    return std::make_tuple(B_unscaled, intercept);
}

std::tuple<arma::vec, arma::vec, double>  Normalizev1_2_0(const arma::mat& X,
                                                          const arma::vec& y, 
                                                          arma::mat & X_normalized,
                                                          arma::vec & y_normalized, 
                                                          bool Normalizey, 
                                                          bool intercept){
    unsigned int n = X.n_rows;
    unsigned int p = X.n_cols;
    
    arma::rowvec meanX;
    if (intercept){meanX = arma::mean(X, 0);}
    else{meanX = arma::zeros<arma::rowvec>(p);}
    X_normalized = X.each_row() - meanX;
    arma::rowvec scaleX = std::sqrt(n) * arma::stddev(X_normalized, 1, 0); // contains the l2norm of every col
    scaleX.replace(0, -1);
    X_normalized.each_row() /= scaleX;
    if (X_normalized.has_nan()){X_normalized.replace(arma::datum::nan, 0); } // can handle numerical instabilities.
    
    arma::vec BetaMultiplier;
    double meany = 0;
    if (Normalizey)
    {
        if (intercept){meany = arma::mean(y);}
        y_normalized = y - meany;
        
        auto stddev = arma::as_scalar(arma::stddev(y_normalized, 1, 0));
        double scaley = 1;
        // below stddev != 0 ensures we properly handle cases where y is constant
        if (stddev != 0){scaley = std::sqrt(n) * stddev;} // contains the l2norm of y
        y_normalized = y_normalized / scaley;
        BetaMultiplier = scaley / (scaleX.t()); // transpose scale to get a col vec
        // Multiplying the learned Beta by BetaMultiplier gives the optimal Beta on the original scale
    }
    else
    {
        y_normalized = y;
        BetaMultiplier = 1 / (scaleX.t()); // transpose scale to get a col vec
    }
    return std::make_tuple(BetaMultiplier, meanX.t(), meany);
}
