#include <Normalize.h>

// scale, meanX, meany
std::tuple<arma::vec, arma::vec, double>  Normalize(const arma::mat& X, const arma::vec& y, arma::mat & X_normalized, arma::vec & y_normalized, bool Normalizey = false)
{
    unsigned int n = X.n_rows;
    unsigned int p = X.n_cols;

    arma::rowvec meanX = arma::mean(X, 0);

    X_normalized = X.each_row() - meanX;
    arma::rowvec scaleX = std::sqrt(n) * arma::stddev(X_normalized, 1, 0); // contains the l2norm of every col
    X_normalized.each_row() /= scaleX;

    arma::vec BetaMultiplier;
    double meany = 0;
    if (Normalizey)
    {
        meany = arma::mean(y);
        y_normalized = y - meany;
        double scaley = arma::as_scalar(std::sqrt(n) * arma::stddev(y_normalized, 1, 0)); // contains the l2norm of y
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

/*
int main(){

	// ToDo: Add as unit test.

	arma::mat X = arma::randu<arma::mat>(11,10);
	arma::vec B = arma::randu<arma::vec>(10);
	arma::vec y = X*B + 10*arma::ones<arma::vec>(11);
	B.print();

	arma::mat Xscaled;
	arma::vec yscaled;

	arma::vec BetaMultiplier;
	arma::vec meanX;
	double meany;

	std::tie(BetaMultiplier, meanX, meany) = Normalize(X,y, Xscaled, yscaled);

	arma::sp_mat B_unscaled;
	double intercept;

	arma::vec Bsolved = arma::solve(Xscaled,yscaled);
	arma::sp_mat Bsparse = arma::conv_to< arma::sp_mat >::from(Bsolved);
	std::tie(B_unscaled, intercept) = DeNormalize(Bsparse, BetaMultiplier, meanX, meany);

	B_unscaled.print();
	std::cout<<"Intercept: "<<intercept<<std::endl;

	return 0;
}
*/
