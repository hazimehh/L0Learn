#include <Normalize.h>

std::tuple<arma::sp_mat, double> DeNormalize(arma::sp_mat & B_scaled, 
                                             arma::vec & BetaMultiplier, 
                                             arma::vec & meanX, double meany)
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
