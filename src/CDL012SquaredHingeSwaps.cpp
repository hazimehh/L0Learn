#include "CDL012SquaredHingeSwaps.h"
#include "CDL012SquaredHinge.h"

#include "Normalize.h" // Remove later

CDL012SquaredHingeSwaps::CDL012SquaredHingeSwaps(const arma::mat& Xi, const arma::vec& yi, const Params& Pi) : CD(Xi, yi, Pi)
{
	MaxNumSwaps = Pi.MaxNumSwaps;
	P = Pi;
	twolambda2 = 2*ModelParams[2];
	qp2lamda2 = (LipschitzConst+twolambda2); // this is the univariate lipschitz const of the differentiable objective
	thr = std::sqrt((2*ModelParams[0])/qp2lamda2);
	stl0Lc = std::sqrt((2*ModelParams[0])*qp2lamda2);
	lambda1 = ModelParams[1];
	lambda1ol = lambda1 / qp2lamda2;
	Xtr = P.Xtr; Iter = P.Iter; result.ModelParams = P.ModelParams;

}

FitResult CDL012SquaredHingeSwaps::Fit() {
	auto result = CDL012SquaredHinge(*X, *y, P).Fit(); // result will be maintained till the end
	b0 = result.intercept; // Initialize from previous later....!
	B = result.B;
	//ExpyXB = result.ExpyXB; // Maintained throughout the algorithm
	arma::vec onemyxb = 1 - *y % (*X * B + b0);

	objective = result.Objective;
	double Fmin = objective;
	unsigned int maxindex;
	double Bmaxindex;

	P.Init = 'u';

	bool foundbetter;
	for (unsigned int t=0; t<MaxNumSwaps; ++t){
		std::cout<<"1Swaps Iteration: "<<t<<". "<<"Obj: "<<objective<<std::endl;
		//B.print();
		arma::sp_mat::const_iterator start = B.begin();
		arma::sp_mat::const_iterator end   = B.end();
		std::vector<unsigned int> NnzIndices;
		for(arma::sp_mat::const_iterator it = start; it != end; ++it)
		{
			NnzIndices.push_back(it.row()); // i is
		}
		// Can easily shuffle here...
		//std::shuffle(std::begin(Order), std::end(Order), engine);
		foundbetter = false;


		for (auto& j: NnzIndices){

			// Remove j
			//arma::vec ExpyXBnoj = ExpyXB % arma::exp( - B[j] *  *y % X->unsafe_col(j));
			arma::vec onemyxbnoj = onemyxb + B[j] * *y % X->unsafe_col(j);
			arma::uvec indices = arma::find(onemyxbnoj > 0);


			for(unsigned int i =0; i<p; ++i){
				if(B[i] == 0){

					double Biold = 0;
					double Binew;


					double partial_i = arma::sum(2 * onemyxbnoj.elem(indices) % (- y->elem(indices) % X->unsafe_col(i).elem(indices)));

					bool Converged = false;
					if (std::abs(partial_i) >= lambda1 + stl0Lc ){

						//std::cout<<"Adding: "<<i<< std::endl;
						arma::vec onemyxbnoji = onemyxbnoj;

						unsigned int l = 0;
						arma::sp_mat Btemp = B;
						Btemp[j] = 0;
						double ObjTemp = Objective(onemyxbnoj,Btemp);
						double Biolddescent = 0;
						while(!Converged)
						{
							//double x = Biold - partial_i/qp2lamda2;

							double x = Biold - partial_i/qp2lamda2;
							double z = std::abs(x) - lambda1ol;

							Binew = std::copysign(z, x); // no need to check if >= sqrt(2lambda_0/Lc)
							onemyxbnoji += (Biold - Binew) * *y % X->unsafe_col(i);
							//std::cout<<"Here222!!"<<std::endl;

							arma::uvec indicesi = arma::find(onemyxbnoji > 0);
							partial_i = arma::sum(2 * onemyxbnoj.elem(indicesi) % (- y->elem(indicesi) % X->unsafe_col(i).elem(indicesi)));

							if (std::abs((Binew - Biold)/Biold) < 0.0001){Converged = true;}


							Biold = Binew;
							l += 1;

						}

						// Can be made much faster (later)
						Btemp[i] = Binew;
						//auto l2norm = arma::norm(Btemp,2);
						//double Fnew = arma::sum(arma::log(1 + 1/ExpyXBnoji)) + ModelParams[0]*Btemp.n_nonzero + ModelParams[1]*arma::norm(Btemp,1) + ModelParams[2]*l2norm*l2norm;
						double Fnew = Objective(onemyxbnoji,Btemp);
						//std::cout<<"Fnew: "<<Fnew<<"Index: "<<i<<std::endl;
						if (Fnew < Fmin){
							Fmin = Fnew;
							maxindex = i;
							Bmaxindex = Binew;
						}

				}

			}

	}

	if (Fmin < objective){
		B[j] = 0;
		B[maxindex] = Bmaxindex;
		P.InitialSol = &B;
		P.b0 = b0;

		result = CDL012SquaredHinge(*X, *y, P).Fit();
		//ExpyXB = result.ExpyXB;
		onemyxb = 1 - *y % (*X * B + b0); // no need to calc. - change later.
		B = result.B;
		b0 = result.intercept;
		objective = result.Objective;
		Fmin = objective;
		foundbetter = true;
		break;

	}

}

		if(!foundbetter){result.Model = this; return result;}
	}


	//std::cout<<"Did not achieve CW Swap min" << std::endl;
	result.Model = this;
	return result;
}


inline double CDL012SquaredHingeSwaps::Objective(arma::vec & onemyxb, arma::sp_mat & B) { // hint inline
	auto l2norm = arma::norm(B,2);
	arma::uvec indices = arma::find(onemyxb > 0);
	return arma::sum(onemyxb.elem(indices) % onemyxb.elem(indices)) + ModelParams[0]*B.n_nonzero + ModelParams[1]*arma::norm(B,1) + ModelParams[2]*l2norm*l2norm;
}

/*
int main(){


	Params P;
	P.ModelType = "L012SquaredHingeSwaps";
	P.ModelParams = std::vector<double>{5,0,0.01};
	P.ActiveSet = true;
	P.ActiveSetNum = 6;
	P.Init = 'r';
	P.MaxIters = 200;
	//P.RandomStartSize = 100;


	arma::mat X;
	X.load("X.csv");

	arma::vec y;
	y.load("y.csv");

	std::vector<double> * Xtr = new std::vector<double>(X.n_cols);
	P.Xtr = Xtr;

	arma::mat Xscaled;
	arma::vec yscaled;

	arma::vec BetaMultiplier;
	arma::vec meanX;
	double meany;

	std::tie(BetaMultiplier, meanX, meany) = Normalize(X,y, Xscaled, yscaled, false);


	auto model = CDL012SquaredHingeSwaps(Xscaled, yscaled, P);
	auto result = model.Fit();

	arma::sp_mat B_unscaled;
	double intercept;
	std::tie(B_unscaled, intercept) = DeNormalize(result.B, BetaMultiplier, meanX, meany);

	result.B.print();
	B_unscaled.print();


	return 0;

}
*/
