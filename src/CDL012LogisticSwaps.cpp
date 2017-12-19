#include "CDL012LogisticSwaps.h"
#include "CDL012Logistic.h"

#include "Normalize.h" // Remove later

CDL012LogisticSwaps::CDL012LogisticSwaps(const arma::mat& Xi, const arma::vec& yi, const Params& Pi) : CD(Xi, yi, Pi)
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

FitResult CDL012LogisticSwaps::Fit() {
	auto result = CDL012Logistic(*X, *y, P).Fit(); // result will be maintained till the end
	b0 = result.intercept; // Initialize from previous later....!
	B = result.B;
	ExpyXB = result.ExpyXB; // Maintained throughout the algorithm

	objective = result.Objective;
	double Fmin = objective;
	unsigned int maxindex;
	double Bmaxindex;

	P.Init = 'u';

	bool foundbetter;
	for (unsigned int t=0; t<MaxNumSwaps; ++t){
		//std::cout<<"1Swaps Iteration: "<<t<<". "<<"Obj: "<<objective<<std::endl;
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
			arma::vec ExpyXBnoj = ExpyXB % arma::exp( - B[j] *  *y % X->unsafe_col(j));


			for(unsigned int i =0; i<p; ++i){
				if(B[i] == 0){
					arma::vec ExpyXBnoji = ExpyXBnoj;

					double Biold = 0;
					double Binew;
					double partial_i = - arma::sum( (*y % X->unsafe_col(i)) / (1 + ExpyXBnoji) ); // + twolambda2 * Biold

					bool Converged = false;
					if (std::abs(partial_i) >= lambda1 + stl0Lc ){
						double partial2_i = arma::sum( (X->unsafe_col(i) % X->unsafe_col(i) % ExpyXBnoji) / ( (1 + ExpyXBnoji) % (1 + ExpyXBnoji) ) ) + twolambda2;

						std::cout<<"Adding: "<<i<< std::endl;

						unsigned int l = 0;
						arma::sp_mat Btemp = B;
						Btemp[j] = 0;
						double ObjTemp = Objective(ExpyXBnoji,Btemp);
						double NewtonStepSize = 1.0;
						double Biolddescent = 0;
						while(!Converged)
						{
							//double x = Biold - partial_i/qp2lamda2;

							double x;
							double TruncatedNewtonStep = NewtonStepSize/partial2_i;
							if (TruncatedNewtonStep > 1/qp2lamda2){ // if truncated step is larger use it
								x = Biold - NewtonStepSize*partial_i/partial2_i;
							}
							else{
								std::cout<<"Netwon: "<<TruncatedNewtonStep<<" Normal: "<<1/qp2lamda2<<std::endl;
								std::cout<<"Truncated Newton is smaller!"<<std::endl;
								x = Biold - partial_i/qp2lamda2;
							}


							double z = std::abs(x) - lambda1ol;

							Binew = std::copysign(z, x); // no need to check if >= sqrt(2lambda_0/Lc)
							ExpyXBnoji %= arma::exp( (Binew - Biold) *  *y % X->unsafe_col(i));
							//std::cout<<"Here222!!"<<std::endl;


							partial_i = - arma::sum( (*y % X->unsafe_col(i)) / (1 + ExpyXBnoji) ) + twolambda2 * Binew;
							partial2_i = arma::sum( (X->unsafe_col(i) % X->unsafe_col(i) % ExpyXBnoji) / ( (1 + ExpyXBnoji) % (1 + ExpyXBnoji) ) ) + twolambda2;

							//if (std::abs((Binew - Biold)/Biold) < 0.000001){Converged = true;}


							if (l%3 == 0){
								Btemp[i] = Binew;
								double ObjTempOld = ObjTemp;
								ObjTemp = Objective(ExpyXBnoji,Btemp);
								std::cout<<"O: "<<ObjTemp<<" Gamma "<<NewtonStepSize<<std::endl;
								if (ObjTemp > ObjTempOld){
									NewtonStepSize /= 2.0;
									Binew = Biolddescent;
									ObjTemp = ObjTempOld;
								}
								else{
									// If the objective decrease set Biolddescent
									Biolddescent = Binew;
								}
								if (std::abs(ObjTemp - ObjTempOld)/ObjTempOld < 0.0001){Converged = true;}
							}

							Biold = Binew;
							l += 1;

						}

						// Can be made much faster (later)
						Btemp[i] = Binew;
						//auto l2norm = arma::norm(Btemp,2);
						//double Fnew = arma::sum(arma::log(1 + 1/ExpyXBnoji)) + ModelParams[0]*Btemp.n_nonzero + ModelParams[1]*arma::norm(Btemp,1) + ModelParams[2]*l2norm*l2norm;
						double Fnew = Objective(ExpyXBnoji,Btemp);
						std::cout<<"Fnew: "<<Fnew<<"Index: "<<i<<std::endl;
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

		result = CDL012Logistic(*X, *y, P).Fit();
		ExpyXB = result.ExpyXB;
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


inline double CDL012LogisticSwaps::Objective(arma::vec & r, arma::sp_mat & B) { // hint inline
	auto l2norm = arma::norm(B,2);
	return arma::sum(arma::log(1 + 1/r)) + ModelParams[0]*B.n_nonzero + ModelParams[1]*arma::norm(B,1) + ModelParams[2]*l2norm*l2norm;
}

/*
int main(){


	Params P;
	P.ModelType = "L012LogisticSwaps";
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


	auto model = CDL012LogisticSwaps(Xscaled, yscaled, P);
	auto result = model.Fit();

	arma::sp_mat B_unscaled;
	double intercept;
	std::tie(B_unscaled, intercept) = DeNormalize(result.B, BetaMultiplier, meanX, meany);

	result.B.print();
	B_unscaled.print();


	return 0;

}
*/
