#include "CDL012Logistic.h"

#include "Normalize.h" // Remove later
#include "Grid.h" // Remove later


CDL012Logistic::CDL012Logistic(const arma::mat& Xi, const arma::vec& yi, const Params& P) : CD(Xi, yi, P)
{
    //LipschitzConst = 0.25; // for logistic loss function only
    twolambda2 = 2 * ModelParams[2];
    qp2lamda2 = (LipschitzConst + twolambda2); // this is the univariate lipschitz const of the differentiable objective
    thr = std::sqrt((2 * ModelParams[0]) / qp2lamda2);
    lambda1 = ModelParams[1];
    lambda1ol = lambda1 / qp2lamda2;
    b0 = P.b0; // Initialize from previous later....!
    ExpyXB = arma::exp(*y % (*X * B + b0)); // Maintained throughout the algorithm
    Xtr = P.Xtr; Iter = P.Iter; result.ModelParams = P.ModelParams; Xy = P.Xy;
    NoSelectK = P.NoSelectK;
}

FitResult CDL012Logistic::Fit()
{
    // arma::mat Xy = X->each_col() % *y; // later

    bool SecondPass = false;
    double NewtonStepSize = 1.0; //Newton's update step size.
    objective = Objective(r, B); ////////

    for (unsigned int t = 0; t < MaxIters; ++t)
    {
        //std::cout<<"CDL012 Logistic: "<< t << " " << objective << " Gamma= "<<NewtonStepSize <<std::endl;
        double Oldobjective = objective;
        Bprev = B;

        // Update the intercept
        double b0old = b0;
        double partial_b0 = - arma::sum( *y / (1 + ExpyXB) );
        b0 -= partial_b0 / (n * LipschitzConst); // intercept is not regularized
        ExpyXB %= arma::exp( (b0 - b0old) * *y);
        //std::cout<<"Intercept. "<<Objective(r,B)<<std::endl;

        if(!Stabilized || SecondPass)
        {
            for (auto& i : Order)
            {

                // Calculate Partial_i
                double Biold = B[i];
                double partial_i = - arma::sum( (Xy->unsafe_col(i)) / (1 + ExpyXB) ) + twolambda2 * Biold;
                (*Xtr)[i] = std::abs(partial_i); // abs value of grad

                double x = Biold - partial_i / qp2lamda2;
                double z = std::abs(x) - lambda1ol;


                if (z >= thr || i < NoSelectK) 	// often false so body is not costly
                {
                    double Bnew = std::copysign(z, x);
                    B[i] = Bnew;
                    ExpyXB %= arma::exp( (Bnew - Biold) *  Xy->unsafe_col(i));
                    //std::cout<<"In. "<<Objective(r,B)<<std::endl;
                }

                else if (Biold != 0)   // do nothing if x=0 and B[i] = 0
                {
                    ExpyXB %= arma::exp( - Biold * Xy->unsafe_col(i));
                    B[i] = 0;
                    //std::cout<<"Out. "<<Objective(r,B)<<std::endl;
                    //}

                    //else{
                    //	std::cout<<"ELSE: "<<B[i]<<std::endl;
                    //}
                }
            }
        }

        else
        {
            for (auto& i : Order)
            {
                //std::cout<<"In Stabilization!!!"<<std::endl;

                // Calculate Partial_i
                double Biold = B[i];
                double partial_i = - arma::sum( (Xy->unsafe_col(i)) / (1 + ExpyXB) ) + twolambda2 * Biold;
                double partial2_i = arma::sum( (X->unsafe_col(i) % X->unsafe_col(i) % ExpyXB) / ( (1 + ExpyXB) % (1 + ExpyXB) ) ) + twolambda2;
                (*Xtr)[i] = std::abs(partial_i); // abs value of grad
                //std::cout<<partial2_i<<std::endl;
                double x;
                double TruncatedNewtonStep = NewtonStepSize / partial2_i;
                if (TruncatedNewtonStep > 1 / qp2lamda2) // if truncated step is larger use it
                {
                    x = Biold - NewtonStepSize * partial_i / partial2_i;
                }
                else
                {
                    x = Biold - partial_i / qp2lamda2;
                }

                double z = std::abs(x) - lambda1ol;


                if (z >= thr) 	// often false so body is not costly
                {
                    double Bnew = std::copysign(z, x);
                    B[i] = Bnew;
                    ExpyXB %= arma::exp( (Bnew - Biold) *  Xy->unsafe_col(i));
                    //std::cout<<"In. "<<Objective(r,B)<<std::endl;
                }

                else if (Biold != 0)   // do nothing if x=0 and B[i] = 0
                {
                    ExpyXB %= arma::exp( - Biold * Xy->unsafe_col(i));
                    B[i] = 0;
                    //std::cout<<"Out. "<<Objective(r,B)<<std::endl;
                }

                //else{
                //	std::cout<<"ELSE: "<<B[i]<<std::endl;
                //}
            }

        }



        //B.print();
        if (Converged())
        {
            if (Stabilized == true && !SecondPass)
            {
                Order = OldOrder; // Recycle over all coordinates to make sure the achieved point is a CW-min.
                //SecondPass = true; // a 2nd pass will be performed
            }

            else
            {
                //std::cout<<"Converged in "<<t+1<<" iterations."<<std::endl;
                break;
            }

        }

        // New objective is obtained after this point
        if (objective > Oldobjective)
        {
            // Reduce step size if the obj increases at some pt.
            NewtonStepSize /= 2.0;

            // Reset B and objective
            B = Bprev;
            objective = Oldobjective;

        }


        if (ActiveSet) {SupportStabilized();}

    }

    result.Objective = objective;
    result.B = B;
    result.Model = this;
    result.intercept = b0;
    result.ExpyXB = ExpyXB;
    result.IterNum = CurrentIters;

    return result;
}

inline double CDL012Logistic::Objective(arma::vec & r, arma::sp_mat & B)   // hint inline
{
    auto l2norm = arma::norm(B, 2);
    // arma::sum(arma::log(1 + 1 / ExpyXB)) is the negative log-likelihood
    return arma::sum(arma::log(1 + 1 / ExpyXB)) + ModelParams[0] * B.n_nonzero + ModelParams[1] * arma::norm(B, 1) + ModelParams[2] * l2norm * l2norm;
}


/*
int main(){


	Params P;
	P.ModelType = "L012Logistic";
	P.ModelParams = std::vector<double>{18.36,0,0.01};
	P.ActiveSet = true;
	P.ActiveSetNum = 6;
	P.Init = 'z';
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


	auto model = CDL012Logistic(Xscaled, yscaled, P);
	auto result = model.Fit();

	arma::sp_mat B_unscaled;
	double intercept;
	std::tie(B_unscaled, intercept) = DeNormalize(result.B, BetaMultiplier, meanX, meany);

	result.B.print();
	B_unscaled.print();


	//std::cout<<result.intercept<<std::endl;
	//arma::sign(Xscaled*result.B + result.intercept).print();
	//std::cout<<"#############"<<std::endl;
	//arma::sign(X*B_unscaled + result.intercept).print();



	// GridParams PG;
	// PG.Type = "L0Logistic";
	// PG.NnzStopNum = 12;
  //
	// arma::mat X;
	// X.load("X.csv");
  //
	// arma::vec y;
	// y.load("y.csv");
	// auto g = Grid(X, y, PG);
  //
	// g.Fit();
  //
	// unsigned int i = g.NnzCount.size();
	// for (uint j=0;j<i;++j){
	// 	std::cout<<g.NnzCount[j]<<"   "<<g.Lambda0[j]<<std::endl;
	// }



	return 0;

}
*/
