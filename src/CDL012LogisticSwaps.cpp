#include "CDL012LogisticSwaps.h"
#include "CDL012Logistic.h"

#include "Normalize.h" // Remove later
#include <chrono>
#include <algorithm>

CDL012LogisticSwaps::CDL012LogisticSwaps(const arma::mat& Xi, const arma::vec& yi, const Params& Pi) : CD(Xi, yi, Pi)
{
    MaxNumSwaps = Pi.MaxNumSwaps;
    P = Pi;
    twolambda2 = 2 * ModelParams[2];
    qp2lamda2 = (LipschitzConst + twolambda2); // this is the univariate lipschitz const of the differentiable objective
    thr = std::sqrt((2 * ModelParams[0]) / qp2lamda2);
    stl0Lc = std::sqrt((2 * ModelParams[0]) * qp2lamda2);
    lambda1 = ModelParams[1];
    lambda1ol = lambda1 / qp2lamda2;
    Xtr = P.Xtr; Iter = P.Iter; result.ModelParams = P.ModelParams; Xy = P.Xy;
    NoSelectK = P.NoSelectK;
}

FitResult CDL012LogisticSwaps::Fit()
{
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
    for (unsigned int t = 0; t < MaxNumSwaps; ++t)
    {
        std::cout<<"1Swaps Iteration: "<<t<<". "<<"Obj: "<<objective<<std::endl;
        //B.print();
        arma::sp_mat::const_iterator start = B.begin();
        arma::sp_mat::const_iterator end   = B.end();
        std::vector<unsigned int> NnzIndices;
        for(arma::sp_mat::const_iterator it = start; it != end; ++it)
        {
          if (it.row() >= NoSelectK){NnzIndices.push_back(it.row());}
        }
        // Can easily shuffle here...
        //std::shuffle(std::begin(Order), std::end(Order), engine);
        foundbetter = false;

	arma::uvec screened_order = arma::conv_to< arma::uvec >::from(std::vector<unsigned int>(P.Uorder.begin(), P.Uorder.begin() + std::min(100, P.Uorder.size())));

        for (auto& j : NnzIndices)
        {

            // Remove j
            arma::vec ExpyXBnoj = ExpyXB % arma::exp( - B[j] *  Xy->unsafe_col(j));

            //auto start1 = std::chrono::high_resolution_clock::now();
            ///
		
            arma::rowvec gradient = - arma::sum( Xy->each_col(screened_order) / (1 + ExpyXBnoj) , 0); // + twolambda2 * Biold // sum column-wise
            arma::uvec indices = arma::sort_index(arma::abs(gradient),"descend");
            bool foundbetteri = false;
            ///
            //auto end1 = std::chrono::high_resolution_clock::now();
            //std::cout<<"grad computation:  "<<std::chrono::duration_cast<std::chrono::milliseconds>(end1-start1).count() << " ms " << std::endl;


            //auto start2 = std::chrono::high_resolution_clock::now();
            // Later: make sure this scans at least 10 coordinates from outside supp (now it does not)
            for(unsigned int ll = 0; ll < std::min(5, (int) p); ++ll)
            {
                unsigned int i = screened_order(indices(ll));
                if(B[i] == 0 && i >= NoSelectK)
                {
                    arma::vec ExpyXBnoji = ExpyXBnoj;

                    double Biold = 0;
                    double Binew;
                    //double partial_i = - arma::sum( (Xy->unsafe_col(i)) / (1 + ExpyXBnoji) ); // + twolambda2 * Biold
                    double partial_i = gradient[i];
                    bool Converged = false;

                    arma::sp_mat Btemp = B;
                    Btemp[j] = 0;
                    double ObjTemp = Objective(ExpyXBnoji, Btemp);
                    unsigned int innerindex = 0;

                    double x = Biold - partial_i/qp2lamda2;
                    double z = std::abs(x) - lambda1ol;
                    Binew = std::copysign(z, x); // no need to check if >= sqrt(2lambda_0/Lc)

                    while(!Converged && innerindex < 5  && ObjTemp >= Fmin) // ObjTemp >= Fmin
                    {
                        ExpyXBnoji %= arma::exp( (Binew - Biold) *  Xy->unsafe_col(i));
                        partial_i = - arma::sum( (Xy->unsafe_col(i)) / (1 + ExpyXBnoji) ) + twolambda2 * Binew;

                        if (std::abs((Binew - Biold)/Biold) < 0.0001){
                          Converged = true;
                          //std::cout<<"swaps converged!!!"<<std::endl;

                        }

                        Biold = Binew;
                        x = Biold - partial_i/qp2lamda2;
                        z = std::abs(x) - lambda1ol;
                        Binew = std::copysign(z, x); // no need to check if >= sqrt(2lambda_0/Lc)
                        innerindex += 1;
                    }


                    Btemp[i] = Binew;
                    ObjTemp = Objective(ExpyXBnoji, Btemp);

                    if (ObjTemp < Fmin)
                    {
                        Fmin = ObjTemp;
                        maxindex = i;
                        Bmaxindex = Binew;
                        foundbetteri = true;
                    }

                    // Can be made much faster (later)
                    Btemp[i] = Binew;

                    //}



                }

                if (foundbetteri == true)
                {
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
		      std::cout<<"Found better!!!"<< std::endl;
                      break;
                }
            }

            //auto end2 = std::chrono::high_resolution_clock::now();
            //std::cout<<"restricted:  "<<std::chrono::duration_cast<std::chrono::milliseconds>(end2-start2).count() << " ms " << std::endl;

            if (foundbetter){break;}

        }

        if(!foundbetter)
        {
          //result.Model = this;
          return result;
        }
    }

    //result.Model = this;
    return result;
}


inline double CDL012LogisticSwaps::Objective(arma::vec & r, arma::sp_mat & B)   // hint inline
{
    auto l2norm = arma::norm(B, 2);
    return arma::sum(arma::log(1 + 1 / r)) + ModelParams[0] * B.n_nonzero + ModelParams[1] * arma::norm(B, 1) + ModelParams[2] * l2norm * l2norm;
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
