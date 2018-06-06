#include "CDL012LogisticSwaps.h"
#include "CDL012Logistic.h"

#include "Normalize.h" // Remove later
#include <chrono>

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

}

FitResult CDL012LogisticSwaps::Fit()
{
    auto start = std::chrono::high_resolution_clock::now();


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
            NnzIndices.push_back(it.row()); // i is
        }
        // Can easily shuffle here...
        //std::shuffle(std::begin(Order), std::end(Order), engine);
        foundbetter = false;


        for (auto& j : NnzIndices)
        {

            // Remove j
            arma::vec ExpyXBnoj = ExpyXB % arma::exp( - B[j] *  Xy->unsafe_col(j));

            auto start1 = std::chrono::high_resolution_clock::now();
            ///
            arma::rowvec gradient = - arma::sum( Xy->each_col() / (1 + ExpyXBnoj) , 0); // + twolambda2 * Biold // sum column-wise
            arma::uvec indices = arma::sort_index(arma::abs(gradient),"descend");
            bool foundbetteri = false;
            ///
            auto end1 = std::chrono::high_resolution_clock::now();
            std::cout<<"grad computation:  "<<std::chrono::duration_cast<std::chrono::milliseconds>(end1-start1).count() << " ms " << std::endl;


            auto start2 = std::chrono::high_resolution_clock::now();
            for(unsigned int ll = 0; ll < std::min(100, (int) p); ++ll)
            {
                unsigned int i = indices(ll);
                if(B[i] == 0)
                {
                    arma::vec ExpyXBnoji = ExpyXBnoj;

                    double Biold = 0;
                    double Binew;
                    //double partial_i = - arma::sum( (Xy->unsafe_col(i)) / (1 + ExpyXBnoji) ); // + twolambda2 * Biold
                    double partial_i = gradient[i];
                    bool Converged = false;
                    //if (std::abs(partial_i) >= lambda1 + stl0Lc )
                    //{
                        //std::cout<<"Adding: "<<i<< std::endl;
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
                          std::cout<<"swaps converged!!!"<<std::endl;

                        }

                        //Btemp[i] = Binew;
                        //double ObjTempOld = ObjTemp;
                        //ObjTemp = Objective(ExpyXBnoji, Btemp);
                        //if (std::abs(ObjTemp - ObjTempOld) / ObjTempOld < 0.001) {Converged = true;}
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
                      break;
                }
            }

            auto end2 = std::chrono::high_resolution_clock::now();
            std::cout<<"restricted:  "<<std::chrono::duration_cast<std::chrono::milliseconds>(end2-start2).count() << " ms " << std::endl;

            if (foundbetter){break;}

        }

        if(!foundbetter) {result.Model = this; return result;}
    }


    //std::cout<<"Did not achieve CW Swap min" << std::endl;
    result.Model = this;

    auto end = std::chrono::high_resolution_clock::now();
    std::cout<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << " ms " << std::endl;


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
