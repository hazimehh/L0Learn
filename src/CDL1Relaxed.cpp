#include "CDL1Relaxed.h"


CDL1Relaxed::CDL1Relaxed(const arma::mat& Xi, const arma::vec& yi, const Params& P) : CD(Xi, yi, P) {thr = ModelParams[0]; Xtr = P.Xtr; result.ModelParams = P.ModelParams;}

FitResult CDL1Relaxed::Fit()
{

    bool SecondPass = false;
    objective = Objective(r, B);

    for (unsigned int t = 0; t < MaxIters; ++t)
    {
        Bprev = B;
        //std::cout<< t << " " << objective << std::endl;
        bool SameSupp = true;
        for (auto& i : Order)
        {
            double cor = arma::dot(r, X->unsafe_col(i));
            (*Xtr)[i] = std::abs(cor);
            double x = cor + B[i];
            if (x >= thr) 	// often false so body is not costly
            {
                double Biold = B[i];
                if (SameSupp && B[i] == 0) {SameSupp = false;}
                B[i] = x - thr;
                r = r + X->unsafe_col(i) * (Biold - B[i]);
            }
            else if (x <= -thr)
            {
                double Biold = B[i];
                if (SameSupp && B[i] == 0) {SameSupp = false;}
                B[i] = x + thr;
                r = r + X->unsafe_col(i) * (Biold - B[i]);
            }
            else if (B[i] != 0)   // do nothing if x=0 and B[i] = 0
            {
                r = r + X->unsafe_col(i) * B[i];
                SameSupp = false;
                B[i] = 0;
            }
        }
        //B.print();
        if (Converged())
        {

            result.IterNum = t + 1;
            if (Stabilized == true && !SecondPass)
            {
                Order = OldOrder; // Recycle over all coordinates to make sure the achieved point is a CW-min.
                SecondPass = true; // a 2nd pass will be performed
            }

            else
            {
                //std::cout<<"Converged in "<<t+1<<" iterations."<<std::endl;
                break;
            }

        }
        if (ActiveSet) {SupportStabilized();}
    }


    std::vector<unsigned int> Support;

    arma::sp_mat::const_iterator i;
    for(i = B.begin(); i != B.end(); ++i)
    {
        Support.push_back(i.row());
    }

    arma::uvec aSupport = arma::conv_to< arma::uvec >::from(Support);
    arma::mat XLS = X->submat(arma::regspace<arma::uvec>(0, n - 1), aSupport);
    //arma::vec BLS = arma::inv_sympd(XLS.t()*XLS)*XLS.t()* *y;
    arma::vec BLS = arma::solve(XLS, *y); // arma::pinv(XLS)* *y;

    arma::sp_mat BLSE (p, 1);
    for(unsigned int i = 0; i < Support.size(); ++i)
    {
        BLSE[Support[i]] = BLS[i];
    }

    B = ModelParams[1] * B + (1 - ModelParams[1]) * BLSE;

    result.Objective = objective;
    result.B = B;
    result.Model = this;
    return result;
}

inline double CDL1Relaxed::Objective(arma::vec & r, arma::sp_mat & B)   // hint inline
{
    return 0.5 * arma::dot(r, r) + ModelParams[0] * arma::norm(B, 1);
}