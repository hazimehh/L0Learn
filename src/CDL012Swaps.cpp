#include "CDL012Swaps.h"
#include "CDL012.h"

CDL012Swaps::CDL012Swaps(const arma::mat& Xi, const arma::vec& yi, const Params& Pi) : CD(Xi, yi, Pi) {MaxNumSwaps = Pi.MaxNumSwaps; P = Pi; NoSelectK=P.NoSelectK;}

FitResult CDL012Swaps::Fit()
{
    auto result = CDL012(*X, *y, P).Fit(); // result will be maintained till the end
    B = result.B;
    objective = result.Objective;
    P.Init = 'u';

    bool foundbetter;
    for (unsigned int t = 0; t < MaxNumSwaps; ++t)
    {
        //std::cout<<"1Swaps Iteration: "<<t<<". "<<"Obj: "<<objective<<std::endl;
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



        //arma::uvec Sc (Sctemp);


        arma::vec r = *y - *X * B; // ToDO: take from CD !! No need for this computation.


        for (auto& i : NnzIndices)
        {


            arma::rowvec riX = (r + B[i] * X->unsafe_col(i)).t() * *X; // expensive computation ## call the new function here, inlined? ##
            //std::cout<<"HERE1!!!" << std::endl;

            //auto maxindex = arma::index_max(arma::abs(riX.elem(Sc)));

            //std::vector<arma::uword> Sctemp; // ToDO: Very slow change later..
            double maxcorr = -1;
            unsigned int maxindex;
            for(unsigned int j = NoSelectK; j < p; ++j) // Can be made much faster..
            {
                if (std::fabs(riX[j]) > maxcorr && B[j] == 0)
                {
                    maxcorr = std::fabs(riX[j]);
                    maxindex = j;
                }
            }

            if(maxcorr > (1 + 2 * ModelParams[2])*std::fabs(B[i]) + ModelParams[1] )
            {
                B[i] = 0;
                B[maxindex] = (riX[maxindex] - std::copysign(ModelParams[1], riX[maxindex])) / (1 + 2 * ModelParams[2]);
                P.InitialSol = &B;
                result = CDL012(*X, *y, P).Fit();
                B = result.B;
                objective = result.Objective;
                foundbetter = true;
                break;

            }
        }

        if(!foundbetter) {result.Model = this; return result;}
    }


    //std::cout<<"Did not achieve CW Swap min" << std::endl;
    result.Model = this;
    return result;
}

inline double CDL012Swaps::Objective(arma::vec & r, arma::sp_mat & B)   // hint inline
{
    auto l2norm = arma::norm(B, 2);
    return 0.5 * arma::dot(r, r) + ModelParams[0] * B.n_nonzero + ModelParams[1] * arma::norm(B, 1) + ModelParams[2] * l2norm * l2norm;
}
