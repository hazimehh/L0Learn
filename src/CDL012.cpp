#include "CDL012.h"


CDL012::CDL012(const arma::mat& Xi, const arma::vec& yi, const Params& P) : CD(Xi, yi, P)
{
    thr = std::sqrt((2 * ModelParams[0]) / (1 + 2 * ModelParams[2]));
    Onep2lamda2 = 1 + 2 * ModelParams[2]; lambda1 = ModelParams[1]; Xtr = P.Xtr; Iter = P.Iter; result.ModelParams = P.ModelParams; ScreenSize = P.ScreenSize; r = *P.r;
    Range1p.resize(p); std::iota(std::begin(Range1p), std::end(Range1p), 0); NoSelectK = P.NoSelectK;
    result.r = P.r;
}

FitResult CDL012::Fit()
{


    objective = Objective(r, B);

    std::vector<unsigned int> FullOrder = Order;
    if (ActiveSet)
    {
        Order.resize(std::min((int) (B.n_nonzero + ScreenSize + NoSelectK), (int)(p))); // std::min(1000,Order.size())
    }


    for (unsigned int t = 0; t < MaxIters; ++t)
    {
        Bprev = B;

        for (auto& i : Order)
        {
            double cor = arma::dot(r, X->unsafe_col(i));

            (*Xtr)[i] = std::abs(cor); // do abs here instead from when sorting

            double Bi = B[i]; // B[i] is costly
            double x = cor + Bi; // x is beta_tilde_i
            double z = (std::abs(x) - lambda1) / Onep2lamda2;

            if (z >= thr || (i < NoSelectK && z>0) ) 	// often false so body is not costly
            {
                B[i] = std::copysign(z, x);

                r = r + X->unsafe_col(i) * (Bi - B[i]);
            }

            else if (Bi != 0)   // do nothing if x=0 and B[i] = 0
            {
                r = r + X->unsafe_col(i) * Bi;
                B[i] = 0;
            }
        }

        SupportStabilized();
        // only way to terminate is by (i) converging on active set and (ii) CWMinCheck
        if (Converged())
        {
            if (CWMinCheck()) {break;}
        }

    }

    result.Objective = objective;
    result.B = B;
    //result.Model = this;
    *(result.r) = r; // change to pointer later
    result.IterNum = CurrentIters;
    return result;
}

inline double CDL012::Objective(arma::vec & r, arma::sp_mat & B)   // hint inline
{
    auto l2norm = arma::norm(B, 2);
    return 0.5 * arma::dot(r, r) + ModelParams[0] * B.n_nonzero + ModelParams[1] * arma::norm(B, 1) + ModelParams[2] * l2norm * l2norm;
}


bool CDL012::CWMinCheck()
{
    // Get the Sc = FullOrder - Order
    std::vector<unsigned int> S;
    for(arma::sp_mat::const_iterator it = B.begin(); it != B.end(); ++it) {S.push_back(it.row());}

    std::vector<unsigned int> Sc;
    set_difference(
        Range1p.begin(),
        Range1p.end(),
        S.begin(),
        S.end(),
        back_inserter(Sc));

    bool Cwmin = true;
    for (auto& i : Sc)
    {
        double x = arma::dot(r, X->unsafe_col(i));
        double absx = std::abs(x);
        (*Xtr)[i] = absx; // do abs here instead from when sorting
        double z = (absx - lambda1) / Onep2lamda2;

        if (z > thr) 	// often false so body is not costly
        {
            B[i] = std::copysign(z, x);
            r -= X->unsafe_col(i) * B[i];
            Cwmin = false;
            Order.push_back(i);
        }
    }
    return Cwmin;

}
