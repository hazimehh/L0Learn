#include "CDL0.h"

CDL0::CDL0(const arma::mat& Xi, const arma::vec& yi, const Params& P) : CD(Xi, yi, P)
{
    thr = sqrt(2 * ModelParams[0]); Xtr = P.Xtr; Iter = P.Iter;
    result.ModelParams = P.ModelParams; ScreenSize = P.ScreenSize;
    r = *P.r; Range1p.resize(p); std::iota(std::begin(Range1p), std::end(Range1p), 0);
}

FitResult CDL0::Fit()
{

    bool SecondPass = false;
    objective = Objective(r, B);

    std::vector<unsigned int> FullOrder = Order;
    bool FirstRestrictedPass = true;
    if (ActiveSet)
    {
        Order.resize(std::min((int) (B.n_nonzero + ScreenSize), (int)(p))); // std::min(1000,Order.size())
    }

    bool ActiveSetInitial = ActiveSet;

    for (unsigned int t = 0; t < MaxIters; ++t)
    {
        Bprev = B;

        for (auto& i : Order)
        {
            double cor = arma::dot(r, X->unsafe_col(i));
            (*Xtr)[i] = std::abs(cor); // do abs here instead from when sorting
            double Bi = B[i]; // B[i] is costly
            double x = cor + Bi;
            if (x >= thr || x <= -thr) 	// often false so body is not costly
            {
                r += X->unsafe_col(i) * (Bi - x);
                B[i] = x;
            }

            else if (Bi != 0)   // do nothing if x=0 and B[i] = 0
            {
                r += X->unsafe_col(i) * Bi;
                B[i] = 0;
            }
        }

        if (Converged())
        {
            result.IterNum = t + 1;

            if(FirstRestrictedPass && ActiveSetInitial)
            {
                if (CWMinCheck()) {break;}
                FirstRestrictedPass = false;
                Stabilized = false;
                ActiveSet = true;
                Order = FullOrder;

            }

            else
            {
                if (Stabilized == true && ActiveSetInitial)  // && !SecondPass
                {
                    if (CWMinCheck()) {break;}
                    Order = OldOrder; // Recycle over all coordinates to make sure the achieved point is a CW-min.
                    //SecondPass = true; // a 2nd pass will be performed
                    Stabilized = false;
                    ActiveSet = true;
                }

                else
                {
                    break;
                }
            }

        }

        if (ActiveSet) {SupportStabilized();}

    }


    result.Objective = objective;
    result.B = B;
    result.Model = this;
    result.r = r; // change to pointer later
    return result;
}

inline double CDL0::Objective(arma::vec & r, arma::sp_mat & B)   // hint inline
{
    return 0.5 * arma::dot(r, r) + ModelParams[0] * B.n_nonzero;
}


bool CDL0::CWMinCheck()
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
        // B[i] = 0 in this case!
        if (absx >= thr) 	// often false so body is not costly
        {
            r -= X->unsafe_col(i) * x;
            B[i] = x;
            Cwmin = false;
        }
    }
    return Cwmin;

}
