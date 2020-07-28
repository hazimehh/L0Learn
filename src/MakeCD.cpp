#include "MakeCD.h"


CD * make_CD(const arma::mat& Xi, const arma::vec& yi, const Params& P)
{

    if (P.Specs.SquaredError)
    {
        if (P.Specs.CD)
        {
            if (P.Specs.L0) {return new CDL0(Xi, yi, P);}
            else {return new CDL012(Xi, yi, P);}
        }
        else if (P.Specs.PSI) {return new CDL012Swaps(Xi, yi, P);}
    }

    else if (P.Specs.Logistic)
    {
        if (P.Specs.CD) {return new CDL012Logistic(Xi, yi, P);}
        else if (P.Specs.PSI) {return new CDL012LogisticSwaps(Xi, yi, P);}
    }

    else if (P.Specs.SquaredHinge)
    {
        if (P.Specs.CD) {return new CDL012SquaredHinge(Xi, yi, P);}
        else if (P.Specs.PSI) {return new CDL012SquaredHingeSwaps(Xi, yi, P);}
    }

    return new CDL0(Xi, yi, P); // handle later

}
