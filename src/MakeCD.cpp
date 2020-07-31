#include "MakeCD.h"

template <typename T>
CD<T> * make_CD(const T& Xi, const arma::vec& yi, const Params<T>& P)
{

    if (P.Specs.SquaredError)
    {
        if (P.Specs.CD)
        {
            if (P.Specs.L0) {return new CDL0<T>(Xi, yi, P);}
            else {return new CDL012<T>(Xi, yi, P);}
        }
        else if (P.Specs.PSI) {return new CDL012Swaps<T>(Xi, yi, P);}
    }

    else if (P.Specs.Logistic)
    {
        if (P.Specs.CD) {return new CDL012Logistic<T>(Xi, yi, P);}
        else if (P.Specs.PSI) {return new CDL012LogisticSwaps<T>(Xi, yi, P);}
    }

    else if (P.Specs.SquaredHinge)
    {
        if (P.Specs.CD) {return new CDL012SquaredHinge<T>(Xi, yi, P);}
        else if (P.Specs.PSI) {return new CDL012SquaredHingeSwaps<T>(Xi, yi, P);}
    }

    return new CDL0<T>(Xi, yi, P); // handle later

}
