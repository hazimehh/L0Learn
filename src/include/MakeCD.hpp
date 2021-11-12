#ifndef MAKECD_H
#define MAKECD_H
#include <armadillo>
#include "Params.hpp"
#include "CD.hpp"
#include "CDL0.hpp"
#include "CDL012.hpp"
#include "CDL012Swaps.hpp"
#include "CDL012Logistic.hpp"
#include "CDL012SquaredHinge.hpp"
#include "CDL012LogisticSwaps.hpp"
#include "CDL012SquaredHingeSwaps.hpp"


template <class T>
CDBase<T> * make_CD(const T& Xi, const arma::vec& yi, const Params<T>& P) {
    if (P.Specs.SquaredError) {
        if (P.Specs.CD) {
            if (P.Specs.L0) {
                return new CDL0<T>(Xi, yi, P);
            } else {
                return new CDL012<T>(Xi, yi, P);
            }
        } else if (P.Specs.PSI) {
            return new CDL012Swaps<T>(Xi, yi, P);
        }
    } else if (P.Specs.Logistic) {
        if (P.Specs.CD) {
            return new CDL012Logistic<T>(Xi, yi, P);
        } else if (P.Specs.PSI) {
            return new CDL012LogisticSwaps<T>(Xi, yi, P);
        }
    } else if (P.Specs.SquaredHinge) {
        if (P.Specs.CD) {
            return new CDL012SquaredHinge<T>(Xi, yi, P);
        } else if (P.Specs.PSI) {
            return new CDL012SquaredHingeSwaps<T>(Xi, yi, P);
        }
    }
    return new CDL0<T>(Xi, yi, P); // handle later
}


#endif 