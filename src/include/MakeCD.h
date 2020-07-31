#ifndef MAKECD_H
#define MAKECD_H
#include "RcppArmadillo.h"
#include "Params.h"
#include "CD.h"
#include "CDL0.h"
#include "CDL012.h"
#include "CDL012Swaps.h"
#include "CDL012Logistic.h"
#include "CDL012SquaredHinge.h"
#include "CDL012LogisticSwaps.h"
#include "CDL012SquaredHingeSwaps.h"
//class CD;

template <typename T>
CD<T> * make_CD(const T& Xi, const arma::vec& yi, const Params<T>& P);

#endif 