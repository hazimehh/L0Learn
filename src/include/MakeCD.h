#ifndef MAKECD_H
#define MAKECD_H
#include "Params.h"
#include "CD.h"
//class CD;

CD * make_CD(const arma::mat& Xi, const arma::vec& yi, const Params& P);

#endif