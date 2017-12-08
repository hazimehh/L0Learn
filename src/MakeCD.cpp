#include "MakeCD.h"
#include "CDL0.h"
#include "CDL012.h"
#include "CDL012Swaps.h"
#include "CDL012KSwapsExh.h"
#include "IHTL0.h"
#include "CDL1.h"
#include "CDL1Relaxed.h"
#include "CDL012Logistic.h"


CD * make_CD(const arma::mat& Xi, const arma::vec& yi, const Params& P){
	if (P.ModelType == "L0") {return new CDL0(Xi, yi, P);}
	else if (P.ModelType == "L012Swaps") {return new CDL012Swaps(Xi, yi, P);}
	else if (P.ModelType == "IHTL0") {return new IHTL0(Xi, yi, P);}
	else if (P.ModelType == "L1") {return new CDL1(Xi, yi, P);}
	else if (P.ModelType == "L1Relaxed") {return new CDL1Relaxed(Xi, yi, P);}
	else if (P.ModelType == "L012") {return new CDL012(Xi, yi, P);}
	else if (P.ModelType == "L012Logistic") {return new CDL012Logistic(Xi, yi, P);}
	else if (P.ModelType == "L012KSwaps") {return new CDL012KSwapsExh(Xi, yi, P);}
}
