#include "IHTL0.h"


IHTL0::IHTL0(const arma::mat& Xi, const arma::vec& yi, const Params& P) : CD(Xi, yi, P) {

	alpha = ModelParams[1];
	thr = sqrt(2*ModelParams[0]*alpha);}

FitResult IHTL0::Fit() {

	objective = Objective(r, B);

	for (unsigned int t=0; t<MaxIters; ++t){
		//std::cout<< t << " " << objective << std::endl;
		bool SameSupp = true;
		for (auto& i: Order){
			double x = alpha*(arma::dot(r,X->unsafe_col(i))+B[i]) + (1-alpha)*B[i];
			if (x >= thr || x <= -thr){	// often false so body is not costly
				if (SameSupp && B[i] == 0) {SameSupp = false;}
				B[i] = x;
			}
			else if (B[i] != 0) { // do nothing if x=0 and B[i] = 0
				SameSupp = false;
				B[i] = 0;
			}
		}
		r = *y - *X*B;
		//B.print();
		if (Converged()){break;}
		//if (ActiveSet){SupportStabilized(SameSupp);}
	}

	FitResult result;
	result.Objective = objective;
	result.B = B;
	result.Model = this;
	return result;
}

inline double IHTL0::Objective(arma::vec & r, arma::sp_mat & B) { // hint inline
	return 0.5*arma::dot(r,r) + ModelParams[0]*B.n_nonzero;
}