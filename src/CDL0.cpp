#include "CDL0.h"

CDL0::CDL0(const arma::mat& Xi, const arma::vec& yi, const Params& P) : CD(Xi, yi, P) {thr = sqrt(2*ModelParams[0]); Xtr = P.Xtr; Iter = P.Iter; result.ModelParams = P.ModelParams;}

FitResult CDL0::Fit() {

	bool SecondPass = false;
	objective = Objective(r, B);

	for (uint t=0; t<MaxIters; ++t){

		//std::cout<< t << " " << objective << std::endl;
		Bprev = B;

		for (auto& i: Order){
			double cor = arma::dot(r,X->unsafe_col(i));
			(*Xtr)[i] = std::abs(cor); // do abs here instead from when sorting
			double Bi = B[i]; // B[i] is costly
			double x = cor + Bi;
			if (x >= thr || x <= -thr){	// often false so body is not costly
				r = r + X->unsafe_col(i)*(Bi - x);
				B[i] = x;
			}

			else if (Bi != 0) { // do nothing if x=0 and B[i] = 0
				r = r + X->unsafe_col(i)*Bi;
				B[i] = 0;
			}
		}


		if (Converged()){
			
			result.IterNum = t+1; 
			if (Stabilized == true && !SecondPass){
				Order = OldOrder; // Recycle over all coordinates to make sure the achieved point is a CW-min.
				SecondPass = true; // a 2nd pass will be performed
			}

			else{
				//std::cout<<"Converged in "<<t+1<<" iterations."<<std::endl;
				break;
			}

		}			if (ActiveSet){SupportStabilized();}

	}


	result.Objective = objective;
	result.B = B;
	result.Model = this;
	result.r = r; // change to pointer later
	return result;
}

inline double CDL0::Objective(arma::vec & r, arma::sp_mat & B) { // hint inline
	return 0.5*arma::dot(r,r) + ModelParams[0]*B.n_nonzero;
}