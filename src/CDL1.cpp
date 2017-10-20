#include "CDL1.h"

CDL1::CDL1(const arma::mat& Xi, const arma::vec& yi, const Params& P) : CD(Xi, yi, P) {thr = ModelParams[0];Xtr = P.Xtr;result.ModelParams = P.ModelParams;}

FitResult CDL1::Fit() {

	bool SecondPass = false;
	objective = Objective(r, B);

	for (uint t=0; t<MaxIters; ++t){
		Bprev = B;
		//std::cout<< t << " " << objective << std::endl;
		bool SameSupp = true;
		for (auto& i: Order){
			double cor = arma::dot(r,X->unsafe_col(i));
			(*Xtr)[i] = std::abs(cor);
			double x = cor + B[i];
			if (x >= thr){	// often false so body is not costly
				double Biold = B[i];
				if (SameSupp && B[i] == 0) {SameSupp = false;}
				B[i] = x - thr;
				r = r + X->unsafe_col(i)*(Biold - B[i]);
			}
			else if (x <= -thr){
				double Biold = B[i];
				if (SameSupp && B[i] == 0) {SameSupp = false;}
				B[i] = x + thr;
				r = r + X->unsafe_col(i)*(Biold - B[i]);
			}
			else if (B[i] != 0) { // do nothing if x=0 and B[i] = 0
				r = r + X->unsafe_col(i)*B[i];
				SameSupp = false;
				B[i] = 0;
			}
		}
		//B.print();
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

		}
		if (ActiveSet){SupportStabilized();}
	}

	result.Objective = objective;
	result.B = B;
	result.Model = this;
	return result;
}

inline double CDL1::Objective(arma::vec & r, arma::sp_mat & B) { // hint inline
	return 0.5*arma::dot(r,r) + ModelParams[0]*arma::norm(B,1);
}