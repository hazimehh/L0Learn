#include "CDL012.h"


CDL012::CDL012(const arma::mat& Xi, const arma::vec& yi, const Params& P) : CD(Xi, yi, P) {thr = std::sqrt((2*ModelParams[0])/(1+2*ModelParams[2]));
Onep2lamda2 = 1+2*ModelParams[2]; lambda1 = ModelParams[1]; Xtr = P.Xtr; Iter = P.Iter; result.ModelParams = P.ModelParams; ScreenSize = P.ScreenSize; r = *P.r;}

FitResult CDL012::Fit() {

	bool SecondPass = false;

	objective = Objective(r, B);

	std::vector<unsigned int> FullOrder = Order;
	bool FirstRestrictedPass = true;
	if (ActiveSet){
		Order.resize(std::min((int) (B.n_nonzero + ScreenSize),(int)(p))); // std::min(1000,Order.size())
	}

	bool ActiveSetInitial = ActiveSet;


	for (unsigned int t=0; t<MaxIters; ++t){
		//std::cout<<"CDL012 "<< t << " " << objective << std::endl;
		Bprev = B;

		for (auto& i: Order){
			double cor = arma::dot(r,X->unsafe_col(i));

			(*Xtr)[i] = std::abs(cor); // do abs here instead from when sorting

			double Bi = B[i]; // B[i] is costly
			double x = cor + Bi; // x is beta_tilde_i
			double z = (std::abs(x) - lambda1)/Onep2lamda2;

			if (z >= thr){	// often false so body is not costly
				B[i] = std::copysign(z, x);

				r = r + X->unsafe_col(i)*(Bi - B[i]);
			}

			else if (Bi != 0) { // do nothing if x=0 and B[i] = 0
				r = r + X->unsafe_col(i)*Bi;
				B[i] = 0;
			}
		}

		//B.print();
		if (Converged()){
			result.IterNum = t+1;

			if(FirstRestrictedPass && ActiveSetInitial){
				FirstRestrictedPass = false;
				Order = FullOrder;
				Stabilized = false;
				ActiveSet = true;
			}

			else{
				if (Stabilized == true && ActiveSetInitial){ // && !SecondPass
					Order = OldOrder; // Recycle over all coordinates to make sure the achieved point is a CW-min.
					//SecondPass = true; // a 2nd pass will be performed
					Stabilized = false;
					ActiveSet = true;
				}

				else{
					break;
				}
			}

		}
		if (ActiveSet){SupportStabilized();}

	}

	result.Objective = objective;
	result.B = B;
	result.Model = this;
	result.r = r; // change to pointer later
	return result;
}

inline double CDL012::Objective(arma::vec & r, arma::sp_mat & B) { // hint inline
	auto l2norm = arma::norm(B,2);
	return 0.5*arma::dot(r,r) + ModelParams[0]*B.n_nonzero + ModelParams[1]*arma::norm(B,1) + ModelParams[2]*l2norm*l2norm;
}
