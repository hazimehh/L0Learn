#include "CDL012Cons.h"
#include <set>


CDL012Cons::CDL012Cons(const arma::mat& Xi, const arma::vec& yi, const Params& P) : CD(Xi, yi, P) {thr = sqrt((2*ModelParams[0])/(1+2*ModelParams[2])); 
	 //k = ModelParams[3]; //CAREFUL!!!!!!!!!!!!!!!!
	 Onep2lamda2 = 1+2*ModelParams[2];  Iter = P.Iter; result.ModelParams = P.ModelParams;}

FitResult CDL012Cons::Fit() {

		objective = Objective(r, B);

		std::set<unsigned int> S; 
		arma::sp_mat::const_iterator start = B.begin();
		arma::sp_mat::const_iterator end   = B.end();
		for(arma::sp_mat::const_iterator it = start; it != end; ++it){S.insert(it.row());}
			
		std::vector<unsigned int> rangetemp(p);
		std::iota(std::begin(rangetemp), std::end(rangetemp), 0);
		std::set<unsigned int> Sc;
		std::set_difference(rangetemp.begin(), rangetemp.end(), S.begin(), S.end(), std::inserter(Sc, Sc.end()));


	for (unsigned int t=0; t<MaxIters; ++t){
		//std::cout<<"CD Cons Iter: "<< t << " " << objective<< " " <<B.n_nonzero<< std::endl;
		//std::cout<<"S len: "<< S.size() <<" Sc len: "<< Sc.size() <<std::endl;


		if (B.n_nonzero < k){
			unsigned int best_index=-1;
			double best_value;
			// Find ||r'X||_inf
			arma::rowvec rtX = r.t() * *X;
			double rtXmax = -1;
			unsigned int rtXmaxind = 0;
			for (auto& i: Sc){if (std::abs(rtX[i]) > rtXmax) {rtXmax = std::abs(rtX[i]); rtXmaxind = i;}}
			double Beta_jabs =  (std::abs(rtX[rtXmaxind]) - ModelParams[1])/Onep2lamda2; ///////// still wrong because rtXmax has an abs...
			arma::sp_mat Btemp = B; 

			double best_obj = objective; // stays the same if < thr 

			if (Beta_jabs >= thr){
				Btemp[rtXmaxind] = std::copysign(Beta_jabs,rtX[rtXmaxind]);
				best_index=rtXmaxind;
				best_value = Btemp[rtXmaxind];
				arma::vec rtemp = r - X->unsafe_col(rtXmaxind)*Btemp[rtXmaxind];
				best_obj = Objective(rtemp, Btemp);
			}
			//std::cout<<"Best_obj Sc: "<<best_obj<<std::endl;

			for(auto &i: S){
				Beta_jabs =  (std::abs(rtX[i] + B[i]) - ModelParams[1])/Onep2lamda2;
				Btemp = B;
				arma::vec rtemp;
				if (Beta_jabs >= thr){Btemp[i] = std::copysign(Beta_jabs, rtX[i] + B[i]); rtemp = r + X->unsafe_col(i)*(B[i] - Btemp[i]);}
				else {Btemp[i] = 0; rtemp = r + X->unsafe_col(i)*B[i];}
				double current_objective = Objective(rtemp,Btemp);
				if (current_objective < best_obj){best_obj = current_objective; best_index = i; best_value = Btemp[i];}
			}

			//std::cout<<"Best_obj "<<best_obj<<std::endl;

			if (best_index != -1){
				//std::cout<<"Updating B and r"<<std::endl;
				//std::cout<<"Best_index: "<<best_index<<" Best_value= "<<best_value<<std::endl;
				r = r + X->unsafe_col(best_index)*(B[best_index] - best_value);
				if (best_value != 0 && B[best_index] == 0){S.insert(best_index); Sc.erase(best_index);}
				else if(best_value == 0) {S.erase(best_index); Sc.insert(best_index);}
				B[best_index] = best_value;
			}

		}

		else{

			unsigned int best_index=-1;
			unsigned int best_z = -1;
			double best_value;
			double best_obj = objective; // stays the same if < thr 

			for (auto& z: S){

				// Find ||r'X||_inf
				arma::vec rz = r+X->unsafe_col(z)*B[z];
				arma::rowvec rtX = rz.t() * *X;
				double rtXmax = -1;
				unsigned int rtXmaxind = 0;
				for (auto& i: Sc){if (std::abs(rtX[i]) > rtXmax) {rtXmax = std::abs(rtX[i]); rtXmaxind = i;}}
				double Beta_jabs =  (std::abs(rtX[rtXmaxind]) - ModelParams[1])/Onep2lamda2;
				arma::sp_mat Btemp = B; 


				Btemp[rtXmaxind] = std::copysign(Beta_jabs,rtX[rtXmaxind]);

				arma::vec rtemp = rz - X->unsafe_col(rtXmaxind)*Btemp[rtXmaxind];

				double current_objective = Objective(rtemp, Btemp);

				if (Beta_jabs >= thr && current_objective < best_obj){
					best_index=rtXmaxind;
					best_z = z;
					best_value = Btemp[rtXmaxind];
					best_obj = current_objective;
				}

				//std::cout<<"Best_obj until Sc"<<best_obj<<std::endl;

				for(auto &i: S){
					if (i != z){Beta_jabs =  (std::abs(rtX[i] + B[i]) - ModelParams[1])/Onep2lamda2;}
					else {Beta_jabs =  (std::abs(rtX[i]) - ModelParams[1])/Onep2lamda2;}
					Btemp = B;
					arma::vec rtemp = rz; // remove z;
					Btemp[z] = 0;
					if (i!=z && Beta_jabs >= thr){Btemp[i] = std::copysign(Beta_jabs, rtX[i] + B[i]); rtemp = rtemp + X->unsafe_col(i)*(B[i] - Btemp[i]);}
					else if (i==z && Beta_jabs >= thr){Btemp[i] = std::copysign(Beta_jabs, rtX[i]); rtemp = rtemp - X->unsafe_col(i)*(Btemp[i]);} // already removed prev. effect
					else if(i!=z) {Btemp[i] = 0; rtemp = r + X->unsafe_col(i)*B[i];}
					double current_objective = Objective(rtemp,Btemp);
					if (current_objective < best_obj){best_obj = current_objective; best_index = i; best_z = z; best_value = Btemp[i];}
				}

				//std::cout<<"Best_obj "<<best_obj<<std::endl;

			}


			if (best_index != -1){
				//std::cout<<"Updating B and r"<<std::endl;
				//std::cout<<"Best_index: "<<best_index<<" Best_value= "<<best_value<<" Best_z"<<best_z<<std::endl;

				if(best_index != best_z){ // z should be kicked out
					r = r + X->unsafe_col(best_z)*(B[best_z]); /////////////////////
					r = r + X->unsafe_col(best_index)*(B[best_index] - best_value);
					S.erase(best_z); Sc.insert(best_z);
					if (best_value != 0 && B[best_index] == 0){S.insert(best_index); Sc.erase(best_index);}
					else if(best_value == 0) {S.erase(best_index); Sc.insert(best_index);}
				}
				else{
					r = r + X->unsafe_col(best_index)*(B[best_z] - best_value);
					if(best_value == 0) {S.erase(best_index); Sc.insert(best_index);}
					// else nothing
				}
				B[best_z] = 0;
				B[best_index] = best_value;
			}



		}


		if (Converged()){
			result.IterNum = t+1; 
			//std::cout<<"Converged in "<<t+1<<" iterations."<<std::endl;
			break;
		}

	}

	result.Objective = objective;
	result.B = B;
	result.Model = this;
	result.r = r; // change to pointer later
	return result;
}

inline double CDL012Cons::Objective(arma::vec & r, arma::sp_mat & B) { // hint inline
	auto l2norm = arma::norm(B,2);
	return 0.5*arma::dot(r,r) + ModelParams[0]*B.n_nonzero + ModelParams[1]*arma::norm(B,1) + ModelParams[2]*l2norm*l2norm; 
}