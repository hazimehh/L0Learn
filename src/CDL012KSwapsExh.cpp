#include "CDL012KSwapsExh.h"
#include "CDL012Cons.h"
#include "CDL012Swaps.h"


CDL012KSwapsExh::CDL012KSwapsExh(const arma::mat& Xi, const arma::vec& yi, const Params& Pi) : CD(Xi, yi, Pi) {MaxNumSwaps = Pi.MaxNumSwaps; P = Pi; K = ModelParams[3];}


template <typename Iterator>
bool next_combination(const Iterator first, Iterator k, const Iterator last)
{
   /* Credits: Mark Nelson http://marknelson.us */
   if ((first == last) || (first == k) || (last == k))
      return false;
   Iterator i1 = first;
   Iterator i2 = last;
   ++i1;
   if (last == i1)
      return false;
   i1 = last;
   --i1;
   i1 = k;
   --i2;
   while (first != i1)
   {
      if (*--i1 < *i2)
      {
         Iterator j = k;
         while (!(*i1 < *j)) ++j;
         std::iter_swap(i1,j);
         ++i1;
         ++j;
         i2 = k;
         std::rotate(i1,j,last);
         while (last != j)
         {
            ++j;
            ++i2;
         }
         std::rotate(k,i2,last);
         return true;
      }
   }
   std::rotate(first,k,last);
   return false;
}



FitResult CDL012KSwapsExh::Fit() {

	bool Greedy = true;
	FitResult result;

	auto oneSwapsresult = CDL012Swaps(*X, *y, P).Fit();
	double oneSwapsObj = oneSwapsresult.Objective; 

	FitResult BestAbsResult = oneSwapsresult;
	double BestAbsObj = oneSwapsObj;
	std::vector<std::vector<uint>> BestSubsetsRemoved;

	for(uint k=0; k<MaxNumSwaps; ++k)
	{
	
		result = CDL012Swaps(*X, *y, P).Fit(); // result will be the same unless a better sol is found

		if(result.Objective < BestAbsObj){
			BestAbsResult.B = result.B;
			BestAbsResult.r = result.r;
			BestAbsResult.Objective = result.Objective;
			BestAbsObj = result.Objective;
		}

		B = result.B;   // will note be changed by the following code
		if(B.n_nonzero<=1){break;}
		r = *y - *X * B; // will note be changed by the following code
		objective = result.Objective;  // will note be changed by the following code

		//std::cout<<"KSwaps Iteration: "<<k<<". "<<"Obj: "<<objective<<std::endl;

		P.Init = 'u';

		std::vector<uint> S; 
		arma::sp_mat::const_iterator start = B.begin();
		arma::sp_mat::const_iterator end   = B.end();
		for(arma::sp_mat::const_iterator it = start; it != end; ++it){S.push_back(it.row());}
			
		std::vector<uint> rangetemp(p);
		std::iota(std::begin(rangetemp), std::end(rangetemp), 0);
		std::vector<uint> Sc;
		std::set_difference(rangetemp.begin(), rangetemp.end(), S.begin(), S.end(), std::inserter(Sc, Sc.end()));
		arma::uvec Scind = arma::conv_to< arma::uvec >::from(Sc);
		arma::mat XSc = X->submat(arma::regspace<arma::uvec>(0,n-1), Scind);


		
		std::vector<std::vector<uint>> combinations;
		for(uint i=1; i<=K; ++i){ //////////////////
			if (S.size() >= i)
			{
			    do
			    {
			    	combinations.push_back(std::vector<uint>(S.begin(),S.begin()+i));
			    } while (S.begin() + i <= S.end() && next_combination(S.begin(), S.begin() + i, S.end()));
			}

		}
		
		

		/*
		//Do the clustering on XS
		arma::uvec Sind = arma::conv_to< arma::uvec >::from(S);
		arma::mat XS = X->submat(arma::regspace<arma::uvec>(0,n-1), Sind);

		uint clusters = 10;
		arma::Row<size_t> assignments;
		// Initialize with the default arguments.
		mlpack::kmeans::KMeans<> Cluster;
		Cluster.Cluster(XS.t(), clusters, assignments);

		//for(auto& a: assignments){std::cout<<a<<std::endl;}

		// Given assignments vector generate Sis
		std::vector<std::vector<uint>> Sis(clusters);
		for (uint i=0; i<S.size(); ++i){
			Sis[assignments[i]].push_back(S[i]);
		}

		// Given a vector of Sis, add the comb
		std::vector<std::vector<uint>> combinations;
		for(auto& Si: Sis){
			for(uint i=1; i<=K; ++i){ //////////////////
				if(Si.size()>=i){
				    do
				    {
				    	combinations.push_back(std::vector<uint>(Si.begin(),Si.begin()+i));
				    } while (next_combination(Si.begin(), Si.begin() + i, Si.end()));
				}

			}
		}
		

		//
		*/

		Params Pnew = P;
		bool foundbetter = false;
		double best_objective = objective; ///////////////////////////////////////*1.05
		std::vector<uint> best_added;
		std::vector<uint> best_removed;


		for (uint t=0; t<combinations.size(); ++t){
			if(std::find(BestSubsetsRemoved.begin(), BestSubsetsRemoved.end(), combinations[t]) != BestSubsetsRemoved.end()){
				//std::cout<<"SKIPPED!"<<std::endl;
				continue;

			}
			//std::cout<<"Combination Removed: ";

			arma::vec rtemp = r;
			arma::sp_mat Btemp = B;
			for(auto& i: combinations[t]){
				rtemp = rtemp + X->unsafe_col(i)*Btemp[i];
				Btemp[i] = 0;
				//std::cout<<i<<" ";
			}



			arma::sp_mat BSc(Sc.size(),1);
			Pnew.InitialSol = &BSc; // check whether this creates problems for the grid.

			
			//Pnew.ModelParams[3] = 2;

			auto ResultTemp = CDL012Cons(XSc, rtemp, Pnew).Fit(); /////////////


			std::vector<uint> added;
			arma::sp_mat::const_iterator start = ResultTemp.B.begin();
			arma::sp_mat::const_iterator end   = ResultTemp.B.end();
			for(arma::sp_mat::const_iterator it = start; it != end; ++it){
				Btemp[Sc[it.row()]] = ResultTemp.B[it.row()];
				rtemp = rtemp - XSc.unsafe_col(it.row()) * ResultTemp.B[it.row()];
				added.push_back(Sc[it.row()]);
			}

			// Now Btemp and rtemp contain the updated solution.

			double current_obj = Objective(rtemp, Btemp);
			//std::cout<<" Obj: "<<current_obj<<std::endl;
			if(current_obj < best_objective){
				//std::cout<<"!!!!!!!!!BLA: "<<current_obj<<" "<<best_objective<<std::endl;
				best_objective = current_obj;
				best_added = added;
				best_removed = combinations[t];
				foundbetter = true;
				
				result.B = Btemp; ///////////// temporarily (*)
				
				//result.r = rtemp;
				
				//result.Objective = current_obj;



				if(!Greedy){break;}
			}
		
		// internal for ends after this
		}

		if (foundbetter){
			//std::cout<<"!!!!!!!!!!!!!!!!!!!!! Found Better! Obj: " <<best_objective<<std::endl;
			//std::cout<<" Removed: ";
			//for(auto& i: best_removed){std::cout<<i<<" ";}
			//std::cout<<". Added: ";
			//for(auto& i: best_added){std::cout<<i<<" ";}
			//std::cout<<std::endl;
			BestSubsetsRemoved.push_back(best_removed);

		}

		else{
			//std::cout<<"Did not find better :(" << std::endl;
			break;
		}

		// found better
		P.InitialSol = &result.B; /// (*) sets the sol to the swapped version
		//if (result.Objective < oneSwapsObj){break;}
	// for ends after this
	}

	result.Model = this;


	//if (result.Objective > oneSwapsObj){result = oneSwapsresult;}

	//std::cout<<"FINAL OBJ: "<<BestAbsResult.Objective<<std::endl;
	return BestAbsResult;
	
}

inline double CDL012KSwapsExh::Objective(arma::vec & r, arma::sp_mat & B) { // hint inline
	auto l2norm = arma::norm(B,2);
	return 0.5*arma::dot(r,r) + ModelParams[0]*B.n_nonzero + ModelParams[1]*arma::norm(B,1) + ModelParams[2]*l2norm*l2norm; 
}

