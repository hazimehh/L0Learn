#include "Grid1D.h"
#include "MakeCD.h"
#include <algorithm>

Grid1D::Grid1D(const arma::mat& Xi, const arma::vec& yi, const GridParams& PG){
	// automatically selects lambda_0 (but assumes other lambdas are given in PG.P.ModelParams)
	X = &Xi;
	y = &yi;
	p = Xi.n_cols;
	LambdaMinFactor = PG.LambdaMinFactor;
	P = PG.P;
	P.Xtr = new std::vector<double>(X->n_cols); // needed! careful
	Xtr = P.Xtr;
	G_ncols = PG.G_ncols;
	G.reserve(G_ncols);
	if (PG.LambdaU){Lambdas = PG.Lambdas;}
	/*
	else {
		Lambdas.reserve(G_ncols);
		Lambdas.push_back((0.5*arma::square(y->t() * *X)).max());
	}
	*/
	NnzStopNum = PG.NnzStopNum;
	Refine = PG.Refine;
	PartialSort = PG.PartialSort;
	XtrAvailable = PG.XtrAvailable;
	if (XtrAvailable){ytXmax2d = PG.ytXmax; Xtr = PG.Xtr;}
}

std::vector<FitResult*> Grid1D::Fit(){


	if (P.ModelType == "L0" || P.ModelType == "L012" || P.ModelType == "L012Swaps" || P.ModelType == "L012KSwaps" || P.ModelType == "L012Logistic"){
		bool scaledown = false;



		double Lipconst;
		arma::vec Xtrarma;
		if (P.ModelType == "L012Logistic"){
			if (!XtrAvailable){Xtrarma = 0.5*arma::abs(y->t() * *X).t();} // = gradient of logistic loss at zero}
			Lipconst = 0.25+2*P.ModelParams[2];
		}
		else{
			if (!XtrAvailable){Xtrarma = arma::abs(y->t() * *X).t();}
			Lipconst = 1+2*P.ModelParams[2];
		}

		double ytXmax;
		if (!XtrAvailable){
			*Xtr = arma::conv_to< std::vector<double> >::from(Xtrarma);
			ytXmax = arma::max(Xtrarma);
		}
		else{
			ytXmax = ytXmax2d;
		}

		double lambdamax = ((ytXmax - P.ModelParams[1])*(ytXmax - P.ModelParams[1]))/(2*(Lipconst));


		//std::cout<< "Lambda max: "<< lambdamax << std::endl;
		double lambdamin = lambdamax*LambdaMinFactor;
		Lambdas = arma::logspace(std::log10(lambdamin), std::log10(lambdamax), G_ncols);
		Lambdas = arma::flipud(Lambdas);


		//unsigned int StopNum = (X->n_rows < NnzStopNum) ? X->n_rows : NnzStopNum;
		P.Init = 'z'; //////////
		unsigned int StopNum = NnzStopNum;
		P.ModelParams[0] = Lambdas[0];
		//std::vector<double>* Xtr = P.Xtr;
		std::vector<unsigned int> idx(p);
		double Xrmax;
		bool prevskip = false; //previous grid point was skipped
		bool currentskip = false; // current grid point should be skipped
		for (unsigned int i=0; i<G_ncols; ++i){
			//std::cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! STARTED GRID ITER: "<<i<<std::endl;

			auto prevresult = G.back(); // prevresult is ptr to the prev result object


			currentskip = false;

			if (prevskip == false)
			{
			  	std::iota(idx.begin(), idx.end(), 0); // make global class var later
			  	if (PartialSort && p>5000)
			  		std::partial_sort(idx.begin(),idx.begin()+5000, idx.end(),[this](unsigned int i1, unsigned int i2) {return (*Xtr)[i1] > (*Xtr)[i2] ;});
			  	else
			  		std::sort(idx.begin(), idx.end(),[this](unsigned int i1, unsigned int i2) {return (*Xtr)[i1] > (*Xtr)[i2] ;});
			  	P.CyclingOrder = 'u';
			  	P.Uorder = idx; // can be made faster

					//
					Xrmax = (*Xtr)[idx[0]];

					if (i > 0)
					{
						std::vector<unsigned int> Sp;
						arma::sp_mat::const_iterator it;
						for(it = prevresult->B.begin(); it != prevresult->B.end(); ++it)
						{
							Sp.push_back(it.row());
						}

						for(unsigned int l=0; l<p;++l){
							if ( std::binary_search(Sp.begin(),Sp.end(),idx[l]) == false ){
								Xrmax = (*Xtr)[idx[l]];
								break;
							}
						}
					}
		  	}

			//std::cout<< "||X'r||_inf = "<<Xrmax<< std::endl;

			//if(i>=1){//std::cout<< "||X'r||_tru = "<<  arma::norm(prevresult->r.t() * *X,"inf") << std::endl;

			//Xrmax = arma::norm(prevresult->r.t() * *X,"inf");} // this is a bypass for the internal way of calc. Xrmax. can be safely turned off.
			//std::cout<< "||X'r||_inf = "<<Xrmax<< std::endl;




			// Following part assumes that lambda_0 has been set to the new value
			if(i>= 1 && !scaledown){
				P.ModelParams[0] = (((Xrmax - P.ModelParams[1])*(Xrmax - P.ModelParams[1]))/(2*(Lipconst)))*0.99; // for numerical stability issues.
				if (P.ModelParams[0] >= prevresult->ModelParams[0]){
					P.ModelParams[0] = prevresult->ModelParams[0]*0.97;
					//std::cout<<"INSTABILITY HANDELED"<<std::endl;
				} // handles numerical instability.
			}
			else{
				P.ModelParams[0] = std::min(P.ModelParams[0]*0.97, (((Xrmax - P.ModelParams[1])*(Xrmax - P.ModelParams[1]))/(2*(Lipconst)))*0.97 );
			}

			double thr = sqrt(2*P.ModelParams[0]*(Lipconst)) + P.ModelParams[1]; // pass this to class? we're calc this twice now

			if (P.Iter>0 && std::abs(Xrmax) < thr) // not needed anymore. Remove later.
			{ // Iternum>1 ensures that we have a good approximation to Xtr

				if (prevresult->IterNum>1 || prevskip == true){currentskip = true;} //// Rethink logic this is correct as

				else
				{
					arma::vec Xtra = (arma::abs(prevresult->r.t() * *X)).t();
					*Xtr = arma::conv_to< std::vector<double> >::from(Xtra);

					//sort again
				  	std::iota(idx.begin(), idx.end(), 0); // make global class var later

				  	//std::partial_sort(idx.begin(),idx.begin()+100, idx.end(),[this](unsigned int i1, unsigned int i2) {return (*Xtr)[i1] > (*Xtr)[i2] ;});
				  	std::sort(idx.begin(), idx.end(),[this](unsigned int i1, unsigned int i2) {return (*Xtr)[i1] > (*Xtr)[i2] ;}); /////////////////////////////////////////////////////

				  	P.CyclingOrder = 'u';
				  	P.Uorder = idx; // can be made faster
			  		Xrmax = (*Xtr)[idx[0]];

				  	if (std::fabs(Xrmax) < thr){currentskip = true;}

				}

			}

			//if(currentskip == true){ // Debugging remove later
			//	std::cout<<"!!!!Skipped!!!"<<std::endl; // nothing will be pushed back to G, which is fine from MSE perspective.
			//}

			if(currentskip == false)
			{

				auto Model = make_CD(*X, *y, P);
				FitResult * result = new FitResult; // Later: Double check memory leaks..

				*result = Model->Fit();

				if (i>=1 && arma::norm(result->B-(G.back())->B,"inf")<10e-5){scaledown = true;} // got same solution
				else {scaledown = false;}

				G.push_back(result);
				//std::cout<<"### ### ###"<<std::endl;
				//std::cout<<"Iteration: "<<i<<". "<<"Nnz: "<< result->B.n_nonzero << ". Lambda: "<<P.ModelParams[0]<< std::endl;
				if(result->B.n_nonzero > StopNum) {break;}
				//result->B.t().print();
				P.InitialSol = &(result->B);
			}

			//std::cout<<"Lambda0, Lambda1, Lambda2: "<<P.ModelParams[0]<<", "<<P.ModelParams[1]<<", "<<P.ModelParams[2]<<std::endl;

			// Prepare P for next iteration
			//P.ModelParams[0] = Lambdas[i+1];
			//P.ModelParams[0] = ((Xrmax - P.ModelParams[1])*(Xrmax - P.ModelParams[1]))/(2*(1+2*P.ModelParams[2]));

			P.Init = 'u';
			P.Iter += 1;
			prevskip = currentskip;
		}
	}


	else if (P.ModelType == "L1" || P.ModelType == "L1Relaxed"){


		*Xtr = arma::conv_to< std::vector<double> >::from(arma::abs(y->t() * *X).t()); // ToDO: double computation, handle later
		std::vector<unsigned int> idx(p);
		// Derive lambda_max
		double lambdamax  = arma::norm(y->t() * *X, "inf");

		//std::cout<< "Lambda max: "<< lambdamax << std::endl;
		double lambdamin = lambdamax*LambdaMinFactor;
		Lambdas = arma::logspace(std::log10(lambdamin), std::log10(lambdamax), G_ncols);
		Lambdas = arma::flipud(Lambdas);


		//unsigned int StopNum = (X->n_rows < NnzStopNum) ? X->n_rows : NnzStopNum;
		P.Init = 'z'; //////////
		unsigned int StopNum = NnzStopNum;

		for (unsigned int i=0; i<G_ncols; ++i){


		  	std::iota(idx.begin(), idx.end(), 0); // make global class var later
		  	//std::partial_sort(idx.begin(),idx.begin()+100, idx.end(),[this](unsigned int i1, unsigned int i2) {return (*Xtr)[i1] > (*Xtr)[i2] ;});
		  	std::sort(idx.begin(), idx.end(),[this](unsigned int i1, unsigned int i2) {return (*Xtr)[i1] > (*Xtr)[i2] ;}); /////////////////////////////////////////////////////
		  	P.CyclingOrder = 'u';
		  	P.Uorder = idx; // can be made faster

			// Following part assumes that lambda_0 has been set to the new value

			P.ModelParams[0] = Lambdas[i];



			auto Model = make_CD(*X, *y, P);

			FitResult * result = new FitResult; // Later: Double check memory leaks..

			*result = Model->Fit();

			G.push_back(result);
			//std::cout<<"### ### ###"<<std::endl;
			//std::cout<<"Iteration: "<<i<<". "<<"Nnz: "<< result->B.n_nonzero << ". Lambda: "<<P.ModelParams[0]<< std::endl;
			if(result->B.n_nonzero > StopNum) {break;}
			//result->B.t().print();
			P.InitialSol = &(result->B);

			//std::cout<<"Lambda0, Lambda1, Lambda2: "<<P.ModelParams[0]<<", "<<P.ModelParams[1]<<", "<<P.ModelParams[2]<<std::endl;

			P.Init = 'u';
			P.Iter += 1;
		}
	}





	/*
	else{

		arma::mat Corr(NnzStopNum, X->n_cols);
		std::map<unsigned int,unsigned int> I;
		arma::rowvec ytX = y->t() * *X;

		//unsigned int StopNum = (X->n_rows < NnzStopNum) ? X->n_rows : NnzStopNum;
		unsigned int StopNum = NnzStopNum;
		P.ModelParams[0] = Lambdas[0];

		for (unsigned int i=0; i<G_ncols; ++i){

			auto Model = CDL0SwapsGrid(*X, *y, P); // CdL0SwapsGrid maintains Corr and I.

			FitResult * result = new FitResult; // Later: Double check memory leaks..
			*result = Model.Fit(Corr, I, ytX);

			G.push_back(result);

			std::cout<<"Iteration: "<<i<<". "<<"Nnz: "<< result->B.n_nonzero << std::endl;
			if(result->B.n_nonzero > StopNum) {break;}

			// Prepare P for next iteration
			P.ModelParams[0] = Lambdas[i+1];
			P.Init = 'u';
			P.InitialSol = &(result->B);
		}
	}
*/

	if (Refine == true){
		for (unsigned int i = 0;i<20;++i)
		{
			bool better = false;
			//std::cout<<"!!!!!!!!!!!!! Backward-Forward Iteration: "<<i<<std::endl;
			for (auto it = G.rbegin()+1; it != G.rend(); ++it)
				{
					P.InitialSol = &((*(it-1))->B);
					P.ModelParams[0] = (*it)->ModelParams[0];
					//std::cout<<"### ModelParams[0]: "<<P.ModelParams[0]<<std::endl;

					auto Model = make_CD(*X, *y, P);

					FitResult * result = new FitResult; // Later: Double check memory leaks..

					*result = Model->Fit();



					if (result->Objective < (*it)->Objective){
						*it = result;
						better = true;
						//std::cout<<"Found better in reverse grid"<<std::endl;
					}


				}

			for (auto it = G.begin()+1; it != G.end(); ++it)
				{
					P.InitialSol = &((*(it-1))->B);
					P.ModelParams[0] = (*it)->ModelParams[0];
					//std::cout<<"### ModelParams[0]: "<<P.ModelParams[0]<<std::endl;

					auto Model = make_CD(*X, *y, P);

					FitResult * result = new FitResult; // Later: Double check memory leaks..

					*result = Model->Fit();

					if (result->Objective < (*it)->Objective){
						*it = result;
						better = true;
						//std::cout<<"Found better in forward grid"<<std::endl;
					}

				}
			if (better == false){break;} //fixed grid.
		}
	}

	return G;
}
