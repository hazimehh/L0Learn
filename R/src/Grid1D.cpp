#include "Grid1D.hpp"

template <class T>
Grid1D<T>::Grid1D(const T& Xi, const arma::vec& yi, const GridParams<T>& PG) {
    // automatically selects lambda_0 (but assumes other lambdas are given in PG.P.ModelParams)
    
    X = &Xi;
    y = &yi;
    p = Xi.n_cols;
    LambdaMinFactor = PG.LambdaMinFactor;
    ScaleDownFactor = PG.ScaleDownFactor;
    P = PG.P;
    P.Xtr = new std::vector<double>(X->n_cols); // needed! careful
    P.ytX = new arma::rowvec(X->n_cols);
    P.D = new std::map<std::size_t, arma::rowvec>();
    P.r = new arma::vec(Xi.n_rows);
    Xtr = P.Xtr;
    ytX = P.ytX;
    NoSelectK = P.NoSelectK;
    
    LambdaU = PG.LambdaU;
    
    if (!LambdaU) {
        G_ncols = PG.G_ncols;
    } else {
        G_ncols = PG.Lambdas.n_rows; // override the user's ncols if LambdaU = 1
    }
    
    G.reserve(G_ncols);
    if (LambdaU) {
        Lambdas = PG.Lambdas;
    } // user-defined lambda0 grid
    /*
     else {
     Lambdas.reserve(G_ncols);
     Lambdas.push_back((0.5*arma::square(y->t() * *X)).max());
     }
     */
    NnzStopNum = PG.NnzStopNum;
    PartialSort = PG.PartialSort;
    XtrAvailable = PG.XtrAvailable;
    if (XtrAvailable) {
        ytXmax2d = PG.ytXmax;
        Xtr = PG.Xtr;
    }
}

template <class T>
Grid1D<T>::~Grid1D() {
    // delete all dynamically allocated memory
    delete P.Xtr;
    delete P.ytX;
    delete P.D;
    delete P.r;
}


template <class T>
std::vector<std::unique_ptr<FitResult<T>>> Grid1D<T>::Fit() {
    
    if (P.Specs.L0 || P.Specs.L0L2 || P.Specs.L0L1) {
        bool scaledown = false;
        
        double Lipconst;
        arma::vec Xtrarma;
        if (P.Specs.Logistic) {
            if (!XtrAvailable) {
                Xtrarma = 0.5 * arma::abs(y->t() * *X).t();
            } // = gradient of logistic loss at zero}
            Lipconst = 0.25 + 2 * P.ModelParams[2];
        } else if (P.Specs.SquaredHinge) {
            if (!XtrAvailable) {
                // gradient of loss function at zero
                Xtrarma = 2 * arma::abs(y->t() * *X).t();
            } 
            Lipconst = 2 + 2 * P.ModelParams[2];
        } else {
            if (!XtrAvailable) {
                *ytX =  y->t() * *X;
                Xtrarma = arma::abs(*ytX).t(); // Least squares
            }
            Lipconst = 1 + 2 * P.ModelParams[2];
            *P.r = *y - P.b0; // B = 0 initially
        }
        
        double ytXmax;
        if (!XtrAvailable) {
            *Xtr = arma::conv_to< std::vector<double> >::from(Xtrarma);
            ytXmax = arma::max(Xtrarma);
        } else {
            ytXmax = ytXmax2d;
        }
        
        double lambdamax = ((ytXmax - P.ModelParams[1]) * (ytXmax - P.ModelParams[1])) / (2 * (Lipconst));
        
        // Rcpp::Rcout << "lambdamax: " << lambdamax << "\n";
        
        if (!LambdaU) {
            P.ModelParams[0] = lambdamax;
        } else {
            P.ModelParams[0] = Lambdas[0];
        }
        
        // Rcpp::Rcout << "P ModelParams: {" << P.ModelParams[0] << ", " << P.ModelParams[1] << ", " << P.ModelParams[2] << ", " << P.ModelParams[3] <<   "}\n";
        
        P.Init = 'z';
        
        
        //std::cout<< "Lambda max: "<< lambdamax << std::endl;
        //double lambdamin = lambdamax*LambdaMinFactor;
        //Lambdas = arma::logspace(std::log10(lambdamin), std::log10(lambdamax), G_ncols);
        //Lambdas = arma::flipud(Lambdas);
        
        
        //std::size_t StopNum = (X->n_rows < NnzStopNum) ? X->n_rows : NnzStopNum;
        std::size_t StopNum = NnzStopNum;
        //std::vector<double>* Xtr = P.Xtr;
        std::vector<std::size_t> idx(p);
        double Xrmax;
        bool prevskip = false; //previous grid point was skipped
        bool currentskip = false; // current grid point should be skipped
        
        for (std::size_t i = 0; i < G_ncols; ++i) {
            // Rcpp::checkUserInterrupt();
            // Rcpp::Rcout << "Grid1D: " << i << "\n";
            FitResult<T> *  prevresult = new FitResult<T>; // prevresult is ptr to the prev result object
            //std::unique_ptr<FitResult> prevresult;
            if (i > 0) {
                //prevresult = std::move(G.back());
                *prevresult = *(G.back());
                
            }
      
            currentskip = false;
            
            if (!prevskip) {
                
                std::iota(idx.begin(), idx.end(), 0); // make global class var later
                // Exclude the first NoSelectK features from sorting.
                if (PartialSort && p > 5000 + NoSelectK)
                    std::partial_sort(idx.begin() + NoSelectK, idx.begin() + 5000 + NoSelectK, idx.end(), [this](std::size_t i1, std::size_t i2) {return (*Xtr)[i1] > (*Xtr)[i2] ;});
                else
                    std::sort(idx.begin() + NoSelectK, idx.end(), [this](std::size_t i1, std::size_t i2) {return (*Xtr)[i1] > (*Xtr)[i2] ;});
                P.CyclingOrder = 'u';
                P.Uorder = idx; // can be made faster
                
                //
                Xrmax = (*Xtr)[idx[NoSelectK]];
                
                if (i > 0) {
                    std::vector<std::size_t> Sp = nnzIndicies(prevresult->B);
                    
                    for(std::size_t l = NoSelectK; l < p; ++l) {
                        if ( std::binary_search(Sp.begin(), Sp.end(), idx[l]) == false ) {
                            Xrmax = (*Xtr)[idx[l]];
                            //std::cout<<"Grid Iteration: "<<i<<" Xrmax= "<<Xrmax<<std::endl;
                            break;
                        }
                    }
                }
            }
            
            // Following part assumes that lambda_0 has been set to the new value
            if(i >= 1 && !scaledown && !LambdaU) {
                P.ModelParams[0] = (((Xrmax - P.ModelParams[1]) * (Xrmax - P.ModelParams[1])) / (2 * (Lipconst))) * 0.99; // for numerical stability issues.
                
                if (P.ModelParams[0] >= prevresult->ModelParams[0]) {
                    P.ModelParams[0] = prevresult->ModelParams[0] * 0.97;
                } // handles numerical instability.
            } else if (i >= 1 && !LambdaU) {
                P.ModelParams[0] = std::min(P.ModelParams[0] * ScaleDownFactor, (((Xrmax - P.ModelParams[1]) * (Xrmax - P.ModelParams[1])) / (2 * (Lipconst))) * 0.97 );
                // add 0.9 as an R param
            } else if (i >= 1 && LambdaU) {
                P.ModelParams[0] = Lambdas[i];
            }
            
            // Rcpp::Rcout << "P.ModelParams[0]: " << P.ModelParams[0] << "\n";
           
            if (!currentskip) {
            
                auto Model = make_CD(*X, *y, P);
                
                std::unique_ptr<FitResult<T>> result(new FitResult<T>);
                *result = Model->Fit();
    
                delete Model;
                
                scaledown = false;
                if (i >= 1) {
                    std::vector<std::size_t> Spold = nnzIndicies(prevresult->B);
      
                    std::vector<std::size_t> Spnew = nnzIndicies(result->B);
                   
                    bool samesupp = false;
                    
                    if (Spold == Spnew) {
                        samesupp = true;
                        scaledown = true;
                    }
                    
                    // //
                    // 
                    // if (samesupp) {
                    //     scaledown = true;
                    // } // got same solution
                }
                
                //else {scaledown = false;}
                G.push_back(std::move(result));
                
                
                if(n_nonzero(G.back()->B) >= StopNum) {
                    break;
                }
                //result->B.t().print();
                P.InitialSol = &(G.back()->B);
                P.b0 = G.back()->b0;
                // Udate: After 1.1.0, P.r is automatically updated by the previous call to CD
                //*P.r = G.back()->r;
                
            }
            
            delete prevresult;
            
            
            P.Init = 'u';
            P.Iter += 1;
            prevskip = currentskip;
        }
    }
    
    return std::move(G);
}


template class Grid1D<arma::mat>;
template class Grid1D<arma::sp_mat>;
