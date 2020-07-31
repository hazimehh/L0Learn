#ifndef GRID1D_H
#define GRID1D_H
#include <memory>
#include <algorithm>
#include <map>
#include "RcppArmadillo.h"
#include "Params.h"
#include "GridParams.h"
#include "FitResult.h"
#include "MakeCD.h"

template <typename T>
class Grid1D
{
    private:
        unsigned int G_ncols;
        Params<T> P;
        const T * X;
        const arma::vec * y;
        unsigned int p;
        //std::vector<FitResult*> G;
        std::vector<std::unique_ptr<FitResult<T>>> G;
        arma::vec Lambdas;
        bool LambdaU;
        unsigned int NnzStopNum;
        std::vector<double> * Xtr;
        arma::rowvec * ytX;
        double LambdaMinFactor;
        bool Refine;
        bool PartialSort;
        bool XtrAvailable;
        double ytXmax2d;
        double ScaleDownFactor;
        unsigned int NoSelectK;

    public:
        Grid1D(const T& Xi, const arma::vec& yi, const GridParams<T>& PG);
        ~Grid1D();
        std::vector<std::unique_ptr<FitResult<T>>> Fit();
};

template <typename T>
Grid1D<T>::Grid1D(const T& Xi, const arma::vec& yi, const GridParams<T>& PG)
{
    // automatically selects lambda_0 (but assumes other lambdas are given in PG.P.ModelParams)
    X = &Xi;
    y = &yi;
    p = Xi.n_cols;
    LambdaMinFactor = PG.LambdaMinFactor;
    ScaleDownFactor = PG.ScaleDownFactor;
    P = PG.P;
    P.Xtr = new std::vector<double>(X->n_cols); // needed! careful
    P.ytX = new arma::rowvec(X->n_cols);
    P.D = new std::map<unsigned int, arma::rowvec>();
    P.r = new arma::vec(Xi.n_rows);
    Xtr = P.Xtr;
    ytX = P.ytX;
    NoSelectK = P.NoSelectK;
    
    LambdaU = PG.LambdaU;
    
    if (!LambdaU)
    {
        G_ncols = PG.G_ncols;
    }
    else
    {
        G_ncols = PG.Lambdas.n_rows; // override the user's ncols if LambdaU = 1
    }
    
    G.reserve(G_ncols);
    if (LambdaU) {Lambdas = PG.Lambdas;} // user-defined lambda0 grid
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
    if (XtrAvailable) {ytXmax2d = PG.ytXmax; Xtr = PG.Xtr;}
}

template <typename T>
Grid1D<T>::~Grid1D()
{
    // delete all dynamically allocated memory
    delete P.Xtr;
    delete P.ytX;
    delete P.D;
    delete P.r;
}


template <typename T>
std::vector<std::unique_ptr<FitResult<T>>> Grid1D<T>::Fit()
{
    if (P.Specs.L0 || P.Specs.L0L2 || P.Specs.L0L1)
    {
        bool scaledown = false;
        
        double Lipconst;
        arma::vec Xtrarma;
        if (P.Specs.Logistic)
        {
            if (!XtrAvailable) {Xtrarma = 0.5 * arma::abs(y->t() * *X).t();} // = gradient of logistic loss at zero}
            Lipconst = 0.25 + 2 * P.ModelParams[2];
        }
        else if (P.Specs.SquaredHinge)
        {
            
            /*
             Params Ptemp = P;
             Ptemp.CyclingOrder = 'u';
             Ptemp.Uorder = std::vector<unsigned int>();
             
             auto Model = make_CD(*X, *y, Ptemp);
             FitResult * result = new FitResult;
             *result = Model->Fit();
             
             std::cout<<"!!!!!!! Intercept: "<<result->intercept<<std::endl;
             if (!XtrAvailable){Xtrarma = 2*arma::abs( (y->t() - result->intercept) * *X).t();} // = gradient of loss function at zero}
             */
            
            
            if (!XtrAvailable) {Xtrarma = 2 * arma::abs(y->t() * *X).t();} // = gradient of loss function at zero}
            Lipconst = 2 + 2 * P.ModelParams[2];
        }
        
        else  // SquaredError
        {
            if (!XtrAvailable)
            {
                *ytX =  y->t() * *X;
                Xtrarma = arma::abs(*ytX).t(); // Least squares
            }
            Lipconst = 1 + 2 * P.ModelParams[2];
            *P.r = *y; // B = 0 initially
        }
        
        double ytXmax;
        if (!XtrAvailable)
        {
            *Xtr = arma::conv_to< std::vector<double> >::from(Xtrarma);
            ytXmax = arma::max(Xtrarma);
        }
        else
        {
            ytXmax = ytXmax2d;
        }
        
        double lambdamax = ((ytXmax - P.ModelParams[1]) * (ytXmax - P.ModelParams[1])) / (2 * (Lipconst));
        if (!LambdaU)
        {
            P.ModelParams[0] = lambdamax;
        }
        else
        {
            P.ModelParams[0] = Lambdas[0];
        }
        
        P.Init = 'z';
        
        
        //std::cout<< "Lambda max: "<< lambdamax << std::endl;
        //double lambdamin = lambdamax*LambdaMinFactor;
        //Lambdas = arma::logspace(std::log10(lambdamin), std::log10(lambdamax), G_ncols);
        //Lambdas = arma::flipud(Lambdas);
        
        
        //unsigned int StopNum = (X->n_rows < NnzStopNum) ? X->n_rows : NnzStopNum;
        unsigned int StopNum = NnzStopNum;
        //std::vector<double>* Xtr = P.Xtr;
        std::vector<unsigned int> idx(p);
        double Xrmax;
        bool prevskip = false; //previous grid point was skipped
        bool currentskip = false; // current grid point should be skipped
        for (unsigned int i = 0; i < G_ncols; ++i)
        {
            //std::cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! STARTED GRID ITER: "<<i<<std::endl;
            
            
            FitResult<T> *  prevresult = new FitResult<T>; // prevresult is ptr to the prev result object
            //std::unique_ptr<FitResult> prevresult;
            if (i > 0)
            {
                //prevresult = std::move(G.back());
                *prevresult = *(G.back());
            }
            
            currentskip = false;
            
            if (prevskip == false)
            {
                std::iota(idx.begin(), idx.end(), 0); // make global class var later
                // Exclude the first NoSelectK features from sorting.
                if (PartialSort && p > 5000 + NoSelectK)
                    std::partial_sort(idx.begin() + NoSelectK, idx.begin() + 5000 + NoSelectK, idx.end(), [this](unsigned int i1, unsigned int i2) {return (*Xtr)[i1] > (*Xtr)[i2] ;});
                else
                    std::sort(idx.begin() + NoSelectK, idx.end(), [this](unsigned int i1, unsigned int i2) {return (*Xtr)[i1] > (*Xtr)[i2] ;});
                P.CyclingOrder = 'u';
                P.Uorder = idx; // can be made faster
                
                //
                Xrmax = (*Xtr)[idx[NoSelectK]];
                
                if (i > 0)
                {
                    std::vector<unsigned int> Sp;
                    arma::sp_mat::const_iterator it;
                    for(it = prevresult->B.begin(); it != prevresult->B.end(); ++it)
                    {
                        Sp.push_back(it.row());
                    }
                    
                    for(unsigned int l = NoSelectK; l < p; ++l)
                    {
                        if ( std::binary_search(Sp.begin(), Sp.end(), idx[l]) == false )
                        {
                            Xrmax = (*Xtr)[idx[l]];
                            //std::cout<<"Grid Iteration: "<<i<<" Xrmax= "<<Xrmax<<std::endl;
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
            if(i >= 1 && !scaledown && !LambdaU)
            {
                P.ModelParams[0] = (((Xrmax - P.ModelParams[1]) * (Xrmax - P.ModelParams[1])) / (2 * (Lipconst))) * 0.99; // for numerical stability issues.
                if (P.ModelParams[0] >= prevresult->ModelParams[0])
                {
                    P.ModelParams[0] = prevresult->ModelParams[0] * 0.97;
                    //std::cout<<"INSTABILITY HANDELED"<<std::endl;
                } // handles numerical instability.
            }
            else if (i >= 1 && !LambdaU)
            {
                P.ModelParams[0] = std::min(P.ModelParams[0] * ScaleDownFactor, (((Xrmax - P.ModelParams[1]) * (Xrmax - P.ModelParams[1])) / (2 * (Lipconst))) * 0.97 );
            } // add 0.9 as an R param
            
            else if (i >= 1 && LambdaU)
            {
                P.ModelParams[0] = Lambdas[i];
            }
            
            // double thr = sqrt(2 * P.ModelParams[0] * (Lipconst)) + P.ModelParams[1]; // pass this to class? we're calc this twice now
            
            /*
             if (P.Iter>0 && std::abs(Xrmax) < thr) // not needed anymore. Remove later.
             { // Iternum>1 ensures that we have a good approximation to Xtr
             std::cout<<"Wrong Branch!"<<std::endl;
             
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
             
             */
            
            //if(currentskip == true){ // Debugging remove later
            //	std::cout<<"!!!!Skipped!!!"<<std::endl; // nothing will be pushed back to G, which is fine from MSE perspective.
            //}
            
            if(currentskip == false)
            {
                
                auto Model = make_CD(*X, *y, P);
                std::unique_ptr<FitResult<T>> result(new FitResult<T>);
                *result = Model->Fit();
                delete Model;
                
                //if (i>=1 && arma::norm(result->B-(G.back())->B,"inf")/arma::norm((G.back())->B,"inf") < 0.05){scaledown = true;} // got same solution
                
                scaledown = false;
                if (i >= 1)
                {
                    std::vector<unsigned int> Spold;
                    arma::sp_mat::const_iterator itold;
                    for(itold = prevresult->B.begin(); itold != prevresult->B.end(); ++itold)
                    {
                        Spold.push_back(itold.row());
                    }
                    
                    std::vector<unsigned int> Spnew;
                    arma::sp_mat::const_iterator itnew;
                    for(itnew = result->B.begin(); itnew != result->B.end(); ++itnew)
                    {
                        Spnew.push_back(itnew.row());
                    }
                    
                    bool samesupp = false;
                    
                    if (Spold == Spnew) {samesupp = true;}
                    
                    //
                    
                    if (samesupp) {scaledown = true;} // got same solution
                }
                //else {scaledown = false;}
                G.push_back(std::move(result));
                //std::cout<<"### ### ###"<<std::endl;
                //std::cout<<"Iteration: "<<i<<". "<<"Nnz: "<< result->B.n_nonzero << ". Lambda: "<<P.ModelParams[0]<< std::endl;
                if(G.back()->B.n_nonzero >= StopNum) {break;}
                //result->B.t().print();
                P.InitialSol = &(G.back()->B);
                P.b0 = G.back()->intercept;
                // Udate: After 1.1.0, P.r is automatically updated by the previous call to CD
                //*P.r = G.back()->r;
            }
            
            delete prevresult;
            
            //std::cout<<"Lambda0, Lambda1, Lambda2: "<<P.ModelParams[0]<<", "<<P.ModelParams[1]<<", "<<P.ModelParams[2]<<std::endl;
            
            // Prepare P for next iteration
            //P.ModelParams[0] = Lambdas[i+1];
            //P.ModelParams[0] = ((Xrmax - P.ModelParams[1])*(Xrmax - P.ModelParams[1]))/(2*(1+2*P.ModelParams[2]));
            
            P.Init = 'u';
            P.Iter += 1;
            prevskip = currentskip;
        }
    }
    
    
    /*
     else if (P.Specs.L1 || P.Specs.L1Relaxed)
     {
     
     
     *Xtr = arma::conv_to< std::vector<double> >::from(arma::abs(y->t() * *X).t()); // ToDO: double computation, handle later
     std::vector<unsigned int> idx(p);
     // Derive lambda_max
     double lambdamax  = arma::norm(y->t() * *X, "inf");
     
     //std::cout<< "Lambda max: "<< lambdamax << std::endl;
     double lambdamin = lambdamax * LambdaMinFactor;
     Lambdas = arma::logspace(std::log10(lambdamin), std::log10(lambdamax), G_ncols);
     Lambdas = arma::flipud(Lambdas);
     
     
     //unsigned int StopNum = (X->n_rows < NnzStopNum) ? X->n_rows : NnzStopNum;
     P.Init = 'z'; //////////
     unsigned int StopNum = NnzStopNum;
     
     for (unsigned int i = 0; i < G_ncols; ++i)
     {
     
     
     std::iota(idx.begin(), idx.end(), 0); // make global class var later
     //std::partial_sort(idx.begin(),idx.begin()+100, idx.end(),[this](unsigned int i1, unsigned int i2) {return (*Xtr)[i1] > (*Xtr)[i2] ;});
     std::sort(idx.begin(), idx.end(), [this](unsigned int i1, unsigned int i2) {return (*Xtr)[i1] > (*Xtr)[i2] ;}); /////////////////////////////////////////////////////
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
     
     */
    
    
    
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
    
    /*
     if (Refine == true)
     {
     for (unsigned int i = 0; i < 20; ++i)
     {
     bool better = false;
     //std::cout<<"!!!!!!!!!!!!! Backward-Forward Iteration: "<<i<<std::endl;
     for (auto it = G.rbegin() + 1; it != G.rend(); ++it)
     {
     P.InitialSol = &((*(it - 1))->B);
     P.ModelParams[0] = (*it)->ModelParams[0];
     //std::cout<<"### ModelParams[0]: "<<P.ModelParams[0]<<std::endl;
     
     auto Model = make_CD(*X, *y, P);
     
     FitResult * result = new FitResult; // Later: Double check memory leaks..
     
     *result = Model->Fit();
     
     
     
     if (result->Objective < (*it)->Objective)
     {
     *it = result;
     better = true;
     //std::cout<<"Found better in reverse grid"<<std::endl;
     }
     
     
     }
     
     for (auto it = G.begin() + 1; it != G.end(); ++it)
     {
     P.InitialSol = &((*(it - 1))->B);
     P.ModelParams[0] = (*it)->ModelParams[0];
     //std::cout<<"### ModelParams[0]: "<<P.ModelParams[0]<<std::endl;
     
     auto Model = make_CD(*X, *y, P);
     
     FitResult * result = new FitResult; // Later: Double check memory leaks..
     
     *result = Model->Fit();
     
     if (result->Objective < (*it)->Objective)
     {
     *it = result;
     better = true;
     //std::cout<<"Found better in forward grid"<<std::endl;
     }
     
     }
     if (better == false) {break;} //fixed grid.
     }
     }
     */
    
    return std::move(G);
}

#endif
