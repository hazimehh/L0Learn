# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool as cppbool
from l0learn.cyarma cimport field, dmat, sp_dmat, dvec

TEST_INTERFACE = "INSTALLED"

cdef extern from "src/include/L0LearnCore.h":

    cdef cppclass fitmodel:
        vector[vector[double]] Lambda0;
        vector[double] Lambda12;
        vector[vector[size_t]] NnzCount;
        field[sp_dmat] Beta;
        vector[vector[double]] Intercept;
        vector[vector[cppbool]] Converged;

        fitmodel() except +
        fitmodel(const fitmodel &) except +

    cdef cppclass cvfitmodel(fitmodel):
        field[dvec] CVMeans;
        field[dvec] CVSDs;

        cvfitmodel() except +
        cvfitmodel(const cvfitmodel &) except +

    cdef fitmodel L0LearnFit[T](const T& X,
                                const dvec& y,
                                const string Loss,
                                const string Penalty,
                                const string Algorithm,
                                const size_t NnzStopNum,
                                const size_t G_ncols,
                                const size_t G_nrows,
                                const double Lambda2Max,
                                const double Lambda2Min,
                                const cppbool PartialSort,
                                const size_t MaxIters,
                                const double rtol,
                                const double atol,
                                const cppbool ActiveSet,
                                const size_t ActiveSetNum,
                                const size_t MaxNumSwaps,
                                const double ScaleDownFactor,
                                const size_t ScreenSize,
                                const cppbool LambdaU,
                                const vector[vector[double]]& Lambdas,
                                const size_t ExcludeFirstK,
                                const cppbool Intercept,
                                const cppbool withBounds,
                                const dvec &Lows,
                                const dvec &Highs);

    cdef cvfitmodel L0LearnCV[T](const T& X,
                                const dvec& y,
                                const string Loss,
                                const string Penalty,
                                const string Algorithm,
                                const size_t NnzStopNum,
                                const size_t G_ncols,
                                const size_t G_nrows,
                                const double Lambda2Max,
                                const double Lambda2Min,
                                const cppbool PartialSort,
                                const size_t MaxIters,
                                const double rtol,
                                const double atol,
                                const cppbool ActiveSet,
                                const size_t ActiveSetNum,
                                const size_t MaxNumSwaps,
                                const double ScaleDownFactor,
                                const size_t ScreenSize,
                                const cppbool LambdaU,
                                const vector[vector[double]]& Lambdas,
                                const unsigned int nfolds,
                                const double seed,
                                const size_t ExcludeFirstK,
                                const cppbool Intercept,
                                const cppbool withBounds,
                                const dvec &Lows,
                                const dvec &Highs)

cdef fitmodel _L0LearnFit_dense(const dmat& X,
                                const dvec& y,
                                const string Loss,
                                const string Penalty,
                                const string Algorithm,
                                const size_t NnzStopNum,
                                const size_t G_ncols,
                                const size_t G_nrows,
                                const double Lambda2Max,
                                const double Lambda2Min,
                                const cppbool PartialSort,
                                const size_t MaxIters,
                                const double rtol,
                                const double atol,
                                const cppbool ActiveSet,
                                const size_t ActiveSetNum,
                                const size_t MaxNumSwaps,
                                const double ScaleDownFactor,
                                const size_t ScreenSize,
                                const cppbool LambdaU,
                                const vector[vector[double]]& Lambdas,
                                const size_t ExcludeFirstK,
                                const cppbool Intercept,
                                const cppbool withBounds,
                                const dvec &Lows,
                                const dvec &Highs);


cdef fitmodel _L0LearnFit_sparse(const sp_dmat& X,
                                 const dvec& y,
                                 const string Loss,
                                 const string Penalty,
                                 const string Algorithm,
                                 const size_t NnzStopNum,
                                 const size_t G_ncols,
                                 const size_t G_nrows,
                                 const double Lambda2Max,
                                 const double Lambda2Min,
                                 const cppbool PartialSort,
                                 const size_t MaxIters,
                                 const double rtol,
                                 const double atol,
                                 const cppbool ActiveSet,
                                 const size_t ActiveSetNum,
                                 const size_t MaxNumSwaps,
                                 const double ScaleDownFactor,
                                 const size_t ScreenSize,
                                 const cppbool LambdaU,
                                 const vector[vector[double]]& Lambdas,
                                 const size_t ExcludeFirstK,
                                 const cppbool Intercept,
                                 const cppbool withBounds,
                                 const dvec &Lows,
                                 const dvec &Highs);


cdef cvfitmodel _L0LearnCV_dense(const dmat& X,
                                 const dvec& y,
                                 const string Loss,
                                 const string Penalty,
                                 const string Algorithm,
                                 const size_t NnzStopNum,
                                 const size_t G_ncols,
                                 const size_t G_nrows,
                                 const double Lambda2Max,
                                 const double Lambda2Min,
                                 const cppbool PartialSort,
                                 const size_t MaxIters,
                                 const double rtol,
                                 const double atol,
                                 const cppbool ActiveSet,
                                 const size_t ActiveSetNum,
                                 const size_t MaxNumSwaps,
                                 const double ScaleDownFactor,
                                 const size_t ScreenSize,
                                 const cppbool LambdaU,
                                 const vector[vector[double]]& Lambdas,
                                 const unsigned int nfolds,
                                 const double seed,
                                 const size_t ExcludeFirstK,
                                 const cppbool Intercept,
                                 const cppbool withBounds,
                                 const dvec &Lows,
                                 const dvec &Highs);


cdef cvfitmodel _L0LearnCV_sparse(const sp_dmat& X,
                                 const dvec& y,
                                 const string Loss,
                                 const string Penalty,
                                 const string Algorithm,
                                 const size_t NnzStopNum,
                                 const size_t G_ncols,
                                 const size_t G_nrows,
                                 const double Lambda2Max,
                                 const double Lambda2Min,
                                 const cppbool PartialSort,
                                 const size_t MaxIters,
                                 const double rtol,
                                 const double atol,
                                 const cppbool ActiveSet,
                                 const size_t ActiveSetNum,
                                 const size_t MaxNumSwaps,
                                 const double ScaleDownFactor,
                                 const size_t ScreenSize,
                                 const cppbool LambdaU,
                                 const vector[vector[double]]& Lambdas,
                                 const unsigned int nfolds,
                                 const double seed,
                                 const size_t ExcludeFirstK,
                                 const cppbool Intercept,
                                 const cppbool withBounds,
                                 const dvec &Lows,
                                 const dvec &Highs);
