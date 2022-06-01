#include "pyl0learn.h"

namespace py = pybind11;


arma::sp_mat to_sparse(const py::object& S){
  py::tuple shape = S.attr("shape").cast< py::tuple >();
  const auto nr = shape[0].cast< size_t >(), nc = shape[1].cast< size_t >();
  const arma::uvec ind = carma::arr_to_col(S.attr("indices").cast< py::array_t< arma::uword > >());
  const arma::uvec ind_ptr = carma::arr_to_col(S.attr("indptr").cast< py::array_t< arma::uword > >());
  const arma::vec data = carma::arr_to_col(S.attr("data").cast< py::array_t< double > >());
  return arma::sp_mat(ind, ind_ptr, data, nr, nc);
}

py::object to_py_sparse(const arma::sp_mat& X){
  py::module_ scipy_sparse = py::module_::import("scipy.sparse");

  py::array_t< double > values = carma::col_to_arr<double>(arma::vec(X.values, X.n_elem));
  py::array_t< arma::uword > rowind = carma::col_to_arr<arma::uword>(arma::uvec (X.row_indices, X.n_elem));
  py::array_t< arma::uword > colptr = carma::col_to_arr<arma::uword>(arma::uvec (X.col_ptrs, X.n_cols + 1));
  const py::tuple shape = py::make_tuple(X.n_rows, X.n_cols);

  return scipy_sparse.attr("csc_matrix")(py::make_tuple(values.squeeze(), rowind.squeeze(), colptr.squeeze()), shape);
}

py::list col_field_to_list(const arma::field<arma::vec> &x) {
  py::list lst(x.n_elem);

  for (auto i = 0; i < x.n_elem; i++) {
    lst[i] = carma::col_to_arr<double>(x[i]);
  }
  return lst;
}


py::list sparse_field_to_list(const arma::field<arma::sp_mat> &x) {
  py::list lst(x.n_elem);

  for (auto i = 0; i < x.n_elem; i++) {
    lst[i] = to_py_sparse(x[i]);
  }
  return lst;
}

py_fitmodel L0LearnFit_sparse_wrapper(
    const py::object &X, const arma::vec &y, const std::string &Loss,
    const std::string &Penalty, const std::string &Algorithm,
    const unsigned int NnzStopNum, const unsigned int G_ncols,
    const unsigned int G_nrows, const double Lambda2Max,
    const double Lambda2Min, const bool PartialSort,
    const unsigned int MaxIters, const double rtol, const double atol,
    const bool ActiveSet, const unsigned int ActiveSetNum,
    const unsigned int MaxNumSwaps, const double ScaleDownFactor,
    const unsigned int ScreenSize, const bool LambdaU,
    const std::vector<std::vector<double>> &Lambdas,
    const unsigned int ExcludeFirstK, const bool Intercept,
    const bool withBounds, const arma::vec &Lows, const arma::vec &Highs) {

  return py_fitmodel(L0LearnFit<arma::sp_mat>(
        to_sparse(X), y, Loss, Penalty, Algorithm, NnzStopNum, G_ncols,
      G_nrows, Lambda2Max, Lambda2Min, PartialSort, MaxIters, rtol, atol,
      ActiveSet, ActiveSetNum, MaxNumSwaps, ScaleDownFactor, ScreenSize,
      LambdaU, Lambdas, ExcludeFirstK, Intercept, withBounds, Lows, Highs));
}

py_fitmodel L0LearnFit_dense_wrapper(
    const arma::mat &X, const arma::vec &y, const std::string &Loss,
    const std::string &Penalty, const std::string &Algorithm,
    const unsigned int NnzStopNum, const unsigned int G_ncols,
    const unsigned int G_nrows, const double Lambda2Max,
    const double Lambda2Min, const bool PartialSort,
    const unsigned int MaxIters, const double rtol, const double atol,
    const bool ActiveSet, const unsigned int ActiveSetNum,
    const unsigned int MaxNumSwaps, const double ScaleDownFactor,
    const unsigned int ScreenSize, const bool LambdaU,
    const std::vector<std::vector<double>> &Lambdas,
    const unsigned int ExcludeFirstK, const bool Intercept,
    const bool withBounds, const arma::vec &Lows, const arma::vec &Highs) {
  return py_fitmodel(L0LearnFit<arma::mat>(
      X, y, Loss, Penalty, Algorithm, NnzStopNum, G_ncols, G_nrows, Lambda2Max,
      Lambda2Min, PartialSort, MaxIters, rtol, atol, ActiveSet, ActiveSetNum,
      MaxNumSwaps, ScaleDownFactor, ScreenSize, LambdaU, Lambdas, ExcludeFirstK,
      Intercept, withBounds, Lows, Highs));
}

py_cvfitmodel L0LearnCV_dense_wrapper(
    const arma::mat &X, const arma::vec &y, const std::string &Loss,
    const std::string &Penalty, const std::string &Algorithm,
    const unsigned int NnzStopNum, const unsigned int G_ncols,
    const unsigned int G_nrows, const double Lambda2Max,
    const double Lambda2Min, const bool PartialSort,
    const unsigned int MaxIters, const double rtol, const double atol,
    const bool ActiveSet, const unsigned int ActiveSetNum,
    const unsigned int MaxNumSwaps, const double ScaleDownFactor,
    const unsigned int ScreenSize, const bool LambdaU,
    const std::vector<std::vector<double>> &Lambdas, const unsigned int nfolds,
    const size_t seed, const unsigned int ExcludeFirstK, const bool Intercept,
    const bool withBounds, const arma::vec &Lows, const arma::vec &Highs) {
  return py_cvfitmodel(L0LearnCV<arma::mat>(
      X, y, Loss, Penalty, Algorithm, NnzStopNum, G_ncols, G_nrows, Lambda2Max,
      Lambda2Min, PartialSort, MaxIters, rtol, atol, ActiveSet, ActiveSetNum,
      MaxNumSwaps, ScaleDownFactor, ScreenSize, LambdaU, Lambdas, nfolds, seed,
      ExcludeFirstK, Intercept, withBounds, Lows, Highs));
}

py_cvfitmodel L0LearnCV_sparse_wrapper(
    const py::object &X, const arma::vec &y, const std::string &Loss,
    const std::string &Penalty, const std::string &Algorithm,
    const unsigned int NnzStopNum, const unsigned int G_ncols,
    const unsigned int G_nrows, const double Lambda2Max,
    const double Lambda2Min, const bool PartialSort,
    const unsigned int MaxIters, const double rtol, const double atol,
    const bool ActiveSet, const unsigned int ActiveSetNum,
    const unsigned int MaxNumSwaps, const double ScaleDownFactor,
    const unsigned int ScreenSize, const bool LambdaU,
    const std::vector<std::vector<double>> &Lambdas, const unsigned int nfolds,
    const size_t seed, const unsigned int ExcludeFirstK, const bool Intercept,
    const bool withBounds, const arma::vec &Lows, const arma::vec &Highs) {
  return py_cvfitmodel(L0LearnCV<arma::sp_mat>(
      to_sparse(X), y, Loss, Penalty, Algorithm, NnzStopNum, G_ncols,
      G_nrows, Lambda2Max, Lambda2Min, PartialSort, MaxIters, rtol, atol,
      ActiveSet, ActiveSetNum, MaxNumSwaps, ScaleDownFactor, ScreenSize,
      LambdaU, Lambdas, nfolds, seed, ExcludeFirstK, Intercept, withBounds,
      Lows, Highs));
}

PYBIND11_MODULE(l0learn, m) {
  m.def("_L0LearnFit_dense", &L0LearnFit_dense_wrapper);

  m.def("_L0LearnFit_sparse", &L0LearnFit_sparse_wrapper);

  m.def("_L0LearnCV_dense", &L0LearnCV_dense_wrapper);

  m.def("_L0LearnCV_sparse", &L0LearnCV_sparse_wrapper);

  py::class_<py_fitmodel>(m, "_py_fitmodel")
      .def(py::init<fitmodel const &>())
      .def_readonly("Lambda0", &py_fitmodel::Lambda0)
      .def_readonly("Lambda12", &py_fitmodel::Lambda12)
      .def_readonly("NnzCount", &py_fitmodel::NnzCount)
      .def_readonly("Beta", &py_fitmodel::Beta)
      .def_readonly("Intercept", &py_fitmodel::Intercept)
      .def_readonly("Converged", &py_fitmodel::Converged);

  py::class_<py_cvfitmodel>(m, "_py_cvfitmodel")
      .def(py::init<cvfitmodel const &>())
      .def_readonly("Lambda0", &py_cvfitmodel::Lambda0)
      .def_readonly("Lambda12", &py_cvfitmodel::Lambda12)
      .def_readonly("NnzCount", &py_cvfitmodel::NnzCount)
      .def_readonly("Beta", &py_cvfitmodel::Beta)
      .def_readonly("Intercept", &py_cvfitmodel::Intercept)
      .def_readonly("Converged", &py_cvfitmodel::Converged)
      .def_readonly("CVMeans", &py_cvfitmodel::CVMeans)
      .def_readonly("CVSDs", &py_cvfitmodel::CVSDs);
}
