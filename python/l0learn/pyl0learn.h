#ifndef PYTHON_L0LEARN_CORE_H
#define PYTHON_L0LEARN_CORE_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "L0LearnCore.h"
#include "arma_includes.h"

#include <utility>

//struct sparse_mat {
//  const arma::uvec rowind;
//  const arma::uvec colptr;
//  const arma::vec values;
//  const arma::uword n_rows;
//  const arma::uword n_cols;
//
//  explicit sparse_mat(const arma::sp_mat &x)
//      : rowind(arma::uvec(x.row_indices, x.n_elem)),
//        colptr(arma::uvec(x.col_ptrs, x.n_cols + 1)),
//        values(arma::vec(x.values, x.n_elem)),
//        n_rows(x.n_rows),
//        n_cols(x.n_cols) {}
////  sparse_mat(const sparse_mat &x)
////      : rowind(x.rowind),
////        colptr(x.colptr),
////        values(x.values),
////        n_rows(x.n_rows),
////        n_cols(x.n_cols) {}
//  sparse_mat(arma::uvec rowind, arma::uvec colptr,
//             arma::vec values, const arma::uword n_rows,
//             const arma::uword n_cols)
//      : rowind(std::move(rowind)),
//        colptr(std::move(colptr)),
//        values(std::move(values)),
//        n_rows(n_rows),
//        n_cols(n_cols) {
//    COUT << "rowind/row_indices.size" << this->rowind.size() << " " << arma::sum(this->rowind) <<" \n";
//    COUT << "colptr/col_ptrs" << this->colptr.size() << " " << arma::sum(this->colptr) <<" \n";
//    COUT << "values/values" << this->values.size() << " " << arma::sum(this->values) <<" \n";
//  }
//
//  arma::sp_mat to_arma_object() const {
//    COUT << "to_arma_object begin\n";
//    COUT << "rowind/row_indices " << this->rowind << "\n";
//    COUT << "rowind/row_indices.size" << this->rowind.size() << " " << arma::sum(this->rowind) <<" \n";
//    COUT << "colptr/col_ptrs" << this->colptr.size() << " " << arma::sum(this->colptr) <<" \n";
//    COUT << "values/values" << this->values.size() << " " << arma::sum(this->values) <<" \n";
//    COUT << "n_rows/n_rows" << this->n_rows <<" \n";
//    COUT << "n_cols/n_cols" << this->n_cols <<" \n";
//    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
//    return {this->rowind, this->colptr, this->values, this->n_rows,
//            this->n_cols};
//  }
//};

py::list col_field_to_list(const arma::field<arma::vec> &x);

py::list sparse_field_to_list(const arma::field<arma::sp_mat> &x);



struct py_fitmodel {
  std::vector<std::vector<double>> Lambda0;
  std::vector<double> Lambda12;
  std::vector<std::vector<std::size_t>> NnzCount;
  py::list Beta;
  std::vector<std::vector<double>> Intercept;
  std::vector<std::vector<bool>> Converged;

  py_fitmodel(const py_fitmodel &) = default;
  py_fitmodel(std::vector<std::vector<double>> &lambda0,
              std::vector<double> &lambda12,
              std::vector<std::vector<std::size_t>> &nnzCount,
              py::list &beta,
              std::vector<std::vector<double>> &intercept,
              std::vector<std::vector<bool>> &converged)
      : Lambda0(lambda0),
        Lambda12(lambda12),
        NnzCount(nnzCount),
        Beta(beta),
        Intercept(intercept),
        Converged(converged) {}

  explicit py_fitmodel(const fitmodel &f)
      : Lambda0(f.Lambda0),
        Lambda12(f.Lambda12),
        NnzCount(f.NnzCount),
        Beta(sparse_field_to_list(f.Beta)),
        Intercept(f.Intercept),
        Converged(f.Converged) {}
};

struct py_cvfitmodel : py_fitmodel {
  py::list CVMeans;
  py::list CVSDs;

  py_cvfitmodel(const py_cvfitmodel &) = default;

  py_cvfitmodel(std::vector<std::vector<double>> &lambda0,
                std::vector<double> &lambda12,
                std::vector<std::vector<std::size_t>> &nnzCount,
                py::list &beta,
                std::vector<std::vector<double>> &intercept,
                std::vector<std::vector<bool>> &converged,
                py::list &cVMeans, py::list &cVSDs)
      : py_fitmodel(lambda0, lambda12, nnzCount, beta, intercept, converged),
        CVMeans(cVMeans),
        CVSDs(cVSDs) {}

  explicit py_cvfitmodel(const cvfitmodel &f)
      : py_fitmodel(f),
        CVMeans(col_field_to_list(f.CVMeans)),
        CVSDs(col_field_to_list(f.CVSDs)) {}
};

#endif  // PYTHON_L0LEARN_CORE_H
