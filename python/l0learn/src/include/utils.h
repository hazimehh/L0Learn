#ifndef L0LEARN_UTILS_HPP
#define L0LEARN_UTILS_HPP
#include <vector>

#include "BetaVector.h"
#include "arma_includes.h"

template <typename T>
inline T clamp(T x, T low, T high) {
  // -O3 Compiler should remove branches
  if (x < low) x = low;
  if (x > high) x = high;
  return x;
}

template <typename T1>
arma::vec inline matrix_column_get(const arma::mat &mat, T1 col) {
  return mat.unsafe_col(col);
}

template <typename T1>
arma::vec inline matrix_column_get(const arma::sp_mat &mat, T1 col) {
  return arma::vec(mat.col(col));
}

template <typename T1>
arma::mat inline matrix_rows_get(const arma::mat &mat,
                                 const T1 vector_of_row_indices) {
  return mat.rows(vector_of_row_indices);
}

template <typename T1>
arma::sp_mat inline matrix_rows_get(const arma::sp_mat &mat,
                                    const T1 vector_of_row_indices) {
  // Option for CV for random splitting or contiguous splitting.
  // 1 - N without permutations splitting at floor(N/n_folds)
  arma::sp_mat row_mat = arma::sp_mat(vector_of_row_indices.n_elem, mat.n_cols);

  for (auto i = 0; i < vector_of_row_indices.n_elem; i++) {
    auto row_index = vector_of_row_indices(i);
    arma::sp_mat::const_row_iterator begin = mat.begin_row(row_index);
    arma::sp_mat::const_row_iterator end = mat.end_row(row_index);

    for (; begin != end; ++begin) {
      row_mat(i, begin.col()) = *begin;
    }
  }
  return row_mat;
}

template <typename T1>
arma::mat inline matrix_vector_schur_product(const arma::mat &mat,
                                             const T1 &y) {
  // return mat[i, j] * y[i] for each j
  return mat.each_col() % *y;
}

template <typename T1>
arma::sp_mat inline matrix_vector_schur_product(const arma::sp_mat &mat,
                                                const T1 &y) {
  arma::sp_mat Xy = arma::sp_mat(mat);
  arma::sp_mat::iterator begin = Xy.begin();
  arma::sp_mat::iterator end = Xy.end();

  auto yp = (*y);
  for (; begin != end; ++begin) {
    auto row = begin.row();
    *begin = (*begin) * yp(row);
  }
  return Xy;
}

template <typename T1>
arma::sp_mat inline matrix_vector_divide(const arma::sp_mat &mat, const T1 &u) {
  arma::sp_mat divided_mat = arma::sp_mat(mat);

  // auto up = (*u);
  arma::sp_mat::iterator begin = divided_mat.begin();
  arma::sp_mat::iterator end = divided_mat.end();
  for (; begin != end; ++begin) {
    *begin = (*begin) / u(begin.row());
  }
  return divided_mat;
}

template <typename T1>
arma::mat inline matrix_vector_divide(const arma::mat &mat, const T1 &u) {
  return mat.each_col() / u;
}

arma::rowvec inline matrix_column_sums(const arma::mat &mat) {
  return arma::sum(mat, 0);
}

arma::rowvec inline matrix_column_sums(const arma::sp_mat &mat) {
  return arma::rowvec(arma::sum(mat, 0));
}

template <typename T1, typename T2>
double inline matrix_column_dot(const arma::mat &mat, T1 col, const T2 &u) {
  return arma::dot(matrix_column_get(mat, col), u);
}

template <typename T1, typename T2>
double inline matrix_column_dot(const arma::sp_mat &mat, T1 col, const T2 &u) {
  return arma::dot(matrix_column_get(mat, col), u);
}

template <typename T1, typename T2>
arma::vec inline matrix_column_mult(const arma::mat &mat, T1 col, const T2 &u) {
  return matrix_column_get(mat, col) * u;
}

template <typename T1, typename T2>
arma::vec inline matrix_column_mult(const arma::sp_mat &mat, T1 col,
                                    const T2 &u) {
  return matrix_column_get(mat, col) * u;
}

void inline clamp_by_vector(arma::vec &B, const arma::vec &lows,
                            const arma::vec &highs) {
  const std::size_t n = B.n_rows;
  for (std::size_t i = 0; i < n; i++) {
    B.at(i) = clamp(B.at(i), lows.at(i), highs.at(i));
  }
}

// void clamp_by_vector(arma::sp_mat &B, const arma::vec& lows, const arma::vec&
// highs){
//     // See above implementation without filter for error.
//     auto begin = B.begin();
//     auto end = B.end();
//
//     std::vector<std::size_t> inds;
//     for (; begin != end; ++begin)
//         inds.push_back(begin.row());
//
//     auto n = B.size();
//     inds.erase(std::remove_if(inds.begin(),
//                               inds.end(),
//                               [n](size_t x){return (x > n) && (x < 0);}),
//                               inds.end());
//     for (auto& it : inds) {
//         double B_item = B(it, 0);
//         const double low = lows(it);
//         const double high = highs(it);
//         B(it, 0) = clamp(B_item, low, high);
//     }
// }

arma::rowvec inline matrix_normalize(arma::sp_mat &mat_norm) {
  auto p = mat_norm.n_cols;
  arma::rowvec scaleX =
      arma::zeros<arma::rowvec>(p);  // will contain the l2norm of every col

  for (auto col = 0; col < p; col++) {
    double l2norm = arma::norm(matrix_column_get(mat_norm, col), 2);
    scaleX(col) = l2norm;
  }

  scaleX.replace(0, -1);

  for (auto col = 0; col < p; col++) {
    arma::sp_mat::col_iterator begin = mat_norm.begin_col(col);
    arma::sp_mat::col_iterator end = mat_norm.end_col(col);
    for (; begin != end; ++begin) (*begin) = (*begin) / scaleX(col);
  }

  if (mat_norm.has_nan())
    mat_norm.replace(arma::datum::nan,
                     0);  // can handle numerical instabilities.

  return scaleX;
}

arma::rowvec inline matrix_normalize(arma::mat &mat_norm) {
  auto p = mat_norm.n_cols;
  arma::rowvec scaleX =
      arma::zeros<arma::rowvec>(p);  // will contain the l2norm of every col

  for (auto col = 0; col < p; col++) {
    double l2norm = arma::norm(matrix_column_get(mat_norm, col), 2);
    scaleX(col) = l2norm;
  }

  scaleX.replace(0, -1);
  mat_norm.each_row() /= scaleX;

  if (mat_norm.has_nan()) {
    mat_norm.replace(arma::datum::nan,
                     0);  // can handle numerical instabilities.
  }

  return scaleX;
}

arma::rowvec inline matrix_center(const arma::mat &X, arma::mat &X_normalized,
                                  bool intercept) {
  auto p = X.n_cols;
  arma::rowvec meanX;

  if (intercept) {
    meanX = arma::mean(X, 0);
    X_normalized = X.each_row() - meanX;
  } else {
    meanX = arma::zeros<arma::rowvec>(p);
    X_normalized = arma::mat(X);
  }

  return meanX;
}

arma::rowvec inline matrix_center(const arma::sp_mat &X,
                                  arma::sp_mat &X_normalized, bool intercept) {
  auto p = X.n_cols;
  arma::rowvec meanX = arma::zeros<arma::rowvec>(p);
  X_normalized = arma::sp_mat(X);
  return meanX;
}

#endif  // L0LEARN_UTILS_HPP
