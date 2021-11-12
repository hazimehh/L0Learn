import numpy as np
from scipy.sparse import csc_matrix
from hypothesis import given, assume, settings, HealthCheck
from hypothesis.extra import numpy as npst
from hypothesis.strategies import floats, integers, lists

from l0learn.testing_utils import (list_dvec_to_arma_dvec_to_list,
                                   list_csc_to_arma_csc_to_list,
                                   dmat_np_to_arma_to_np,
                                   dvec_np_to_arma_to_np,
                                   uvec_np_to_arma_to_np,
                                   sp_dmat_np_to_arma_to_np,
                                   dmat_dot_dmat)


@given(npst.arrays(np.float64, shape=npst.array_shapes(min_dims=2, max_dims=2, max_side=100)))
def test_dmat_np_to_arma_to_np(arr):
    arr = np.asfortranarray(arr)
    arr2 = dmat_np_to_arma_to_np(arr)
    np.testing.assert_array_equal(arr2, arr)


@given(npst.arrays(np.float64, shape=npst.array_shapes(min_dims=1, max_dims=1, max_side=100)))
def test_dvec_np_to_arma_to_np(arr):
    arr = np.asfortranarray(arr)
    arr2 = dvec_np_to_arma_to_np(arr)
    np.testing.assert_array_equal(arr2, arr)


@given(npst.arrays(np.uint64, shape=npst.array_shapes(min_dims=1, max_dims=1, max_side=100),
                   elements=integers(0, 1000)))
def test_uvec_np_to_arma_to_np(arr):
    arr = np.asfortranarray(arr)
    arr = arr.astype(np.uint64)
    arr2 = uvec_np_to_arma_to_np(arr)
    np.testing.assert_array_equal(arr2, arr)


@given(npst.arrays(np.float64,
                   shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=100),
                   elements=floats()),
       floats(0, 1))
def test_sp_dmat_np_to_arma_to_np(arr, thr):
    arr[abs(arr) >= thr] = 0.0
    assume(np.sum(arr != 0) > 1)
    arr = csc_matrix(arr)
    arr2 = sp_dmat_np_to_arma_to_np(arr)
    np.testing.assert_array_equal(arr2.toarray(), arr.toarray())


@given(lists(elements=npst.arrays(np.float64,
                                  shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=100),
                                  elements=floats()),
             min_size=1),
       floats(0, 1),)
@settings(suppress_health_check=(HealthCheck(3),), deadline=None)
def test_sp_dmat_field_np_to_arma_to_np(lst, thr):
    for i, arr in enumerate(lst):
        arr[abs(arr) <= thr] = 0.0
        assume(np.sum(arr != 0) > 1)
        lst[i] = csc_matrix(arr)

    lst2 = list_csc_to_arma_csc_to_list(lst)
    for arr, arr2 in zip(lst, lst2):
        np.testing.assert_array_equal(arr.toarray(), arr2.toarray())


@given(lists(elements=npst.arrays(np.float64,
                                  shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=100),
                                  elements=floats()),
             min_size=1))
def test_dvec_field_np_to_arma_to_np(lst):
    lst2 = list_dvec_to_arma_dvec_to_list(lst)
    for arr, arr2 in zip(lst, lst2):
        np.testing.assert_array_equal(arr, arr2)


@given(n=integers(min_value=1, max_value=100),
       p=integers(min_value=1, max_value=100),
       k=integers(min_value=1, max_value=100))
def test_dmat_dot_dmat(n, p, k):
    arr = np.random.random((n, p))
    arr = np.asfortranarray(arr)

    x = np.random.random((p, k))
    x = np.asfortranarray(x)

    arr2 = dmat_dot_dmat(arr, x)
    np.testing.assert_array_almost_equal(arr2, arr.dot(x))
