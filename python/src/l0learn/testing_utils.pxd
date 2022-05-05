# distutils: language = c++

cimport numpy as np

from l0learn.cyarma cimport dmat

cdef list c_list_csc_to_arma_csc_to_list(list x)

cdef np.ndarray[np.double_t, ndim=2] c_dmat_np_to_arma_to_np(np.ndarray[np.double_t, ndim=2] x)

cdef np.ndarray[np.uint64_t, ndim=1] c_uvec_np_to_arma_to_np(np.ndarray[np.uint64_t, ndim=1] x)

cdef np.ndarray[np.uint64_t, ndim=1] c_dvec_np_to_arma_to_np(np.ndarray[np.double_t, ndim=1] x)

cdef c_sp_dmat_np_to_arma_to_np(x)

cdef dmat c_dmat_dot_dmat(dmat arma_x, dmat arma_y)
