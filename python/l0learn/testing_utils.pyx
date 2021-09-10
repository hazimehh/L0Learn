cimport cython
import numpy as np
cimport numpy as np

from l0learn.cyarma cimport (sp_dmat_field_to_list, dvec_field_to_list, list_to_sp_dmat_field, list_to_dvec_field,
field, dmat, dvec, sp_dmat, uvec,numpy_to_dmat_d, dmat_to_numpy, numpy_to_uvec_d,
                             uvec_to_numpy, numpy_to_dvec_d, dvec_to_numpy, numpy_to_sp_dmat_d,
                             sp_dmat_to_numpy)

def list_csc_to_arma_csc_to_list(x):
    return  c_list_csc_to_arma_csc_to_list(x)

cdef list c_list_csc_to_arma_csc_to_list(list x):
    cdef field[sp_dmat] f = list_to_sp_dmat_field(x)
    cdef lst = sp_dmat_field_to_list(f)
    return lst

def list_dvec_to_arma_dvec_to_list(x):
    return  c_list_dvec_to_arma_dvec_to_list(x)

cdef list c_list_dvec_to_arma_dvec_to_list(list x):
    cdef field[dvec] f = list_to_dvec_field(x)
    cdef lst = dvec_field_to_list(f)
    return lst


def dmat_np_to_arma_to_np(x):
    cdef np.ndarray[np.double_t, ndim=2] x_pass
    x_pass = c_dmat_np_to_arma_to_np(x)

    return x_pass


cdef np.ndarray[np.double_t, ndim=2] c_dmat_np_to_arma_to_np(np.ndarray[np.double_t, ndim=2] x):
    cdef dmat arma_x = numpy_to_dmat_d(x)

    cdef np.ndarray[np.double_t, ndim=2] np_x = dmat_to_numpy(arma_x, None)

    return np_x


def uvec_np_to_arma_to_np(x):
    cdef np.ndarray[np.uint64_t, ndim=1] x_pass
    x_pass = c_uvec_np_to_arma_to_np(x)

    return x_pass


cdef np.ndarray[np.uint64_t, ndim=1] c_uvec_np_to_arma_to_np(np.ndarray[np.uint64_t, ndim=1] x):
    cdef uvec arma_x = numpy_to_uvec_d(x)

    cdef np.ndarray[np.uint64_t, ndim=1] np_x = uvec_to_numpy(arma_x, None)

    return np_x


def dvec_np_to_arma_to_np(x):
    cdef np.ndarray[np.double_t, ndim=1] x_pass
    x_pass = c_dvec_np_to_arma_to_np(x)

    return x_pass


cdef np.ndarray[np.uint64_t, ndim=1] c_dvec_np_to_arma_to_np(np.ndarray[np.double_t, ndim=1] x):
    cdef dvec arma_x = numpy_to_dvec_d(x)

    cdef np.ndarray[np.double_t, ndim=1] np_x = dvec_to_numpy(arma_x, None)

    return np_x


def sp_dmat_np_to_arma_to_np(x):
    return c_sp_dmat_np_to_arma_to_np(x)


cdef c_sp_dmat_np_to_arma_to_np(x):
    cdef sp_dmat arma_x = numpy_to_sp_dmat_d(x)

    np_x = sp_dmat_to_numpy(arma_x, None, None, None)
    return np_x

def dmat_dot_dmat(x, y):
    cdef dmat arma_x = numpy_to_dmat_d(x)
    cdef dmat arma_y = numpy_to_dmat_d(y)

    cdef dmat arma_xy =  c_dmat_dot_dmat(arma_x, arma_y)

    cdef np.ndarray[np.double_t, ndim=2] np_xy = dmat_to_numpy(arma_xy, None)

    return np_xy

cdef dmat c_dmat_dot_dmat(dmat arma_x,  dmat arma_y):
    cdef dmat arma_xy = arma_x*arma_y

    return arma_xy