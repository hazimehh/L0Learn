cimport cython
import numpy as np
cimport numpy as np

from scipy.sparse.csc import csc_matrix
from cython.operator cimport dereference as deref, preincrement as inc

from libcpp cimport bool

cdef extern from "armadillo" namespace "arma" nogil:
    # matrix class (double)

    cdef cppclass dmat:
        dmat(double * aux_mem, int n_rows, int n_cols, bool copy_aux_mem, bool strict) nogil
        dmat(dmat& x)
        dmat() nogil
        int n_rows
        int n_cols
        int n_elem
        double * memptr() nogil

        # available for testing purposes
        dmat operator*(const dmat) nogil

    # vector class (double)
    cdef cppclass dvec:
        dvec(double * aux_mem, int number_of_elements, bool copy_aux_mem, bool strict) nogil
        dvec() nogil
        int n_rows
        int n_cols
        int n_elem
        double * memptr() nogil

    #vector class (int)
    cdef cppclass uvec:
        uvec(unsigned long long * aux_mem, int number_of_elements, bool copy_aux_mem, bool strict) nogil
        uvec() nogil
        int n_rows
        int n_cols
        int n_elem
        int n_nonzero
        unsigned long long * memptr() nogil

    # sparse matrix class (double)
    cdef cppclass sp_dmat:
        sp_dmat(uvec rowind, uvec colptr, dvec values, int n_rows, int n_cols) nogil
        sp_dmat() nogil
        int n_rows
        int n_cols
        int n_elem
        int n_nonzero
        double * memptr() nogil
        uvec row_indices
        uvec col_ptrs
        dvec values

        # double& operator[](int) nogil
        # double& operator[](int,int) nogil
        # double& at(int,int) nogil
        # double& at(int) nogil

        cppclass const_iterator:
            const_iterator operator++()
            # bint operator==(iterator)
            # bint operator!=(iterator)
            unsigned long long row()
            unsigned long long col()
            double& operator*()

        # iterators
        const_iterator begin()
        const_iterator end()

    cdef cppclass field[T]:
        field() nogil
        field(int n_elem)
        int n_rows
        int n_cols
        int n_elem
        T& operator[](int) nogil


## TODO: refactor to return pointer, but other function derefed.

##### Tools to convert numpy arrays to armadillo arrays ######
@cython.boundscheck(False)
cdef field[sp_dmat] list_to_sp_dmat_field(list values):
    cdef field[sp_dmat] f = field[sp_dmat](len(values))
    for i, value in enumerate(values):
        if not isinstance(value, csc_matrix) or value.ndim != 2 or not np.isrealobj(value):
            raise TypeError(f"expected each value in values to be a 2D real csc_matrix, but got {value}")
        f[i] = numpy_to_sp_dmat_d(value)
    return f


@cython.boundscheck(False)
cdef field[dvec] list_to_dvec_field(list values):
    cdef field[dvec] f = field[dvec](len(values))
    for i, value in enumerate(values):
        if not isinstance(value, np.ndarray) or value.ndim != 1 or not np.isrealobj(value):
            raise TypeError(f"expected each value in values to be a 1D real numpy matrix, but got {value}")
        f[i] = numpy_to_dvec_d(value)
    return f


cdef dmat * numpy_to_dmat(np.ndarray[np.double_t, ndim=2] X):
    # TODO: Add checks on X (Size, ...)
    # TODO: Add flags for new dmat. See: Advanced constructors in http://arma.sourceforge.net/docs.html#Mat
    #  mat(ptr_aux_mem, n_rows, n_cols, copy_aux_mem = true, strict = false)
    # TODO: raise Warning on copy or replication of data
    if not X.flags.f_contiguous:
        X = X.copy(order="F")
    cdef dmat *aR_p  = new dmat(<double*> X.data, X.shape[0], X.shape[1], False, True)
    return aR_p

cdef dmat numpy_to_dmat_d(np.ndarray[np.double_t, ndim=2] X):
    cdef dmat * aR_p = numpy_to_dmat(X)
    cdef dmat aR = deref(aR_p)
    del aR_p
    return aR

cdef dvec * numpy_to_dvec(np.ndarray[np.double_t, ndim=1] x):
    if not (x.flags.f_contiguous or x.flags.owndata):
        x = x.copy()
    cdef dvec *ar_p = new dvec(<double*> x.data, x.shape[0], False, True)
    return ar_p

cdef dvec numpy_to_dvec_d(np.ndarray[np.double_t, ndim=1] x):
    cdef dvec *ar_p = numpy_to_dvec(x)
    cdef dvec ar = deref(ar_p)
    del ar_p
    return ar

cdef uvec * numpy_to_uvec(np.ndarray[np.uint64_t, ndim=1] x):
    if not (x.flags.f_contiguous or x.flags.owndata):
        x = x.copy()
    cdef uvec *ar_p = new uvec(<unsigned long long *> x.data, x.shape[0], False, True)
    return ar_p

cdef uvec numpy_to_uvec_d(np.ndarray[np.uint64_t, ndim=1] x):
    cdef uvec *ar_p = numpy_to_uvec(x)
    cdef uvec ar = deref(ar_p)
    del ar_p
    return ar


cdef sp_dmat * numpy_to_sp_dmat(x):
    if not isinstance(x, csc_matrix):
        raise ValueError(f"expected x to be of type {csc_matrix}, but got {type(x)}")

    cdef np.ndarray[np.double_t, ndim=1] data = x.data
    cdef dvec values = numpy_to_dvec_d(data)

    cdef np.ndarray[np.uint64_t, ndim=1] indptr = x.indptr.astype(np.uint64)
    cdef uvec colptr = numpy_to_uvec_d(indptr)

    cdef np.ndarray[np.uint64_t, ndim=1] indices = x.indices.astype(np.uint64)
    cdef uvec rowind = numpy_to_uvec_d(indices)

    cdef sp_dmat * ar = new sp_dmat(rowind, colptr, values, x.shape[0], x.shape[1])

    return ar


cdef sp_dmat numpy_to_sp_dmat_d(x):
    cdef sp_dmat *ar_p = numpy_to_sp_dmat(x)
    cdef sp_dmat ar = deref(ar_p)
    del ar_p
    return ar

##### Converting back to l0learn arrays, must pass preallocated memory or None
# all data will be copied since numpy doesn't own the data and can't clean up
# otherwise. Maybe this can be improved. #######


@cython.boundscheck(False)
cdef list sp_dmat_field_to_list(field[sp_dmat] f):
    cdef list lst = []
    for i in range(f.n_elem):
        lst.append(sp_dmat_to_numpy(f[i], None, None, None))
    return lst


@cython.boundscheck(False)
cdef list dvec_field_to_list(field[dvec] f):
    cdef list lst = []
    for i in range(f.n_elem):
        lst.append(dvec_to_numpy(f[i], None))
    return lst

@cython.boundscheck(False)
cdef np.ndarray[np.double_t, ndim=2] dmat_to_numpy(const dmat & X, np.ndarray[np.double_t, ndim=2] D):
    # TODO: Check order of X and D
    cdef const double * Xptr = X.memptr()

    if D is None:
        D = np.empty((X.n_rows, X.n_cols), dtype=np.double, order="F")
    cdef double * Dptr = <double*> D.data
    for i in range(X.n_rows*X.n_cols):
        Dptr[i] = Xptr[i]
    return D

@cython.boundscheck(False)
cdef np.ndarray[np.double_t, ndim=1] dvec_to_numpy(const dvec & X, np.ndarray[np.double_t, ndim=1] D):
    cdef const double * Xptr = X.memptr()

    if D is None:
        D = np.empty(X.n_elem, dtype=np.double)
    cdef double * Dptr = <double*> D.data
    for i in range(X.n_elem):
        Dptr[i] = Xptr[i]
    return D


@cython.boundscheck(False)
cdef np.ndarray[np.uint64_t, ndim=1] uvec_to_numpy(const uvec & X, np.ndarray[np.uint64_t, ndim=1] D):
    cdef const unsigned long long * Xptr = X.memptr()

    if D is None:
        D = np.empty(X.n_elem, dtype=np.uint64)
    cdef unsigned long long * Dptr = <unsigned long long *> D.data
    for i in range(X.n_elem):
        Dptr[i] = Xptr[i]
    return D


@cython.boundscheck(False)
cdef sp_dmat_to_numpy(const sp_dmat & X, np.ndarray[np.uint64_t, ndim=1] rowind,
                       np.ndarray[np.uint64_t, ndim=1] colind, np.ndarray[np.double_t, ndim=1] values):
    # TODO: Check order of X and D
    if rowind is None:
        rowind = np.empty(X.n_nonzero, dtype=np.uint64)
    if colind is None:
        colind = np.empty(X.n_nonzero, dtype=np.uint64)
    if values is None:
        values = np.empty(X.n_nonzero, dtype=np.double)

    cdef sp_dmat.const_iterator it = X.begin()

    for i in range(X.n_nonzero):
        # TODO: Double check the following comment...
        # Arma is column major so rows are columns and columns are rows when converting back to l0learn (row major)
        rowind[i] = it.col()
        colind[i] = it.row()
        values[i] = deref(it)
        inc(it)

    return csc_matrix((values, (colind, rowind)), shape=(X.n_rows, X.n_cols))
