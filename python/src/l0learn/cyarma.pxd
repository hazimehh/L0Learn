# distutils: language = c++

from libcpp cimport bool
cimport numpy as np

cdef extern from "armadillo" namespace "arma" nogil:

    # matrix class (double)
    cdef cppclass dmat:
        dmat(double * aux_mem, int n_rows, int n_cols, bool copy_aux_mem, bool strict) nogil
        dmat(dmat x)
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
        int n_nonzero
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
        # const_iterator end()

    cdef cppclass field[T]:
        field() nogil
        field(int n_elem)
        int n_rows
        int n_cols
        int n_elem
        T& operator[](int) nogil

cdef field[sp_dmat] list_to_sp_dmat_field(list values)

cdef field[dvec] list_to_dvec_field(list values)

cdef dmat * numpy_to_dmat(np.ndarray[np.double_t, ndim=2] X)

cdef dmat numpy_to_dmat_d(np.ndarray[np.double_t, ndim=2] X)

# cdef const dmat numpy_to_const_dmat_d(np.ndarray[np.double_t, ndim=2] X)

cdef dvec * numpy_to_dvec(np.ndarray[np.double_t, ndim=1] x)

cdef dvec numpy_to_dvec_d(np.ndarray[np.double_t, ndim=1] x)

cdef uvec * numpy_to_uvec(np.ndarray[np.uint64_t, ndim=1] x)

cdef uvec numpy_to_uvec_d(np.ndarray[np.uint64_t, ndim=1] x)

cdef sp_dmat * numpy_to_sp_dmat(x)

cdef sp_dmat numpy_to_sp_dmat_d(x)

cdef list sp_dmat_field_to_list(field[sp_dmat] f)

cdef list dvec_field_to_list(field[dvec] f)

cdef np.ndarray[np.double_t, ndim=2] dmat_to_numpy(const dmat & X, np.ndarray[np.double_t, ndim=2] D)

cdef np.ndarray[np.double_t, ndim=1] dvec_to_numpy(const dvec & X, np.ndarray[np.double_t, ndim=1] D)

cdef np.ndarray[np.uint64_t, ndim=1] uvec_to_numpy(const uvec & X, np.ndarray[np.uint64_t, ndim=1] D)

cdef sp_dmat_to_numpy(const sp_dmat & X, np.ndarray[np.uint64_t, ndim=1] rowind,
                       np.ndarray[np.uint64_t, ndim=1] colptr, np.ndarray[np.double_t, ndim=1] values)
