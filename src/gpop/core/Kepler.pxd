# cython: language_level = 3
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False

cimport numpy as cnp

cdef class ENRKE:

    cdef double tol, ec, PI, TWOPI

    cdef void ENRKE_evaluate(self, double M, double * Eout)


cdef class Kepler(ENRKE):

    cdef:
        double a, e, Omega, i, omega, t0, T, sqrt_ec, NU
        cnp.ndarray state_array
        double [:] state_memview

    cdef void time2nu(self, double t, double * nu)

    cdef void _nu2state(self, double nu, double[:] state)
