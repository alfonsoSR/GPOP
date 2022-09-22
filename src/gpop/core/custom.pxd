# cython: language_level = 3
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False

cdef class Perturbation:

    cdef void perturb(self, double t, double [:] state, double [:] ds)
