# cython: language_level = 3
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: initializedcheck = False

from gpop.pck.pck cimport Body
from gpop.utils.utils import Case
import numpy as np
cimport numpy as cnp
from gpop.core.custom cimport Perturbation

cdef extern from "SpiceUsr.h":

    ctypedef char SpiceChar
    ctypedef double SpiceDouble
    ctypedef float SpiceFloat
    ctypedef int SpiceInt
    ctypedef const char ConstSpiceChar
    ctypedef const double ConstSpiceDouble
    ctypedef const float ConstSpiceFloat
    ctypedef const int ConstSpiceInt

    cdef void furnsh_c(ConstSpiceChar * file)
    cdef void kclear_c()
    cdef void str2et_c(ConstSpiceChar * epoch, SpiceDouble * et)
    cdef void pxform_c(
        ConstSpiceChar * old_frame,
        ConstSpiceChar * new_frame,
        SpiceDouble epoch,
        SpiceDouble matrix[3][3]
    )
    cdef void mxv_c(
        ConstSpiceDouble matrix[3][3],
        ConstSpiceDouble old_r[3],
        SpiceDouble new_r[3]
    )
    cdef void spkpos_c(
        ConstSpiceChar * target,
        SpiceDouble epoch,
        ConstSpiceChar * frame,
        ConstSpiceChar * abcorr,
        ConstSpiceChar * obs,
        SpiceDouble X_obs_targ[3],
        SpiceDouble * light_time
    )
    cdef double clight_c()
    cdef void spkgps_c(
        SpiceInt target,
        SpiceDouble epoch,
        ConstSpiceChar * frame,
        SpiceInt obs,
        SpiceDouble X_obs_targ[3],
        SpiceDouble * light_time
    )


cdef class motion_law:

    cdef:

        # Common

        cnp.ndarray ds_array
        double [:] ds

        # Third body

        bint use_third_body
        char * metak_path
        bint use_body_list[11]
        Body body_list[11]

        double c

        # Non-sphericity

        bint use_non_sphericity
        str harmonics_db_path
        int limit

        double * _Clm
        double [:, ::1] Clm

        double * _Slm
        double [:, ::1] Slm

        double * norm_n1
        double * norm_n2
        double * norm_n1_n1
        
        double * _norm_n1_m
        double [:, ::1] norm_n1_m

        double * _norm_n2_m
        double [:, ::1] norm_n2_m

        # Custom

        bint use_custom
        Perturbation custom

    cdef void main_body(self, double r_vec[3])

    cdef void harmonics(self, const int max_deg, double t, double r_vec[3])

    cdef void third_body(self, double t, double r_vec[3], Body body)
