# cython: language_level = 3
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: initializedcheck = False

from gpop.utils.utils import Case, sqlite
from gpop.pck.pck cimport moon, earth, Body_list, Body
import numpy as np
from libc.stdlib cimport calloc, free
from libc.math cimport sqrt

cdef class motion_law:

    def __cinit__(self, case: Case):

        cdef:

            int i, j, d, o, n
            double c, s

        # Initialize acceleration array

        self.ds_array = np.zeros((6,))
        self.ds = self.ds_array

        # SPICE kernels

        self.metak_path = case.metak_path

        furnsh_c(self.metak_path)

        # Initialize perturbations

        if case.use_non_sphericity:

            self.use_non_sphericity = 1

            self.harmonics_db_path = case.non_sphericity.db_path
            self.limit = case.non_sphericity.max_deg

            n = self.limit + 1

            # Retrieve spherical harmonics coefficients from database

            self._Clm = <double*>calloc(n * n, sizeof(double))
            if self._Clm == NULL:
                raise MemoryError('Memory allocation failed for self._Clm')
            self.Clm = <double[:n, :n]>self._Clm

            self._Slm = <double*>calloc(n * n, sizeof(double))
            if self._Slm == NULL:
                raise MemoryError('Memory allocation failed for self._Slm')
            self.Slm = <double[:n, :n]>self._Slm

            with sqlite(self.harmonics_db_path) as db:

                db.execute(
                    """
                        select fk_deg, ord, Clm, Slm from ord
                        where fk_deg <= ?;
                    """, (self.limit,)
                )

                for d, o, c, s in db:

                    self.Clm[d, o] = c
                    self.Slm[d, o] = s

            # Calculate normalization parameters

            self.norm_n1 = <double*>calloc(self.limit, sizeof(double))
            if self.norm_n1 == NULL:
                raise MemoryError('Memory allocation failed for self.norm_n1')

            self.norm_n2 = <double*>calloc(self.limit, sizeof(double))
            if self.norm_n2 == NULL:
                raise MemoryError('Memory allocation failed for self.norm_n2')

            self.norm_n1_n1 = <double*>calloc(self.limit, sizeof(double))
            if self.norm_n1_n1 == NULL:
                raise MemoryError(
                    'Memory allocation failed for self.norm_n1_n1'
                )

            self._norm_n1_m = <double*>calloc(self.limit * n, sizeof(double))
            if self._norm_n1_m == NULL:
                raise MemoryError(
                    'Memory allocation failed for self._norm_n1_m'
                )
            self.norm_n1_m = <double[:self.limit, :n]>self._norm_n1_m

            self._norm_n2_m = <double*>calloc(self.limit * n, sizeof(double))
            if self._norm_n2_m == NULL:
                raise MemoryError(
                    'Memory allocation failed for self._norm_n2_m'
                )
            self.norm_n2_m = <double[:self.limit, :n]>self._norm_n2_m

            for i in range(2, self.limit + 1):

                self.norm_n1[i - 1] = sqrt((2. * i + 1.) / (2. * i - 1.))
                self.norm_n2[i - 1] = sqrt((2. * i + 1.) / (2. * i - 3.))
                self.norm_n1_n1[i - 1] = (
                    sqrt((2. * i + 1.) / (2. * i)) / (2. * i - 1.)
                )

                for j in range(1, i + 1):

                    self.norm_n1_m[i - 1, j - 1] = sqrt(
                        (i - j) * (2. * i + 1.) / ((i + j) * (2. * i - 1.))
                    )

                    self.norm_n2_m[i - 1, j - 1] = sqrt(
                        (i - j) * (i - j - 1.) * (2. * i + 1.) /
                        ((i + j) * (i + j - 1.) * (2. * i - 3.))
                    )

        if case.use_third_body:

            self.use_third_body = 1
            
            for i in range(11):

                self.use_body_list[i] = case.third_body.body_list[i]
                self.body_list[i] = Body_list[i]

            # Speed of light

            self.c = clight_c()

        if case.use_custom_perturbations:

            self.use_custom = 1

            self.custom = case.custom()

        if case.use_solar_radiation_pressure:

            raise NotImplementedError

    cdef void main_body(self, double r_vec[3]):

        cdef:

            int idx

            double r

        r = sqrt(
            r_vec[0] * r_vec[0] +
            r_vec[1] * r_vec[1] +
            r_vec[2] * r_vec[2]
        )

        for idx in range(3):

            self.ds[idx + 3] += - moon.mu * r_vec[idx] / (r * r * r)

    cdef void harmonics(
        self, const int max_deg, double t, double r_vec[3]
    ):
        """ Acceleration due to non-sphericity of the Moon.

        Input
        ------
        `max_deg` : const int

            Truncation degree for the spherical harmonics expansion of the
            lunar gravity field.

        `t` : double

            Epoch at which the acceleration is to be computed.

        `r_vec` : double [3]

            Satellite's position vector with respect to SCRF at given epoch.
        """

        cdef:

            int idx, n_idx, m_idx, nm1_idx, nm2_idx, mm1_idx
            double e1, r2, r, r_cos_phi, sin_phi, cos_phi, root3, root5
            double n, m, nm1, nm2, e2, e3, e4, e5

            double R_icrf2pa[3][3]
            double r_i[3]
            double cos_m_lambda[max_deg + 1]
            double sin_m_lambda[max_deg + 1]
            double R_r[max_deg + 1]
            double Pn[max_deg + 1]
            double dPn[max_deg + 1]
            double z_partials[3]
            double xy_partials[3]
            double ddr[3]

            double sec_Pnm[max_deg + 1][max_deg + 1]
            double cos_dPnm[max_deg + 1][max_deg + 1]

        # Initialize sec_Pnm and cos_dPnm

        for idx in range(max_deg + 1):
            for n_idx in range(max_deg + 1):
                sec_Pnm[idx][n_idx] = 0.
                cos_dPnm[idx][n_idx] = 0.

        # Compute position vector in PA reference frame

        pxform_c("J2000", "MOON_PA_DE421", t, R_icrf2pa)

        for idx in range(3):
            r_i[idx] = (
                R_icrf2pa[idx][0] * r_vec[0] +
                R_icrf2pa[idx][1] * r_vec[1] +
                R_icrf2pa[idx][2] * r_vec[2]
            )

        # Compute spherical coordinates: r, lat, lon

        e1 = r_i[0] * r_i[0] + r_i[1] * r_i[1]

        r2 = e1 + r_i[2] * r_i[2]

        r = sqrt(r2)

        r_cos_phi = sqrt(e1)

        sin_phi = r_i[2] / r

        cos_phi = r_cos_phi / r

        cos_m_lambda[0] = 1.
        sin_m_lambda[0] = 0.

        if r_cos_phi != 0:
            sin_m_lambda[1] = r_i[1] / r_cos_phi
            cos_m_lambda[1] = r_i[0] / r_cos_phi

        R_r[0] = 1.
        R_r[1] = moon.R / r

        # Initialize normalised associated Legendre functions (Lear algorithm)

        root3 = sqrt(3.)
        root5 = sqrt(5.)

        Pn[0] = 1.
        Pn[1] = root3 * sin_phi

        dPn[0] = 0.
        dPn[1] = root3

        sec_Pnm[1][1] = root3

        cos_dPnm[1][1] = -root3 * sin_phi

        # Normalized associated Legendre functions

        if max_deg >= 2:

            for n_idx in range(2, max_deg + 1):

                n = n_idx
                nm1 = n - 1.
                nm2 = n - 2.

                nm1_idx = n_idx - 1
                nm2_idx = n_idx - 2

                R_r[n_idx] = R_r[nm1_idx] * R_r[1]

                sin_m_lambda[n_idx] = (
                  2. * cos_m_lambda[1] * sin_m_lambda[nm1_idx]
                  - sin_m_lambda[nm2_idx]
                )
                cos_m_lambda[n_idx] = (
                  2. * cos_m_lambda[1] * cos_m_lambda[nm1_idx]
                  - cos_m_lambda[nm2_idx]
                )

                e1 = 2. * n - 1.

                Pn[n_idx] = (
                    e1 * sin_phi * self.norm_n1[nm1_idx] * Pn[nm1_idx]
                    - nm1 * self.norm_n2[nm1_idx] * Pn[nm2_idx]
                ) / n

                dPn[n_idx] = self.norm_n1[nm1_idx] * (
                    sin_phi * dPn[nm1_idx] + n * Pn[nm1_idx]
                )

                sec_Pnm[n_idx][n_idx] = (
                    e1 * cos_phi * self.norm_n1_n1[nm1_idx]
                    * sec_Pnm[nm1_idx][nm1_idx]
                )

                cos_dPnm[n_idx][n_idx] = -sin_phi * n * sec_Pnm[n_idx][n_idx]

                e1 = e1 * sin_phi
                e2 = -sin_phi * n

                for m_idx in range(1, n_idx):

                    m = m_idx
                    mm1_idx = m_idx - 1

                    e3 = (
                        self.norm_n1_m[nm1_idx, m_idx - 1]
                        * sec_Pnm[nm1_idx][m_idx]
                    )
                    e4 = n + m
                    e5 = (
                        e1 * e3
                        - (e4 - 1.) * self.norm_n2_m[nm1_idx, m_idx - 1]
                        * sec_Pnm[n_idx - 2][m_idx]
                    ) / (n - m)

                    sec_Pnm[n_idx][m_idx] = e5

                    cos_dPnm[n_idx][m_idx] = e2 * e5 + e3 * e4


        # Though Pn[0] should be 1, I prefere to separate the effect of the
        # Moon as a punctual mass from the rest of the accelerations

        Pn[0] = 0.

        # Acceleration with respect to PA reference frame

        z_partials[0] = -sin_phi * cos_m_lambda[1]
        z_partials[1] = -sin_phi * sin_m_lambda[1]
        z_partials[2] = cos_phi

        xy_partials[0] = -sin_m_lambda[1]
        xy_partials[1] = cos_m_lambda[1]
        xy_partials[2] = 0.

        for idx in range(3):

            # ddr[idx] = - moon.mu * r_i[idx] / (r * r2)
            ddr[idx] = 0.

        if max_deg >= 2:

            for n_idx in range(2, max_deg + 1):

                n = n_idx

                for idx in range(3):

                    ddr[idx] += (moon.mu / r2) * R_r[n_idx] * self.Clm[n_idx, 0] * (
                        (-r_i[idx] / r) * (n + 1.) * Pn[n_idx]
                        + z_partials[idx] * cos_phi * dPn[n_idx]
                    )

                for m_idx in range(1, n_idx + 1):

                    m = m_idx

                    for idx in range(3):

                        ddr[idx] += (moon.mu / r2) * R_r[n_idx] * (
                            (
                                cos_m_lambda[m_idx] * self.Clm[n_idx, m_idx]
                                + sin_m_lambda[m_idx] * self.Slm[n_idx, m_idx]
                            ) * (
                                (-r_i[idx] / r) * (n + 1.) * cos_phi
                                * sec_Pnm[n_idx][m_idx]
                                + z_partials[idx] * cos_dPnm[n_idx][m_idx]
                            ) + xy_partials[idx] * sec_Pnm[n_idx][m_idx] * m *
                            (
                                cos_m_lambda[m_idx] * self.Slm[n_idx, m_idx]
                                - sin_m_lambda[m_idx] * self.Clm[n_idx, m_idx]
                            )
                        )

        # Compute acceleration in SCRF

        for idx in range(3):

            self.ds[3 + idx] += (
                R_icrf2pa[0][idx] * ddr[0] +
                R_icrf2pa[1][idx] * ddr[1] +
                R_icrf2pa[2][idx] * ddr[2]

            )

    cdef void third_body(
        self, double t, double r_vec[3], Body body
    ):
        """ Acceleration due to third body perturbation.

        Input
        -----
        `t` : double
            
            Epoch at which the acceleration is to be computed.

        `r_vec` : double [3]

            Satellite's position vector with respect to SCRF at given epoch.

        `id` : int

            Third body NAIF ID code. (https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html)

        `mu` : double
            
            Standard gravitational parameter of the third body [km^3/s^2]
        """
        cdef:
            double X_moon[3]
            double X_sat[3]
            double d_moon, d_sat, d_moon3, d_sat3
            int i

        spkgps_c(moon.id, t, "J2000", body.id, X_moon, &d_moon)
        
        d_moon = d_moon * self.c
        d_moon3 = d_moon * d_moon * d_moon

        for i in range(3):
            X_sat[i] = X_moon[i] + r_vec[i]

        d_sat = sqrt(
            X_sat[0] * X_sat[0] +
            X_sat[1] * X_sat[1] +
            X_sat[2] * X_sat[2]
        )

        d_sat3 = d_sat * d_sat * d_sat

        for i in range(3):
            self.ds[i + 3] += - body.mu * (
                (X_sat[i] / d_sat3) - (X_moon[i] / d_moon3)
            )

    def __call__(self, double t, double [:] s):

        cdef:
            
            double r_vec[3]
            int idx

        for idx in range(3):

            r_vec[idx] = s[idx]
            self.ds[idx] = s[idx + 3]
            self.ds[idx + 3] = 0.

        self.main_body(r_vec)

        if self.use_non_sphericity:

            self.harmonics(self.limit, t, r_vec)

        if self.use_third_body:

            for idx in range(11):

                if self.use_body_list[idx]:

                    self.third_body(t, r_vec, self.body_list[idx])

        if self.use_custom:

            self.custom.perturb(t, s, self.ds)

        return self.ds_array

    def __dealloc__(self):

        kclear_c()

        if self._Clm is not NULL:
            free(self._Clm)

        if self._Slm is not NULL:
            free(self._Slm)

        if self.norm_n1 is not NULL:
            free(self.norm_n1)
        
        if self.norm_n2 is not NULL:
            free(self.norm_n2)
        
        if self.norm_n1_n1 is not NULL:
            free(self.norm_n1_n1)

        if self._norm_n1_m is not NULL:
            free(self._norm_n1_m)
        
        if self._norm_n2_m is not NULL:
            free(self._norm_n2_m)

        print("Memory deallocation completed")
        
        
        

