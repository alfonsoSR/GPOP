# cython: language_level = 3
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False

from libc.math cimport sin, cos, sqrt, atan2, M_PI as pi
from gpop.pck.pck cimport moon
import numpy as np

cdef class ENRKE:

    def __init__(self, const double ec, const double tol = 3.0e-15):
        """ENRKE class constructor

        Input
        -----
        `ec` : const double

            Eccentricity of the keplerian orbit.

        `tol` : const double, optional

            Method's accuracy.
            Default is `3.0e-15`.
        """

        self.ec = ec
        self.tol = tol

        self.PI = 3.1415926535897932385
        self.TWOPI = 6.2831853071795864779

    cdef void ENRKE_evaluate(self, double M, double * Eout):
        """Compute eccentric anomaly from mean anomaly.

        Input
        ------
        `M` : double

            Value of the mean anomaly.

        `Eout` : double *

            Pointer to the value of the eccentric anomaly.
        """

        cdef:

            double delta, Eapp, flip, f, fp, fpp, fppp, fp3, ffpfpp, f2fppp
            double Mr
            double tol2s = 2. * self.tol / (self.ec + 2.2e-16)
            double al = self.tol / 1.0e7
            double be = self.tol / 0.3

        # Fit angle in range (0, 2pi) if needed

        Mr = M % self.TWOPI

        if Mr > self.PI:

            Mr = self.TWOPI - Mr
            flip = 1.

        else:

            flip = -1.

        if (self.ec > 0.99 and Mr < 0.0045):

            fp = 2.7 * Mr
            fpp = 0.301
            f = 0.154

            while (fpp - fp > (al + be * f)):

                if (f - self.ec * sin(f) - Mr) > 0.:

                    fpp = f

                else:

                    fp = f

                f = 0.5 * (fp + fpp)

            Eout[0] = (M + flip * (Mr - f)) % self.TWOPI

        else:

            Eapp = Mr + 0.999999 * Mr * (self.PI - Mr) / (
                2. * Mr + self.ec - self.PI +
                2.4674011002723395 / (self.ec + 2.2e-16)
            )

            fpp = self.ec * sin(Eapp)
            fppp = self.ec * cos(Eapp)
            fp = 1. - fppp
            f = Eapp - fpp - Mr

            delta = - f / fp

            fp3 = fp * fp * fp
            ffpfpp = f * fp * fpp
            f2fppp = f * f * fppp

            delta = delta * (fp3 - 0.5 * ffpfpp + f2fppp / 3.) / (
                fp3 - ffpfpp + 0.5 * f2fppp
            )

            while (delta * delta > fp * tol2s):
                Eapp += delta
                fp = 1. - self.ec * cos(Eapp)
                delta = (Mr - Eapp + self.ec * sin(Eapp)) / fp

            Eapp += delta

            Eout[0] = (M + flip * (Mr - Eapp)) % self.TWOPI


cdef class Kepler(ENRKE):

    def __init__(
        self, const double a, const double e, const double Omega,
        const double i, const double omega, const double t0
    ):
        """Kepler extension type constructor.

        Input
        -----
        `a` : const double

            Semimajor axis [ km ].

        `e` : const double

            Eccentricity.

        `Omega` : const double

            Right ascension of the ascending node [ rad ].

        `i` : const double

            Inclination angle [ rad ].

        `omega` : const double

            Argument of periapsis [ rad ].

        `t0` : const double

            Initial epoch.
        """

        self.a = a
        self.e = e
        self.Omega = Omega
        self.i = i
        self.omega = omega
        self.t0 = t0

        # ----- Initialize ENRKE class ----- #

        super().__init__(self.e)

        # ----- Orbital period ----- #

        self.T = 2. * pi * sqrt(self.a * self.a * self.a / moon.mu)

        # ----- Recurrent parameters ----- #

        self.sqrt_ec = sqrt((1. + self.e) / (1. - self.e))

        self.state_array = np.zeros((6,), dtype=np.float64)
        self.state_memview = self.state_array

    cdef void time2nu(self, double t, double * nu):
        """Compute epoch from true anomaly.

        Input
        -----
        `t` : double

            Epoch in which true anomaly is to be computed.

        `nu` : double *

            Pointer to the value of the true anomaly.
        """

        cdef double M = 2. * pi * (t - self.t0) / self.T
        cdef double E

        self.ENRKE_evaluate(M, &E)

        nu[0] = 2. * atan2(self.sqrt_ec * sin(0.5 * E), cos(0.5 * E))

    cdef void _nu2state(self, double nu, double[:] state):
        """Calculate cartesian state from true anomaly.

        Input
        ------
        `nu` : double

            True anomaly [ rad ].

        `state` : double[:]

            Memory view where the value where the new state is to be stored.
        """

        cdef:
            double omega_nu, cos_omega, sin_omega, cos_omega_nu
            double sin_omega_nu, cos_i, sin_i, cos_Omega, sin_Omega, e_e, r_mod
            double u_mod, f_cos, f_sin, sin_omega_nu_cos_i, f_cos_cos_i

        omega_nu = self.omega + nu

        cos_omega = cos(self.omega)
        sin_omega = sin(self.omega)

        cos_omega_nu = cos(omega_nu)
        sin_omega_nu = sin(omega_nu)

        cos_i = cos(self.i)
        sin_i = sin(self.i)

        cos_Omega = cos(self.Omega)
        sin_Omega = sin(self.Omega)

        e_e = self.e * self.e

        r_mod = self.a * (1. - e_e) / (1. + self.e * cos(nu))

        u_mod = sqrt(moon.mu / (self.a * (1. - e_e)))

        f_cos = cos_omega_nu + self.e * cos_omega
        f_sin = sin_omega_nu + self.e * sin_omega

        sin_omega_nu_cos_i = sin_omega_nu * cos_i
        f_cos_cos_i = f_cos * cos_i

        state[0] = r_mod * (cos_omega_nu * cos_Omega
                            - sin_omega_nu_cos_i * sin_Omega)
        state[1] = r_mod * (cos_omega_nu * sin_Omega
                            + sin_omega_nu_cos_i * cos_Omega)
        state[2] = r_mod * sin_omega_nu * sin_i
        state[3] = u_mod * (-f_cos_cos_i * sin_Omega - f_sin * cos_Omega)
        state[4] = u_mod * (f_cos_cos_i * cos_Omega - f_sin * sin_Omega)
        state[5] = u_mod * f_cos * sin_i

    def nu2state(self, double nu):
        """Python interface for the _nu2state method.

        Input
        ------
        `nu` : double

            True anomaly [ rad ]
        """

        state = np.zeros((6,), dtype=np.float64)

        self._nu2state(nu, state)

        return state

    def __call__(self, double t):
        """State of the ideal satellite at given epoch.

        Input
        -----
        `t` : double

            Epoch at which the ideal state of the satellite is to be computed.

        Output
        -------
        NumPy array with the cartesian state of the ideal satellite.
        """

        self.time2nu(t, &self.NU)
        self._nu2state(self.NU, self.state_memview)

        return self.state_array
