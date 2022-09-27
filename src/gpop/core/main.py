from gpop.utils.utils import (
    Case, case_info, assure_harmonics, get_kernels, Kernels
)
import sys
import spiceypy as spice
from datetime import datetime
from gpop.pck.pck import make_accessible
import numpy as np
import numpy.typing as npt
from gpop.core.Kepler import Kepler
from gpop.core.equations import motion_law
from gpop.old_solver.rkn1210 import rkn1210


# Type alias for numpy arrays

ndarray = npt.NDArray[np.float64]


class Simulation:
    """Base class for orbital propagation"""

    def __init__(self, case: Case):

        self.case = case

        # Display information about the case

        case_info(self.case)

        # ----- Central body properties ----- #

        if self.case.central_body == "moon":

            self.ID, self.MU, self.R = make_accessible(self.case.central_body)

        else:

            # Not all the code is adapted to other bodies yet

            raise NotImplementedError

        # ----- Basic perturbation initialization ----- #

        # SPICE kernels

        get_kernels(self.case.kernels_root, self.case.kernels_list)

        # Non sphericity

        if self.case.use_non_sphericity:

            if self.case.non_sphericity.is_usable:

                assure_harmonics(
                    self.case.non_sphericity.db_root,
                    self.case.non_sphericity.db_source,
                    self.case.non_sphericity.db_name
                )

            else:

                print(
                    "No data provided for non-sphericity perturbation. "
                    "Exit code: -1"
                )
                sys.exit(-1)

        # Third body

        if self.case.use_third_body:

            if self.case.third_body.is_usable:

                body_list = [0] * 11

                for body in self.case.third_body.body_list:

                    match body:

                        case "sun": body_list[0] = 1

                        case "mercury": body_list[1] = 1

                        case "venus": body_list[2] = 1

                        case "earth": body_list[3] = 1

                        case "moon": body_list[4] = 1

                        case "mars": body_list[5] = 1

                        case "jupiter": body_list[6] = 1

                        case "saturn": body_list[7] = 1

                        case "uranus": body_list[8] = 1

                        case "neptune": body_list[9] = 1

                        case "pluto": body_list[10] = 1

                        case _: raise ValueError("Unknown celestial body")

                self.case.third_body.body_list = body_list

            else:

                print(
                    "No data provided for third body perturbation. "
                    "Exit code: 0"
                )
                sys.exit(0)

        # J2

        if self.case.use_J2:

            if self.case.J2.is_usable:

                assure_harmonics(
                    self.case.J2.db_root,
                    self.case.J2.db_source,
                    self.case.J2.db_name
                )

            else:

                print(
                    "No data provided for J2 perturbation.\nExit code: -1"
                )
                sys.exit(-1)

        # C22

        if self.case.use_C22:

            if self.case.C22.is_usable:

                assure_harmonics(
                    self.case.C22.db_root,
                    self.case.C22.db_source,
                    self.case.C22.db_name
                )

            else:

                print(
                    "No data provided for C22 perturbation.\nExit code: -1"
                )
                sys.exit(-1)
        
        # Solar radiation pressure

        if self.case.use_solar_radiation_pressure:

            raise NotImplementedError

        # Custom perturbations

        if self.case.use_custom_perturbations:

            pass

            # raise NotImplementedError

        # ----- Simulation epochs ----- #

        # Convert time span to seconds if needed

        if self.case.days:
            
            self.tspan = self.case.tspan * 24. * 3600.

        else:

            self.tspan = self.case.tspan

        # Convert initial epoch to TDB

        with Kernels(self.case.kernels_root):

            try:

                # Checks if date is given in ISO format by calling
                # a function that fails otherwise (Dubious, but works).

                datetime.fromisoformat(self.case.initial_epoch)
                self.t0 = spice.str2et(self.case.initial_epoch)

            except ValueError:

                raise

        # ----- Initial state ----- #

        if self.case.cartesian:

            # Should add further checking on input

            self.y0 = self.case.initial_state

        else:

            self.check_orbital_elements()

            kepler = Kepler(
                self.a, self.e, self.Omega, self.i, self.omega, self.t0
            )

            self.y0 = kepler.nu2state(self.nu)

        # ----- Initialize solver ----- #

        self.solver = rkn1210()

    def check_orbital_elements(self, nu: float = 0.):

        a, e, Omega, i, omega = self.case.initial_state

        if self.case.nu is not None:

            nu = self.case.nu

        try:

            # Check if the orbit is not elliptical

            if 0. <= e < 1.:
                self.e = e
            else:
                raise ValueError("Eccentricity is higher or equal to 1")

            # Check if the satellite will crash against the central body

            if a * (1. - self.e) > self.R:
                self.a = a
            else:
                raise ValueError(
                    "The radius of the periapsis is smaller than the mean "
                    "radius of the central body."
                )

            # Check if all angles have valid values

            angle_list: ndarray = np.array([Omega, i, omega, nu])

            if all(angle_list >= 0.) and all(angle_list < 360.):
                self.Omega = np.deg2rad(Omega)
                self.i = np.deg2rad(i)
                self.omega = np.deg2rad(omega)
                self.nu = np.deg2rad(nu)
            else:
                raise ValueError(
                    "Orientation angle or true anomaly is out of range"
                )

        except ValueError:
            raise

        return None

    def propagate_case(self):

        with Kernels(self.case.kernels_root):

            natural_motion = motion_law(self.case)

            t, s = self.solver(
                natural_motion, self.t0, self.t0 + self.tspan, self.y0,
                rtol=self.case.rtol, atol=self.case.atol
            )

        return np.asarray(t), np.asarray(s).swapaxes(0, 1)
