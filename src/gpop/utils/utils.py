from dataclasses import dataclass, field
import os
import sqlite3
import spiceypy as spice
import numpy as np
import numpy.typing as npt
from gpop.core.custom import Perturbation
from gpop.pck.pck import moon_data

# Type alias for Numpy arrays

ndarray = npt.NDArray[np.float64]

# ----- Case configuration ----- #

MU_MOON, R_MOON = moon_data()


@dataclass
class config_non_sphericity:

    # Initialization indicator

    is_usable: bool = field(init=False)

    # Spherical harmonics' coefficients database

    db_root: str = ""
    db_name: str = ""
    db_source: str = ""
    db_path: str = field(init=False)

    # Spherical harmonics expansion

    max_deg: int = 0

    # Check for initialization

    def __post_init__(self):

        if self.db_root != "":
            self.is_usable = True
            self.db_path = f"{self.db_root}/{self.db_name}.db"
        else:
            self.is_usable = False
            self.db_path = ""


@dataclass
class config_third_body:

    # Initialization indicator

    is_usable: bool = field(init=False)

    # List of bodies to use

    body_list: list = field(default_factory=list)

    # Check for initialization

    def __post_init__(self):

        if len(self.body_list) != 0:
            self.is_usable = True
        else:
            self.is_usable = False


@dataclass
class config_J2:

    # Initialization indicator

    is_usable: bool = field(init=False)

    # Spherical harmonics' coefficients database

    db_root: str = ""
    db_name: str = ""
    db_source: str = ""
    db_path: str = field(init=False)

    # Check for initialization

    def __post_init__(self):

        if self.db_root != "":

            self.is_usable = True
            self.db_path = f"{self.db_root}/{self.db_name}.db"

        else:

            self.is_usable = False
            self.db_path = ""


@dataclass
class config_C22:

    # Initialization indicator

    is_usable: bool = field(init=False)

    # Spherical harmonics' coefficients database

    db_root: str = ""
    db_name: str = ""
    db_source: str = ""
    db_path: str = field(init=False)

    # Check for initialization

    def __post_init__(self):

        if self.db_root != "":

            self.is_usable = True
            self.db_path = f"{self.db_root}/{self.db_name}.db"

        else:

            self.is_usable = False
            self.db_path = ""


@dataclass(kw_only=True)
class Case:

    # ----- Choose what central body to use ----- #

    central_body: str

    # ----- Choose what perturbations to use ----- #

    use_non_sphericity: bool
    use_third_body: bool
    use_solar_radiation_pressure: bool
    use_custom_perturbations: bool
    use_J2: bool
    use_C22: bool

    # Choose what solver to use

    rtol: float
    atol: float

    # Choose what SPICE kernels to use

    kernels_root: str
    kernels_list: list[str]
    metak_path: bytes = field(init=False)

    # Simulation parameters

    initial_epoch: str
    tspan: float
    days: bool
    initial_state: np.ndarray
    cartesian: bool
    nu: float = None  # type: ignore

    # Perturbations' configuration

    non_sphericity: config_non_sphericity = config_non_sphericity()
    third_body: config_third_body = config_third_body()
    custom: type = Perturbation
    J2: config_J2 = config_J2()
    C22: config_C22 = config_C22()

    def __post_init__(self):

        self.metak_path = bytes(
            f"{self.kernels_root}/metak", "utf-8"
        )


# ----- Initialisation utilities ----- #

def case_info(case: Case) -> None:

    print("#########################################\n")
    print(f"Initial epoch:\t\t\t\t{case.initial_epoch}")

    if case.days:

        print(f"Time span:\t\t\t\t{case.tspan:0.0f} days")

    else:

        print(f"Time span: {(case.tspan / (24. * 3600.)):0.4f} days")

    print("Perturbations:")

    if case.use_non_sphericity:

        print("\tNon-sphericity")
        print(f"\t\tGravity field model:\t{case.non_sphericity.db_name}")
        print(f"\t\tTruncation degree:\t{case.non_sphericity.max_deg}")

    if case.use_third_body:

        print("\tThird body")
        print(f"\t\tBody list:\t\t{case.third_body.body_list}")

    if case.use_J2:

        print("\tJ2 Zonal harmonic")
        print(f"\t\tGravity field model:\t{case.J2.db_name}")

    if case.use_C22:

        print("\tC22 Sectoral harmonic")
        print(f"\t\tGravity field model:\t{case.C22.db_name}")

    if case.use_custom_perturbations:

        print("\tCustom perturbation")

    print("\n#########################################")


class sqlite:
    """Context manager for SQLite3 databases
  
    Input
    -----
    `file_name` : str
      
        Name of the database (with extension) to be used.
    """

    def __init__(self, file_name: str) -> None:

        self.file_name = file_name
        self.connection = sqlite3.connect(self.file_name)

        return None

    def __enter__(self):

        return self.connection.cursor()

    def __exit__(self, exc_type, exc_value, traceback) -> None:

        self.connection.commit()
        self.connection.close()

        return None


class assure_harmonics():

    def __init__(self, dir: str, coeff_source: str, db_name: str) -> None:
        """
        Input
        ------
            `dir` : str
                Relative path to the directory in which the database
                should be stored.
                
            `coeff_source` : str
                Link to the webpage with the coefficients' file.
                
            `db_name` : str
                Name of the database without extension.
        """

        # Create root directory if it does not exist

        if not os.path.isdir(dir):

            os.system(f"mkdir {dir}")

        # Download coefficients if needed

        self.source = f"{dir}/{os.path.basename(coeff_source)}"

        if not os.path.isfile(self.source):

            print(f"Downloading {db_name}")

            os.system(
                f"curl -# -o {self.source} {coeff_source}"
            )

        # Create database if needed

        self.path = f"{dir}/{db_name}.db"

        if not os.path.isfile(self.path):

            print(self.path)

            self.create_database()

        return None

    def create_database(self) -> None:

        with sqlite(self.path) as db:

            db.execute("""
                create table deg (deg int not null, primary key (deg));
            """)

            db.execute("""
                create table ord (
                    id integer primary key autoincrement,
                    fk_deg int not null,
                    ord int not null,
                    Clm double not null,
                    Slm double not null,
                    foreign key(fk_deg) references deg(deg)
                );
            """)

            with open(self.source) as source:

                rows = source.readlines()[1:]

                for row in rows:

                    _d, _o, _C, _S = row.split(",")[:4]

                    d = int(_d)
                    o = int(_o)
                    C = float(_C)
                    S = float(_S)

                    try:

                        db.execute(
                            """
                                insert into deg (deg) values (?);
                            """, (d,)
                        )

                    except sqlite3.IntegrityError:

                        pass

                    db.execute(
                        """
                            insert into ord (id, fk_deg, ord, Clm, Slm)
                            values (NULL, ?, ?, ?, ?);
                        """, (d, o, C, S)
                    )

        return None


class Kernels:

    def __init__(self, root: str) -> None:
        """Context manager for SPICE kernels
        
        Input
        -----
            `root` : str
                Relative path to the directory to which SPICE kernels
                should be saved.
        """

        self.root = root
        spice.furnsh(f"{self.root}/metak")

    def __enter__(self) -> None:

        return None

    def __exit__(self, exc_type, exc_value, traceback):

        spice.kclear()


def get_kernels(root: str, kernels: list) -> None:

    # Check if root directory exists and create it otherwise

    if not os.path.isdir(root):

        os.system(f"mkdir {root}")

    # Check if kernels' directory exists and create it otherwise

    kpath = f"{root}/kernels"

    if not os.path.isdir(kpath):

        os.system(f"mkdir {kpath}")

    # Check if all kernels are available and download the missing ones

    with open(f"{root}/metak", "w") as metak:

        metak.write(r"\begindata" + "\nKERNELS_TO_LOAD=(\n")

        for idx, kernel in enumerate(kernels):

            file = os.path.basename(kernel)

            if not os.path.exists(f"{kpath}/{file}"):

                print(f"Downloading {file}")

                os.system(f"curl -# -o {kpath}/{file} {kernel}")

            metak.write(f"'{root}/kernels/{file}'")

            if idx != len(kernels):

                metak.write(",\n")

        metak.write(')\n'+r'\begintext'+'\n')

    return None


def state2rv(state: np.ndarray) -> tuple:
    """Get radius and velocity modulus from state vector

    Input
    ------
    `state` : ndarray
        Satellite's state in a series of epochs.

    Output
    ------
    `r` : ndarray
        Orbital radius at each epoch

    `v` : ndarray
        Orbital velocity at each epoch
    """

    r = np.sqrt(state[0]*state[0] + state[1]*state[1] + state[2]*state[2])
    v = np.sqrt(state[3]*state[3] + state[4]*state[4] + state[5]*state[5])

    return r, v


def rel_error(orbit: np.ndarray, ref: np.ndarray) -> tuple:
    """Position and velocity relative errors with respect to
    a reference orbit.

    Input
    -----
    `orbit` : ndarray
        State vectors' array for the orbit of interest.

    `ref`   : ndarray
        State vectors' array for the orbit to be used as reference

    Output
    ------
    `dr`    : ndarray
        Relative position error for each epoch.

    `dv`    : ndarray
        Relative velocity error for each epoch.
    """

    r, v = state2rv(orbit)

    r_ref, v_ref = state2rv(ref)

    dr = (r_ref - r)/r_ref
    dv = (v_ref - v)/v_ref

    return dr, dv


# def grail_orbit(t: ndarray, case: Case) -> ndarray:

#     with Kernels(case.metak_path):

#         grail = np.swapaxes(
#             spice.spkezr(
#                 "GRAIL-A", t, "J2000", "NONE", "moon"
#             )[0], 0, 1  # type: ignore
#         )

#     return grail


def save_results(t: ndarray, s: ndarray, file: str) -> None:

    sol = np.concatenate((t[None, :], s))

    np.save(file, sol)

    print(f"Results saved to {file}.npy")


def state2orbital(s: ndarray):

    r_i = s[:3]
    u_i = s[3:]

    r, v = state2rv(s)

    v_r = (r_i[0] * u_i[0] + r_i[1] * u_i[1] + r_i[2] * u_i[2]) / r

    h_i = np.cross(r_i, u_i, axis=0)

    h = np.sqrt(h_i[0] * h_i[0] + h_i[1] * h_i[1] + h_i[2] * h_i[2])

    W_i = h_i / h

    i = np.arccos(W_i[2]) * 180. / np.pi

    e_k_i = np.array([[0., 0., 1.]]).swapaxes(0, 1)

    N_i = np.cross(e_k_i, h_i, axis=0)

    N = np.sqrt(N_i[0] * N_i[0] + N_i[1] * N_i[1] + N_i[2] * N_i[2])

    M = (N_i[1] < 0.) * 1
    GE = (N_i[1] >= 0.) * 1

    Omega = 360. * M + (np.arccos(N_i[0] / N) * 180. * (GE - M) / np.pi)

    e_i = ((v * v - MU_MOON / r) * r_i - r * v_r * u_i) / MU_MOON

    e = np.sqrt(e_i[0] * e_i[0] + e_i[1] * e_i[1] + e_i[2] * e_i[2])

    P_i = e_i / e

    M = (P_i[2] < 0.) * 1
    GE = (P_i[2] >= 0.) * 1

    PN_N = (P_i[0] * N_i[0] + P_i[1] * N_i[1] + P_i[2] * N_i[2]) / N

    omega = 360. * M + (np.arccos(PN_N) * 180. * (GE - M) / np.pi)

    # M = (v_r < 0.) * 1
    # GE = (v_r >= 0.) * 1

    # Pr_r = (P_i[0] * r_i[0] + P_i[1] * r_i[1] + P_i[2] * r_i[2]) / r

    # nu = 360. * M + (np.arccos(Pr_r) * 180. * (GE - M) / np.pi)

    a = h * h / (MU_MOON * (1. - e * e))

    return np.array([a, e, Omega, i, omega])
