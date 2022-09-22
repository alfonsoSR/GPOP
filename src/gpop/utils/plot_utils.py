import matplotlib.pyplot as plt
from gpop.pck.pck import moon_data
import gpop.utils.utils as ut
import numpy as np
import numpy.typing as npt

ndarray = npt.NDArray[np.float64]

MU_MOON, R_MOON = moon_data()


def orbit_3D(*orbits: ndarray) -> None:
    """3D representation of an orbit.

    Input
    ------
    `orbit` : ndarray((n, 6))
        Array defining the kinematic state of the satellite at different
        epochs.

        Each state consists on six cartesian components of the position
        and velocity vectors: [x, y, z, u, v, w]
    """
    fig = plt.figure(facecolor=("#292c33"))

    ax = fig.add_subplot(projection="3d", proj_type='ortho')

    ax.set_facecolor('#292c33')

    ax.axis('off')

    for orbit in orbits:
        ax.plot(orbit[0], orbit[1], orbit[2])

    ax.set_box_aspect(
        np.ptp([
            ax.get_xlim(),
            ax.get_ylim(),
            ax.get_zlim()  # type: ignore
        ], axis=1)  # type: ignore
    )

    plt.show()

    return None


def satellite_altitude(t: ndarray, orbit: ndarray) -> None:
    """Graphical representation of the evolution with time of the
    satellite's altitude with respect to mean lunar radius.

    Input
    ------
    `t` : ndarray((n,))
        Epochs when the satellite's state is known.

    `orbit` : ndarray((n, 6))
        Array defining the kinematic state of the satellite at different
        epochs.

        Each state consists on six cartesian components of the position
        and velocity vectors: [x, y, z, u, v, w]
    """

    h = np.sqrt(np.sum(((orbit*orbit)[:3]), axis=0)) - R_MOON

    t = (t - t[0])/(24.*3600.)

    fig, ax = plt.subplots()

    ax.plot(t, h)

    ax.set_xlabel("Days since initial epoch")
    ax.set_ylabel("Height above lunar surface [km]")
    ax.set_title("Evolution of satellite's altitude")

    plt.show()

    return None


def position_vector(*states: tuple) -> None:
    '''Time-variation of the position vector components

    Input
    -----

    `states`: tuple, *
        Each tuple is expected to have two elements: t, and orbit.
        `t` : ndarray
            Epochs when the satellite's state is known.

        `orbit` : ndarray
            Array defining the kinematic state of the satellite at different
            epochs.

            Each state consists on six cartesian components of the position
            and velocity vectors: [x, y, z, u, v, w]'''

    fig, ax = plt.subplots(3, 1, sharex=True)

    labels = ["$x - x_{ref}$", "$y - y_{ref}$", "$z - z_{ref}$"]

    for idx, label in enumerate(labels):

        ax[idx].set_ylabel(label)  # type: ignore

    for t, orbit in states:

        t = (t - t[0])/(24.*3600.)

        for ax_i, r_i, label in zip(ax, orbit[:3], labels):  # type: ignore

            ax_i.plot(t, r_i, label=label)

    ax[-1].set_xlabel("Days since initial epoch")  # type: ignore

    plt.show()

    return None


def velocity_vector(*states: tuple) -> None:
    '''Time-variation of the velocity vector components

    Input
    -----

    `states`: tuple, *
        Each tuple is expected to have two elements: t, and orbit.
        `t` : ndarray
            Epochs when the satellite's state is known.

        `orbit` : ndarray
            Array defining the kinematic state of the satellite at different
            epochs.

            Each state consists on six cartesian components of the position
            and velocity vectors: [x, y, z, u, v, w]'''

    fig, ax = plt.subplots(3, 1, sharex=True)

    labels = ["$u - u_{ref}$", "$v - v_{ref}$", "$w - w_{ref}$"]

    for idx, label in enumerate(labels):

        ax[idx].set_ylabel(label)  # type: ignore

    for t, orbit in states:

        t = (t - t[0])/(24.*3600.)

        for ax_i, r_i, label in zip(ax, orbit[:3], labels):  # type: ignore

            ax_i.plot(t, r_i, label=label)

    ax[-1].set_xlabel("Days since initial epoch")  # type: ignore

    plt.show()

    return None


def error(t: ndarray, orbit: ndarray, ref: ndarray) -> None:

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(11, 5))

    labels = [r"$1 - \dfrac{r}{r_{ref}}$", r"$1 - \dfrac{v}{v_{ref}}$"]

    for idx, label in enumerate(labels):

        ax[idx].set_ylabel(label)  # type: ignore

    t = (t - t[0])/(24.*3600.)

    dr, dv = ut.rel_error(orbit, ref)

    # diff = orbit - ref

    # dr = np.sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2])
    # dv = np.sqrt(diff[3]*diff[3] + diff[4]*diff[4] + diff[5]*diff[5])

    ax[0].plot(t, dr)  # type: ignore
    ax[1].plot(t, dv)  # type: ignore

    ax[-1].set_xlabel("Days since initial epoch")  # type: ignore

    plt.show()

    return None


def compute_periapsis(orbit: ndarray) -> ndarray:
    """Compute the radius of periapsis from array of states

    Input
    ------
    `orbit` : ndarray((n, 6))
        Array defining the kinematic state of the satellite at different
        epochs.

        Each state consists on six cartesian components of the position
        and velocity vectors: [x, y, z, u, v, w]

    Output
    ------
    `r_p` : ndarray((n,))
        Periapsis' radius of the osculating orbit fitting each state.

    """

    r = np.sqrt(np.sum(((orbit*orbit)[:3]), axis=0))
    v = np.sqrt(np.sum(((orbit*orbit)[3:]), axis=0))

    h_vec = np.array([
        orbit[1]*orbit[5] - orbit[2]*orbit[4],
        orbit[2]*orbit[3] - orbit[0]*orbit[5],
        orbit[0]*orbit[4] - orbit[1]*orbit[3]
    ])

    h = np.sqrt(np.sum(h_vec*h_vec, axis=0))

    p = h*h/MU_MOON

    e = np.sqrt(
        1. + p*(v*v/MU_MOON - 2./r)
    )

    return p/(1. + e)


def periapsis(*cases) -> None:
    """Graphical representation of the evolution with time of the
    periapsis' radius.

    Input (OLD)
    ------
    `t` : ndarray((n,))
        Epochs when the satellite's state is known.

    `orbits` : tuple(ndarray((n, 6)))
        Tuple of arrays defining the kinematic state of each satellite at
        different epochs.

        Each state consists on six cartesian components of the position
        and velocity vectors: [x, y, z, u, v, w]

    `show` : bool, optional
        Automatically show the plot. Default is True.
    """

    fig, ax = plt.subplots()

    for case in cases:

        t, s = case

        r_p = compute_periapsis(s)

        t = (t - t[0]) / (24. * 3600.)

        ax.plot(t, r_p)

    ax.set_xlabel("Days since initial epoch")
    ax.set_ylabel("Radius of periapsis [km]")
    ax.set_title("Radius of periapsis of the osculating orbit")

    plt.show()

    return None
