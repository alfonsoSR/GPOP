from gpop.core.main import Simulation
from simconfig import test
from gpop.utils.plot_utils import orbital_elements

if __name__ == "__main__":

    # Third body of earth

    earth = Simulation(test)

    t, s = earth.propagate_case()

    orbital_elements(t, s)
