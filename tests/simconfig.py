from gpop.utils.utils import (
    Case, config_non_sphericity, config_third_body, config_J2, config_C22
)
import numpy as np

# Non sphericity configuration

frozen = np.array([
    [3507.1, 0.4987, 331.8424, 89.9961, 269.9405],
    [2509.9, 0.2996, 359.7549, 89.9952, 269.8961],
    [2196.8, 0.1998, 359.9059, 89.9887, 270.3698],
    [1849.9, 0.0497, 359.3479, 89.9997, 270.0220],
    [1774.7, 0.0094, 359.8686, 90.0022, 270.4520],
    [1759.8, 0.0009, 155.3850, 90.1015, 270.8632] 
])

db_source = ("https://pds-geosciences.wustl.edu/grail/"
             "grail-l-lgrs-5-rdr-v1/grail_1001/shadr/gggrx_0900c_sha.tab")

info_harmonics = config_non_sphericity(
    db_root="data/harmonics",
    db_name="grgm900c",
    db_source=db_source,
    max_deg=400
)

# Third body configuration

grail_kernels = [
    ("https://naif.jpl.nasa.gov/pub/naif/pds/data/grail-l-spice-6-v1.0/"
     "grlsp_1000/data/lsk/naif0010.tls"),
    ("https://naif.jpl.nasa.gov/pub/naif/pds/data/grail-l-spice-6-v1.0/"
     "grlsp_1000/data/pck/pck00009.tpc"),
    ("https://naif.jpl.nasa.gov/pub/naif/pds/data/grail-l-spice-6-v1.0/"
     "grlsp_1000/data/spk/de421.bsp"),
    ("https://naif.jpl.nasa.gov/pub/naif/pds/data/grail-l-spice-6-v1.0/"
     "grlsp_1000/data/fk/moon_080317.tf"),
    ("https://naif.jpl.nasa.gov/pub/naif/pds/data/grail-l-spice-6-v1.0/"
     "grlsp_1000/data/pck/moon_pa_de421_1900_2050.bpc"),
    ("https://naif.jpl.nasa.gov/pub/naif/pds/data/grail-l-spice-6-v1.0"
     "/grlsp_1000/data/spk/grail_120301_120529_sci_v02.bsp")
]

info_third = config_third_body(
    body_list=["earth"]
)

info_J2 = config_J2(
    db_root="data/harmonics",
    db_name="grgm900c",
    db_source=db_source
)

info_C22 = config_C22(
    db_root="data/harmonics",
    db_name="grgm900c",
    db_source=db_source
)

# Simulation configuration

test = Case(
    central_body="moon",
    kernels_root="data/grail",
    kernels_list=grail_kernels,
    use_non_sphericity=False,
    non_sphericity=info_harmonics,
    use_third_body=True,
    third_body=info_third,
    use_solar_radiation_pressure=False,
    use_custom_perturbations=False,
    use_J2=True,
    J2=info_J2,
    use_C22=True,
    C22=info_C22,
    rtol=5e-10,
    atol=5e-10,
    initial_epoch="2012-03-06 00:00:00",
    tspan=40.,
    days=True,
    initial_state=frozen[4],
    cartesian=False,
)
