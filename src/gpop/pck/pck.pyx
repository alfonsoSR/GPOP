# cython: language_level = 3
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False

moon = Body("moon", 301, 4902.79996708864, 1738.0)
sun = Body("sun", 10, 132712440041.279419, 0.)
mercury = Body("mercury", 199, 22031.868551, 0.)
venus = Body("venus", 299, 324858.592000, 0.)
earth = Body("earth", 399, 398600.435507, 0.)
mars = Body("mars", 4, 42828.375816, 0.)
jupiter = Body("jupiter", 5, 126712764.100000, 0.)
saturn = Body("saturn", 6, 37940584.841800, 0.)
uranus = Body("uranus", 7, 5794556.400000, 0.)
neptune = Body("neptune", 8, 6836527.100580, 0.)
pluto = Body("pluto", 9, 975.500000, 0.)

Body_list[0] = sun
Body_list[1] = mercury
Body_list[2] = venus
Body_list[3] = earth
Body_list[4] = moon
Body_list[5] = mars
Body_list[6] = jupiter
Body_list[7] = saturn
Body_list[8] = uranus
Body_list[9] = neptune
Body_list[10] = pluto

bodies = {
  "moon" : moon,
  "sun" : sun,
  "mercury" : mercury,
  "venus" : venus,
  "earth" : earth,
  "mars" : mars,
  "jupiter" : jupiter,
  "saturn" : saturn,
  "uranus" : uranus,
  "neptune" : neptune,
  "pluto" : pluto
}

def moon_data():
  
  return moon.mu, moon.R

def make_accessible(planet: str):

  if planet in bodies:

    return (
      bodies[planet]["id"],
      bodies[planet]["mu"],
      bodies[planet]["R"]
    )

  else:

    raise ValueError("Unknow celestial body")
