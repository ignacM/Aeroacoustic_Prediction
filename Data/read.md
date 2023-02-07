Calculations in read.md
Azimuthal Array:
 microphone 12: y = 0.11931, x = 0, z = -0.0025

set all z as -0.0025

To calculate x for every microphone: r*cos(theta-5)
To calculate y for every microphone: r*sin(theta-5)


Polar array:

y can be calculated by copying results from y (azi)
set all x = 0
z_max is calculated by using x_max(azi)*2 = 1744.794


combined_data has been created with the following assumptions:
- Azimuthal array:
  - middle (top) microphone is positioned 119.31mm vertical direction from blades
  - microphones and blades centres are 2.5mm away
- Polar array:
  - middle (top) microphone is positioned 119.31mm vertical direction from blades
  - First microphone is exactly on top of the blade centre axis.


