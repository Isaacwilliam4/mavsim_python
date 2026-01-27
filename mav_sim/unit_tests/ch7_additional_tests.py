"""This module provides some additional tests for chapter 7 that can be used to help you
   debug if you are not passing the unit tests
"""

import numpy as np
from mav_sim.chap7.mav_dynamics import magnetometer


def magnetometer_test() -> None:
    """ This function seemingly does nothing. However, if you run it in the debug mode, you should be able to check the function for intermediary values.
    """

    # Create the input to the magnetometer
    quat_b_to_i_non_unit = np.array([[1.], [2.], [3.], [4.]])
    quat_b_to_i = quat_b_to_i_non_unit / np.linalg.norm(quat_b_to_i_non_unit)

    # Call the magnetometer
    # magnetometer in the inertial frame: [[ 0.39709536], [-0.21643961], [-0.89189078]]
    # magnetometer in the body frame: [[-0.70632024], [-0.70733881], [ 0.02799142]]
    # mag_x, mag_y, mag_z should be different from body frame due to noise
    mag_x, mag_y, mag_z = magnetometer(quat_b_to_i=quat_b_to_i)
    print(mag_x, mag_y, mag_z)


    # Create the input to the magnetometer
    quat_b_to_i_non_unit = np.array([[0.1], [1.1], [0.2], [1.4]])
    quat_b_to_i = quat_b_to_i_non_unit / np.linalg.norm(quat_b_to_i_non_unit)

    # Call the magnetometer
    # magnetometer in the inertial frame: [[ 0.39709536], [-0.21643961], [-0.89189078]]
    # magnetometer in the body frame: [[-0.98662077], [ 0.01340126], [ 0.16248034]]
    # mag_x, mag_y, mag_z should be different from body frame due to noise
    mag_x, mag_y, mag_z = magnetometer(quat_b_to_i=quat_b_to_i)
    print(mag_x, mag_y, mag_z)

if __name__ == "__main__":
    magnetometer_test()
