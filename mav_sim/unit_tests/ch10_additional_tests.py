"""This module provides some additional tests for chapter 10 that can be used to help you
   debug if you are not passing the unit tests
"""

import numpy as np
from mav_sim.chap10.path_follower import follow_orbit, follow_straight_line
from mav_sim.message_types.msg_path import MsgPath
from mav_sim.message_types.msg_state import MsgState


def follow_straight_line_helper() -> None:
    """ This function seemingly does nothing. However, if you run it in the debug mode, you should be able to check the function for intermediary values.
    """

    # Initialize the inputs
    path = MsgPath()                # Generated path to follow
    path.type = 'line'
    path.line_origin = np.array([[0.0, 0.0, -100.0]]).T
    path.line_direction = np.array([[0.5, 1.0, 0.0]]).T
    path.line_direction = path.line_direction / np.linalg.norm(path.line_direction)
    state = MsgState()              # Current state of the aircraft
    state.north = .25
    state.east = 3.
    state.altitude = 100.
    state.phi = 0.002
    state.theta = 0.001
    state.psi = 0.
    state.Va = 25.
    state.alpha = 0.004
    state.beta = 0.0002
    state.p = 0.5
    state.q = 0.29
    state.r = 0.02
    state.Vg = 25.04
    state.gamma = 0.003
    state.chi = 0.0002
    state.wn = 0.001
    state.we = -0.002
    state.bx = 0.
    state.by = 0.
    state.bz = 0.
    k_path = 0.01               # The gain on the path convergence
    chi_inf = 0.87              # The infinite distance course angle

    # Calculate the command
    #   autopilot_cmd:
    #       airspeed_command: 25,
    #       course_command: 1.1009566416609335,
    #       altitude_command: 100.0,
    #       phi_feedforward: 0.0
    #   chi_q = 1.1071487177940904 # Course angle of the path
    #   e_y = 1.1180339887498951 # Path error
    #   n = [[ 0.89442719], [-0.4472136 ], [ 0.        ]]  # Normal vector to the plan
    #   ep = [[0.25], [3.  ], [0.  ]]# Error vector
    #   s = [[1.25], [2.5 ],[0.  ]]# The projection of the error onto the plane
    autopilot_cmd = follow_straight_line(path=path, state=state, k_path=k_path, chi_inf=chi_inf)
    print(autopilot_cmd)

    # Update the problem
    state.altitude = 110.
    state.psi = 3.*np.pi
    state.chi = 3.*np.pi + 0.0002
    np.put(path.line_direction, 1, -0.5)
    np.put(path.line_direction, 2, 0.5)
    path.line_direction = path.line_direction / np.linalg.norm(path.line_direction)

    # Calculate the command
    #   autopilot_cmd:
    #       airspeed_command: 25,
    #       course_command: 11.71319463294801,
    #       altitude_command: 98.45755933208332,
    #       phi_feedforward: 0.0
    #
    #   chi_q =  11.725301943791242 # Course angle of the path
    #   e_y =  2.186338998124982 # Path error
    #   n =   [[-0.74535599],  [-0.66666667], [ 0.        ]]# Normal vector to the plan
    #   ep = [[  0.25], [  3.  ], [-10.  ]]# Error vector
    #   s = [[ -1.37960087], [  1.54244067], [-10.        ]]# The projection of the error onto the plane
    autopilot_cmd = follow_straight_line(path=path, state=state, k_path=k_path, chi_inf=chi_inf)
    print(autopilot_cmd)

def follow_orbit_helper() -> None:
    """ This function seemingly does nothing. However, if you run it in the debug mode, you should be able to check the function for intermediary values.
    """
    # Initialize the inputs
    path = MsgPath()                # Generated path to follow
    path.type = 'orbit'
    path.orbit_center = np.array([[10.0, 0.0, -100.0]]).T
    path.orbit_radius = 500.
    path.orbit_direction = "CW"
    path.airspeed = 35.
    state = MsgState()              # Current state of the aircraft
    state.north = .25
    state.east = 3.
    state.altitude = 100.
    state.phi = 0.002
    state.theta = 0.001
    state.psi = 0.
    state.Va = 25.
    state.alpha = 0.004
    state.beta = 0.0002
    state.p = 0.5
    state.q = 0.29
    state.r = 0.02
    state.Vg = 25.04
    state.gamma = 0.003
    state.chi = 0.0002
    state.wn = 0.001
    state.we = -0.002
    state.bx = 0.
    state.by = 0.
    state.bz = 0.
    k_orbit = 0.01               # The gain on the path convergence
    gravity = 9.81              # Gravity constant

    # Calculate the command
    #   autopilot_cmd:
    #       airspeed_command: 35.0,
    #       course_command: 4.404094384182648,
    #       altitude_command: 100.0,
    #       phi_feedforward: 0.24473879745624252
    #   d = 10.201102881551583 # Distance from the orbit center
    #   varphi = 2.843093722003614 # Phase angle of the relative position
    #   orbit_error = -0.9795977942368967 # Normalized orbit error (d-p)/p
    autopilot_cmd = follow_orbit(path=path, state=state, k_orbit=k_orbit, gravity=gravity)
    print(autopilot_cmd)


    # Update state
    state.altitude = 100.
    state.east = 10000000.
    state.psi = 3.*np.pi
    state.chi = 3.*np.pi + 0.0002
    path.orbit_direction = "CCW"


    # Calculate the command
    #   autopilot_cmd:
    #       airspeed_command: 35.0,
    #       course_command: 4.717390163724896,
    #       altitude_command: 100.0,
    #       phi_feedforward: 0.0
    #   d =  10000000.000004753 # Distance from the orbit center
    #   varphi = 7.853982608974483 # Phase angle of the relative position
    #   orbit_error = 19999.000000009506 # Normalized orbit error (d-p)/p
    autopilot_cmd = follow_orbit(path=path, state=state, k_orbit=k_orbit, gravity=gravity)
    print(autopilot_cmd)


if __name__ == "__main__":
    follow_straight_line_helper()
    #follow_orbit_helper()
