"""Provides an implementation of the fillet path manager for waypoint following as described in
   Chapter 11 Algorithm 8
"""
from typing import cast

import numpy as np
from mav_sim.chap11.path_manager_utilities import (
    EPSILON,
    HalfSpaceParams,
    WaypointIndices,
    extract_waypoints,
    get_airspeed,
    inHalfSpace,
)
from mav_sim.message_types.msg_path import MsgPath
from mav_sim.message_types.msg_state import MsgState
from mav_sim.message_types.msg_waypoints import MsgWaypoints
from mav_sim.tools.types import NP_MAT


def fillet_manager(state: MsgState, waypoints: MsgWaypoints, ptr_prv: WaypointIndices,
                 path_prv: MsgPath, hs_prv: HalfSpaceParams, radius: float, manager_state: int) \
                -> tuple[MsgPath, HalfSpaceParams, WaypointIndices, int]:

    """Update for the fillet manager.
       Updates state machine if the MAV enters into the next halfspace.

    Args:
        state: current state of the vehicle
        waypoints: The waypoints to be followed
        ptr_prv: The indices that were being used on the previous iteration (i.e., current waypoint
                 inidices being followed when manager called)
        hs_prv: The previous halfspace being looked for (i.e., the current halfspace when manager called)
        radius: minimum radius circle for the mav
        manager_state: Integer state of the manager
                Value of 1 corresponds to following the straight line path
                Value of 2 corresponds to following the arc between straight lines

    Returns:
        path (MsgPath): Path to be followed
        hs (HalfSpaceParams): Half space parameters corresponding to the next change in state
        ptr (WaypointIndices): Indices of the current waypoint being followed
        manager_state (int): The current state of the manager

    """
    # Default the outputs to be the inputs
    path = path_prv
    hs = hs_prv
    ptr = ptr_prv

    # Insert code here

    return (path, hs, ptr, manager_state)

def construct_fillet_line(waypoints: MsgWaypoints, ptr: WaypointIndices, radius: float) \
    -> tuple[MsgPath, HalfSpaceParams]:
    """Define the line on a fillet and a halfspace for switching to the next fillet curve.

    The line is created from the previous and current waypoints with halfspace defined for
    switching once a circle of the specified radius can be used to transition to the next line segment.

    Args:
        waypoints: The waypoints to be followed
        ptr: The indices of the waypoints being used for the path
        radius: minimum radius circle for the mav

    Returns:
        path: The straight-line path to be followed
        hs: The halfspace for switching to the next waypoint
    """
    # Extract the waypoints (w_{i-1}, w_i, w_{i+1})
    (previous, current, next_wp) = extract_waypoints(waypoints=waypoints, ptr=ptr)

    q_i_minus_one = (current - previous)
    q_i_minus_one_norm = np.linalg.norm(q_i_minus_one)
    
    q_i = (next_wp - current)
    q_i_norm = np.linalg.norm(next_wp - current)

    rho = np.arccos(q_i_minus_one.T@q_i / (q_i_norm*q_i_minus_one_norm))
    r_1 = current - (radius / np.tan(rho/2))*q_i_minus_one
    n_i = (q_i_minus_one + q_i) / np.linalg.norm(q_i_minus_one + q_i)


    # Construct the path
    path = MsgPath()
    path.plot_updated = False
    path.airspeed = get_airspeed(waypoints, ptr)
    path.line_direction = q_i_minus_one
    path.line_origin = previous

    # Construct the halfspace
    hs = HalfSpaceParams()
    hs.point = r_1
    hs.normal = n_i

    return (path, hs)

def construct_fillet_circle(waypoints: MsgWaypoints, ptr: WaypointIndices, radius: float) \
    -> tuple[MsgPath, HalfSpaceParams]:
    """Define the circle on a fillet

    Args:
        waypoints: The waypoints to be followed
        ptr: The indices of the waypoints being used for the path
        radius: minimum radius circle for the mav

    Returns:
        path: The straight-line path to be followed
        hs: The halfspace for switching to the next waypoint
    """
    # Extract the waypoints (w_{i-1}, w_i, w_{i+1})
    (previous, current, next_wp) = extract_waypoints(waypoints=waypoints, ptr=ptr)

    q_i_minus_one = (current - previous)
    q_i_minus_one_norm = np.linalg.norm(q_i_minus_one)
    
    q_i = (next_wp - current)
    q_i_norm = np.linalg.norm(next_wp - current)

    rho = np.arccos(q_i_minus_one.T@q_i / (q_i_norm*q_i_minus_one_norm))
    c_i = current + ((-q_i_minus_one +  q_i)/(np.linalg.norm(-q_i_minus_one + q_i))) * (radius / np.sin(rho/2))

    orbit_dir = np.cross(q_i_minus_one.flatten(), q_i.flatten())
    orbit_direction = "CW" if orbit_dir >= 0 else "CCW" 

    r_2 = current + (radius / np.tan(rho/2))*q_i
    n_i = q_i

    # Construct the path
    path = MsgPath()
    path.plot_updated = False
    path.orbit_center = c_i
    path.orbit_radius = radius
    path.orbit_direction = orbit_direction

    # Define the switching halfspace
    hs = HalfSpaceParams()
    hs.point = r_2
    hs.normal = n_i

    return (path, hs)
