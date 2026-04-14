"""Provides an implementation of the fillet path manager for waypoint following as described in
   Chapter 11 Algorithm 8
"""

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

    pos = np.array([
        [state.north],
        [state.east],
        [-state.altitude]
    ])

    (previous, current, next_wp) = extract_waypoints(waypoints=waypoints, ptr=ptr)
    # handle case when path folds back on itself
    dir1 = (current - previous) / np.linalg.norm(current - previous)
    dir2 = (next_wp - current) / np.linalg.norm(next_wp - current)
    if (dir1 + dir2).sum() == 0:
        manager_state = 1

    if manager_state == 1:
        if np.linalg.norm(pos - current) < 1:
            manager_state = 2

    # Insert code here
    if manager_state == 1:
        path, hs = construct_fillet_line(waypoints, ptr, radius)
        if inHalfSpace(pos, hs):
            manager_state += 1

    if manager_state == 2:
        path, hs = construct_fillet_circle(waypoints, ptr, radius)
        if inHalfSpace(pos, hs):
            ptr.increment_pointers(waypoints.num_waypoints)
            manager_state -= 1

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

    q_i_minus_one = current - previous
    q_i_minus_one_norm = np.linalg.norm(q_i_minus_one)
    q_i_minus_one /= q_i_minus_one_norm
    
    q_i = next_wp - current
    q_i_norm = np.linalg.norm(next_wp - current)
    q_i /= q_i_norm

    res = np.clip(-q_i_minus_one.T@q_i, -1, 1)
    rho = np.arccos(res).item()

    if np.tan(rho/2) < EPSILON:
        r_1 = current
    else:
        r_1 = current - (radius / np.tan(rho/2))*q_i_minus_one

    n_i = q_i_minus_one

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

    q_i_minus_one = current - previous
    q_i_minus_one_norm = np.linalg.norm(q_i_minus_one)
    q_i_minus_one /= q_i_minus_one_norm
    
    q_i = next_wp - current
    q_i_norm = np.linalg.norm(next_wp - current)
    q_i /= q_i_norm

    rho = np.arccos(-q_i_minus_one.T@q_i)

    orbit_dir = np.cross(q_i_minus_one.flatten(), q_i.flatten())
    _lambda = 1 if orbit_dir[2] > 0 else -1

    J = np.array([
        [0,1,0],
        [-1,0,0],
        [0,0,1],
    ])

    if (q_i - q_i_minus_one).sum() == 0 or rho/2 < EPSILON:
        c_i = current + _lambda*J@q_i_minus_one * radius
    else:
        c_i = current + ((-q_i_minus_one +  q_i)/(np.linalg.norm(-q_i_minus_one + q_i))) * (radius / np.sin(rho/2))

    if rho/2 < EPSILON:
        r_2 = current
    else:
        r_2 = current + (radius / np.tan(rho/2))*q_i

    n_i = q_i
    # Construct the path
    path = MsgPath()
    path.plot_updated = False
    path.orbit_center = c_i
    path.orbit_radius = radius
    path.orbit_direction = 'CW' if orbit_dir[2] >= 0 else 'CCW' 
    path.airspeed = get_airspeed(waypoints, ptr)
    path.type = "orbit"

    # Define the switching halfspace
    hs = HalfSpaceParams()
    hs.point = r_2
    hs.normal = n_i

    return (path, hs)
