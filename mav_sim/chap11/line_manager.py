"""Provides an implementation of the straight line path manager for waypoint following as described in
   Chapter 11 Algorithm 7
"""

from typing import cast

import numpy as np
from mav_sim.chap11.path_manager_utilities import (
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


def line_manager(state: MsgState, waypoints: MsgWaypoints, ptr_prv: WaypointIndices,
                 path_prv: MsgPath, hs_prv: HalfSpaceParams) \
                -> tuple[MsgPath, HalfSpaceParams, WaypointIndices]:
    """Update for the line manager. Only updates the path and next halfspace under two conditions:
        1) The waypoints are new
        2) In a new halfspace

    Args:
        state: current state of the vehicle
        waypoints: The waypoints to be followed
        ptr_prv: The indices of the waypoints being used for the previous path
        path_prv: The previously commanded path
        hs_prv: The currently active halfspace for switching

    Returns:
        path: The updated path to follow
        hs: The updated halfspace for the next switch
        ptr: The updated index pointer
    """
    # Default the outputs to be the inputs
    path = path_prv
    hs = hs_prv
    ptr = ptr_prv

    # Create manager here

    pos = np.array([
        [state.north],
        [state.east],
        [-state.altitude]
    ])

    if inHalfSpace(pos, hs):
        ptr.increment_pointers(waypoints.num_waypoints)
        path, hs = construct_line(waypoints=waypoints, ptr=ptr)

    # Output the updated path, halfspace, and index pointer
    return (path, hs, ptr)

def construct_line(waypoints: MsgWaypoints, ptr: WaypointIndices) \
    -> tuple[MsgPath, HalfSpaceParams]:
    """Creates a line and switching halfspace. The halfspace assumes that the aggregate
       path will consist of a series of straight lines.

    The line is created from the previous and current waypoints with halfspace defined for
    switching once the current waypoint is reached.

    Args:
        waypoints: The waypoints from which to construct the path
        ptr: The indices of the waypoints being used for the path

    Returns:
        path: The straight-line path to be followed
        hs: The halfspace for switching to the next waypoint
    """

    # Extract the waypoints (w_{i-1}, w_i, w_{i+1})
    (previous, current, next_wp) = extract_waypoints(waypoints=waypoints, ptr=ptr)


    line_direction = current - previous
    line_direction /= np.linalg.norm(line_direction)

    q_i_minus_1 = line_direction
    q_i = next_wp - current
    q_i /= np.linalg.norm(q_i)

    type = "line"

    airspeed = get_airspeed(waypoints, ptr)

    line_origin = previous

    # Construct the path
    path = MsgPath(
        type=type,
        plot_updated=False,
        airspeed=airspeed,
        line_direction=line_direction,
        line_origin=line_origin
    )
    path.plot_updated = False

    norm_i = (q_i_minus_1 + q_i) / np.linalg.norm(q_i_minus_1 + q_i)

    # Construct the halfspace
    hs = HalfSpaceParams(
        point=current,
        normal=norm_i
    )
    return (path, hs)
