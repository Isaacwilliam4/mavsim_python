"""This module provides some additional tests for chapter 11 that can be used to help you
   debug if you are not passing the unit tests
"""

import numpy as np
from mav_sim.chap11.fillet_manager import construct_fillet_circle, construct_fillet_line
from mav_sim.chap11.line_manager import construct_line
from mav_sim.chap11.path_manager_utilities import WaypointIndices
from mav_sim.message_types.msg_waypoints import MsgWaypoints

# def line_manager(state: MsgState, waypoints: MsgWaypoints, ptr_prv: WaypointIndices,
#                  path_prv: MsgPath, hs_prv: HalfSpaceParams) \
#                 -> tuple[MsgPath, HalfSpaceParams, WaypointIndices]:
#     """Update for the line manager. Only updates the path and next halfspace under two conditions:
#         1) The waypoints are new
#         2) In a new halfspace

#     Args:
#         state: current state of the vehicle
#         waypoints: The waypoints to be followed
#         ptr_prv: The indices of the waypoints being used for the previous path
#         path_prv: The previously commanded path
#         hs_prv: The currently active halfspace for switching

#     Returns:
#         path: The updated path to follow
#         hs: The updated halfspace for the next switch
#         ptr: The updated index pointer
#     """
#     # Default the outputs to be the inputs
#     path = path_prv
#     hs = hs_prv
#     ptr = ptr_prv

#     # Extract the position
#     mav_pos = np.array([[state.north, state.east, -state.altitude]]).T

#     # if the waypoints have changed, update the waypoint pointer
#     if waypoints.flag_waypoints_changed is True:
#         waypoints.flag_waypoints_changed = False
#         ptr = WaypointIndices() # Resets the pointers - Line 2 of Algorithm 7

#         # Lines 4-7 of Algorithm 7
#         (path, hs) = construct_line(waypoints=waypoints, ptr=ptr)

#     # entered into the half plane separating waypoint segments
#     if inHalfSpace(mav_pos, hs):
#         ptr.increment_pointers(waypoints.num_waypoints)
#         (path, hs) = construct_line(waypoints=waypoints, ptr=ptr)

#     # Output the updated path, halfspace, and index pointer
#     return (path, hs, ptr)


# def fillet_manager(state: MsgState, waypoints: MsgWaypoints, ptr_prv: WaypointIndices,
#                  path_prv: MsgPath, hs_prv: HalfSpaceParams, radius: float, manager_state: int) \
#                 -> tuple[MsgPath, HalfSpaceParams, WaypointIndices, int]:

#     """Update for the fillet manager.
#        Updates state machine if the MAV enters into the next halfspace.

#     Args:
#         state: current state of the vehicle
#         waypoints: The waypoints to be followed
#         ptr_prv: The indices that were being used on the previous iteration (i.e., current waypoint
#                  inidices being followed when manager called)
#         hs_prv: The previous halfspace being looked for (i.e., the current halfspace when manager called)
#         radius: minimum radius circle for the mav
#         manager_state: Integer state of the manager
#                 Value of 1 corresponds to following the straight line path
#                 Value of 2 corresponds to following the arc between straight lines

#     Returns:
#         path (MsgPath): Path to be followed
#         hs (HalfSpaceParams): Half space parameters corresponding to the next change in state
#         ptr (WaypointIndices): Indices of the current waypoint being followed
#         manager_state (int): The current state of the manager

#     """
#     # Default the outputs to be the inputs
#     path = path_prv
#     hs = hs_prv
#     ptr = ptr_prv

#     # Extract the position
#     mav_pos = np.array([[state.north, state.east, -state.altitude]]).T

#     # if the waypoints have changed, update the waypoint pointer
#     # Lines 8-11 of Algorithm 8
#     if waypoints.flag_waypoints_changed is True:
#         waypoints.flag_waypoints_changed = False
#         ptr = WaypointIndices() # Resets the pointers
#         (path, hs) = \
#                 construct_fillet_line(waypoints=waypoints, ptr=ptr, radius=radius)
#         manager_state = 1

#     # state machine for fillet path
#     if manager_state == 1:
#         # follow straight line path from previous to current
#         # Lines 12-14 of Algorithm 8
#         if inHalfSpace(mav_pos, hs):
#             # Update to state 2 as described in Algorithm 8 line 13
#             manager_state = 2

#             # entered into the half plane H1 - construct the orbit
#             # Lines 17-20 of Algorithm 8
#             (path, hs) = \
#                 construct_fillet_circle(waypoints=waypoints, ptr=ptr, radius=radius)

#     if manager_state == 2: # Note that an elif is not used here as the aircraft should immediately switch to
#                            # the next segment when the path is a straight line
#         ### You need to add some logic here and remove the print statement ###
#         print("You need to do something here")

#     if manager_state < 1 or manager_state > 2:
#         raise ValueError("Invalid manager state")

#     return (path, hs, ptr, manager_state)


def construct_line_helper() -> None:
    """ This function seemingly does nothing. However, if you run it in the debug mode, you should be able to check the function for intermediary values.
    """
    # Create the waypoints to be followed
    waypoints = MsgWaypoints()
    waypoints.add(ned=np.array([[100.], [1.02], [1000.1]]), airspeed=25.)
    waypoints.add(ned=np.array([[500.], [2.03], [1001.1]]), airspeed=35.)
    waypoints.add(ned=np.array([[10.], [5.06], [1005.2]]), airspeed=25.)
    waypoints.add(ned=np.array([[1100.], [1.02], [1000.1]]), airspeed=25.)

    # Create the pointer
    ptr = WaypointIndices()

    # Construct a line
    #   q_i-1 = [[0.99999369],  [0.00252498], [0.00249998]]
    #   q_i = [[-0.99994588], [ 0.00618334], [ 0.00836689]]
    #   path:
    #       type = "line"
    #       plot_updated = False
    #       airspeed = 35.0
    #       line_origin = [[ 100.  ], [   1.02], [1000.1 ]]
    #       line_direction = [[0.99999369],  [0.00252498], [0.00249998]]
    #   hs:
    #       normal = [[0.00343307], [0.62534007], [0.7803448 ]]
    path1, hs1 = construct_line(waypoints=waypoints, ptr=ptr)
    print("Path1: ", path1)
    print("hs1: ", hs1)

    # Update the pointer
    ptr.increment_pointers(num_waypoints=waypoints.num_waypoints)

    # Construct a line
    #   q_i-1 = [[-0.99994588], [ 0.00618334], [ 0.00836689]]
    #   q_i = [[ 0.99998219], [-0.00370636], [-0.00467882]]
    #   path:
    #       type = "line"
    #       plot_updated = False
    #       airspeed = 25.
    #       line_origin = [[ 500.  ],  [   2.03], [1001.1 ]]
    #       line_direction = [[-0.99994588], [ 0.00618334], [ 0.00836689]]
    #   hs:
    #       normal = [[0.00817192], [0.55752442], [0.83012032]]
    path2, hs2 = construct_line(waypoints=waypoints, ptr=ptr)
    print("Path2: ", path2)
    print("hs2: ", hs2)

def construct_fillet_line_helper() -> None:
    """ This function seemingly does nothing. However, if you run it in the debug mode, you should be able to check the function for intermediary values.
    """
    # Create the waypoints to be followed
    waypoints = MsgWaypoints()
    waypoints.add(ned=np.array([[100.], [1.02], [1000.1]]), airspeed=25.)
    waypoints.add(ned=np.array([[500.], [2.03], [1001.1]]), airspeed=35.)
    waypoints.add(ned=np.array([[10.], [5.06], [1005.2]]), airspeed=25.)
    waypoints.add(ned=np.array([[1100.], [1.02], [1000.1]]), airspeed=25.)

    # Create the pointer
    ptr = WaypointIndices()

    # Construct a line
    #   q_i-1 = [[0.99999369],  [0.00252498], [0.00249998]]
    #   q_i = [[-0.99994588], [ 0.00618334], [ 0.00836689]]
    #   path:
    #       type = "line"
    #       plot_updated = False
    #       airspeed = 35.0
    #       line_origin = [[ 100.  ], [   1.02], [1000.1 ]]
    #       line_direction = [[0.99999369],  [0.00252498], [0.00249998]]
    #   hs:
    #       normal = [[0.99999369],  [0.00252498], [0.00249998]]
    #       point = [[-936.14557185], [  -1.59626757], [ 997.50963607]]
    path1, hs1 = construct_fillet_line(waypoints=waypoints, ptr=ptr, radius=10.)
    print("Path1: ", path1)
    print("hs1: ", hs1)

    # Update the pointer
    ptr.increment_pointers(num_waypoints=waypoints.num_waypoints)

    # Construct a line
    #   q_i-1 = [[-0.99994588], [ 0.00618334], [ 0.00836689]]
    #   q_i = [[ 0.99998219], [-0.00370636], [-0.00467882]]
    #   path:
    #       type = "line"
    #       plot_updated = False
    #       airspeed = 25.
    #       line_origin = [[ 500.  ],  [   2.03], [1001.1 ]]
    #       line_direction = [[-0.99994588], [ 0.00618334], [ 0.00836689]]
    #   hs:
    #       normal = [[-0.99994588],  [ 0.00618334],  [ 0.00836689]]
    #       point = [[4511.38668374], [ -22.77510541], [ 967.53533591]]
    path2, hs2 = construct_fillet_line(waypoints=waypoints, ptr=ptr, radius = 10.)
    print("Path2: ", path2)
    print("hs2: ", hs2)


def construct_fillet_circle_helper() -> None:
    """ This function seemingly does nothing. However, if you run it in the debug mode, you should be able to check the function for intermediary values.
    """
    # Create the waypoints to be followed
    waypoints = MsgWaypoints()
    waypoints.add(ned=np.array([[100.], [1.02], [1000.1]]), airspeed=25.)
    waypoints.add(ned=np.array([[500.], [2.03], [1001.1]]), airspeed=35.)
    waypoints.add(ned=np.array([[10.], [5.06], [1005.2]]), airspeed=25.)
    waypoints.add(ned=np.array([[1100.], [1.02], [1000.1]]), airspeed=25.)

    # Create the pointer
    ptr = WaypointIndices()

    # Construct a line
    #   q_i-1 = [[0.99999369],  [0.00252498], [0.00249998]]
    #   q_i = [[-0.99994588], [ 0.00618334], [ 0.00836689]]
    #   varrho = 0.01392585
    #   path:
    #       type = "orbit"
    #       plot_updated = False
    #       orbit_direction = 'CW'
    #       orbit_center = [[-936.18087028], [   4.65710893], [1005.31309917]]
    #       orbit_radius = 10.
    #       airspeed = 35.0
    #   hs:
    #       normal = [[-0.99994588], [ 0.00618334], [ 0.00836689]]
    #       point = [[-936.07691215], [  10.9102307 ], [1013.11615375]]
    path1, hs1 = construct_fillet_circle(waypoints=waypoints, ptr=ptr, radius=10.)
    print("Path1: ", path1)
    print("hs1: ", hs1)

    # Update the pointer
    ptr.increment_pointers(num_waypoints=waypoints.num_waypoints)

    # Construct a line
    #   q_i-1 = [[-0.99994588], [ 0.00618334], [ 0.00836689]]
    #   q_i = [[ 0.99998219], [-0.00370636], [-0.00467882]]
    #   varrho = 0.00444283
    #   path:
    #       type = "orbit"
    #       plot_updated = False
    #       orbit_direction = 'CCW'
    #       orbit_center = [[4511.49061612], [ -17.1999848 ], [ 975.83637369]]
    #       orbit_radius = 10.
    #       airspeed = 25.0
    #   hs:
    #       normal = [[ 0.99998219], [-0.00370636], [-0.00467882]]
    #       point = [[4511.55012176], [ -11.62464449], [ 984.13770127]]
    path2, hs2 = construct_fillet_circle(waypoints=waypoints, ptr=ptr, radius = 10.)
    print("Path2: ", path2)
    print("hs2: ", hs2)


if __name__ == "__main__":
    #construct_line_helper()
    #construct_fillet_line_helper()
    construct_fillet_circle_helper()
