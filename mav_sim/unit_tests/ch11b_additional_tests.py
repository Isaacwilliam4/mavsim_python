"""This module provides some additional tests for chapter 11 that can be used to help you
   debug if you are not passing the unit tests
"""

import numpy as np
from mav_sim.chap11.dubins_manager import (
    construct_dubins_circle_end,
    construct_dubins_line,
)
from mav_sim.chap11.dubins_parameters import (
    DubinsParameters,
    DubinsPoints,
    calculate_lsl,
    calculate_lsr,
)
from mav_sim.chap11.path_manager_utilities import WaypointIndices, get_airspeed
from mav_sim.message_types.msg_waypoints import MsgWaypoints

# def compute_parameters(points: DubinsPoints) -> DubinsParamsStruct:
#     """Calculate the dubins paths parameters. Returns the parameters defining the shortest
#        path between two oriented waypoints

#     Args:
#         points: Struct defining the oriented points and radius for the Dubins path

#     Returns:
#         dubin: variables for the shortest Dubins path
#     """

#     # Check to ensure sufficient distance between points
#     (p_s, _, p_e, _, R) = points.extract()
#     ell = np.linalg.norm(p_s[0:2] - p_e[0:2])
#     if ell < 2 * R:
#         raise ValueError('Error in Dubins Parameters: The distance between nodes must be larger than 2R.')

#     # Compute the parameter for the Dubins paths and return the variables with shortest length
#     dubin = calculate_rsr(points) # compute parameters for Right-Straight-Right case
#     dubin_rsl = calculate_rsl(points) # compute parameters for the Right-Straight-Left case
#     if dubin_rsl.L < dubin.L:
#         dubin = dubin_rsl

#     ### Keep Going, Implement more here ###
#     print("Need to consider lsl and lsr paths!!!")

#     return dubin


# def calculate_rsr(points: DubinsPoints) -> DubinsParamsStruct:
#     """Calculates the Dubins parameters for the right-straight-right case

#     Args:
#         points: Struct defining the oriented points and radius for the Dubins path

#     Returns:
#         dubin: variables for the Dubins path
#     """

#     # Initialize output and extract inputs
#     dubin = DubinsParamsStruct()
#     (p_s, chi_s, p_e, chi_e, R) = points.extract()

#     # Compute the start and end circles
#     c_rs = p_s + R * np.array([[np.cos(chi_s+np.pi/2.)],[np.sin(chi_s+np.pi/2.)], [0.]])
#     c_re = p_e + R * np.array([[np.cos(chi_e+np.pi/2.)],[np.sin(chi_e+np.pi/2.)], [0.]])

#     # compute length for R-S-R case (Section 11.2.2.1)
#     theta = np.arctan2(c_re.item(1) - c_rs.item(1), c_re.item(0) - c_rs.item(0)) # Angle of line connecting centers
#     dubin.L = cast(float, np.linalg.norm(c_rs - c_re) \
#             + R * mod(2 * np.pi + mod(theta - np.pi / 2) - mod(chi_s - np.pi / 2)) \
#             + R * mod(2 * np.pi + mod(chi_e - np.pi / 2) - mod(theta - np.pi / 2)) )

#     # Compute the dubins parameters (lines 7-11, 34-35 of Algorithm 9)
#     e1 = np.array([[1, 0, 0]]).T
#     dubin.c_s = c_rs                                                            # Line 8
#     dubin.lam_s = 1
#     dubin.c_e = c_re
#     dubin.lam_e = 1
#     dubin.q1 = cast(NP_MAT, (dubin.c_e - dubin.c_s) / np.linalg.norm(dubin.c_e - dubin.c_s) )  # Line 9
#     dubin.z1 = dubin.c_s + R * rotz(-np.pi / 2) @ dubin.q1                      # Line 10
#     dubin.z2 = dubin.c_e + R * rotz(-np.pi / 2) @ dubin.q1                      # Line 11
#     dubin.z3 = p_e                                                              # Line 34
#     dubin.q3 = rotz(chi_e) @ e1                                                 # Line 35

#     return dubin

# def calculate_rsl(points: DubinsPoints) -> DubinsParamsStruct:
#     """Calculates the Dubins parameters for the right-straight-left case

#     Args:
#         points: Struct defining the oriented points and radius for the Dubins path

#     Returns:
#         dubin: variables for the Dubins path
#     """

#     # Initialize output and extract inputs
#     dubin = DubinsParamsStruct()
#     (p_s, chi_s, p_e, chi_e, R) = points.extract()

#     # compute start and end circles (equations (11.5) and (11.6) )
#     c_rs = p_s + R * np.array([[np.cos(chi_s+np.pi/2.)],[np.sin(chi_s+np.pi/2.)], [0.]])
#     c_le = p_e + R * np.array([[np.cos(chi_e-np.pi/2.)],[np.sin(chi_e-np.pi/2.)], [0.]])

#     # compute L2 (Section 11.2.2.2)
#     ell = np.linalg.norm(c_le - c_rs) # Little l in book, distance between center points
#     theta = np.arctan2(c_le.item(1) - c_rs.item(1), c_le.item(0) - c_rs.item(0))
#     theta2 = theta - np.pi / 2 + np.arcsin(2 * R / ell)
#     #if not np.isreal(theta2): # Occurs when 2R > ell
#     if np.isnan(theta2): # Occurs when 2R > ell
#         dubin.L = np.inf
#     else:
#         # Equation (11.10)
#         dubin.L = np.sqrt(ell ** 2 - 4 * R ** 2) \
#                 + R * mod(2 * np.pi + mod(theta2) - mod(chi_s - np.pi / 2)) \
#                 + R * mod(2 * np.pi + mod(theta2 + np.pi) - mod(chi_e + np.pi / 2))


#     # Halfplane parameters (Algorithm 9 lines 12-19 and 34-35)
#     e1 = np.array([[1, 0, 0]]).T
#     dubin.c_s = c_rs                                        # Line 13
#     dubin.lam_s = 1
#     dubin.c_e = c_le
#     dubin.lam_e = -1
#     ell = np.linalg.norm(dubin.c_e - dubin.c_s)             # Line 14
#     dubin.q1 = rotz(theta2 + np.pi / 2) @ e1                # Line 17
#     dubin.z1 = dubin.c_s + R * rotz(theta2) @ e1            # Line 18
#     dubin.z2 = dubin.c_e + R * rotz(theta2 + np.pi) @ e1    # Line 19
#     dubin.z3 = p_e                                          # Line 34
#     dubin.q3 = rotz(chi_e) @ e1                             # Line 35

#     return dubin

# def dubins_manager(state: MsgState, waypoints: MsgWaypoints, ptr_prv: WaypointIndices,
#                 path_prv: MsgPath, hs_prv: HalfSpaceParams, radius: float, manager_state: int, \
#                 dubins_path_prv: DubinsParameters) \
#             -> tuple[MsgPath, HalfSpaceParams, WaypointIndices, int, DubinsParameters]:
#     """Update for the Dubins path manager.
#        Updates state machine if the MAV enters into the next halfspace.

#        Keep in mind that the constructor for DubinsParameters makes a call to dupins_parameters.compute_parameters()

#     Args:
#         state: current state of the vehicle
#         waypoints: The waypoints to be followed
#         ptr_prv: The indices that were being used on the previous iteration (i.e., current waypoint
#                  inidices being followed when manager called)
#         hs_prv: The previous halfspace being looked for (i.e., the current halfspace when manager called)
#         radius: minimum radius circle for the mav
#         manager_state: Integer state of the manager
#                 1: First portion of beginning circle
#                 2: Second portion of beginning circle, up to H_1
#                 3: Straight-line segment, up to H_2
#                 4: First portion of ending circle
#                 5: Second portion of the ending circle, up to H_3
#         dubins_path_prv: The path produced on the previous call to dubins_manager
#                 * Garbage whenever a new set of waypoints is received (i.e. waypoints.flag_waypoints_changed is True)
#                 * Can be completely ignored
#                 * Can be used to avoid recalculatiing the dubins path variables when they should have remained unchanged

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
#     dubins_path = dubins_path_prv # Note that this should be changed when new waypoints received or
#                                   # a transition is made to a new waypoint path

#     # Extract the position and desired airspeed
#     mav_pos = np.array([[state.north, state.east, -state.altitude]]).T
#     desired_airspeed = get_airspeed(waypoints, ptr)

#     # if the waypoints have changed, update the waypoint pointer
#     if waypoints.flag_waypoints_changed is True:
#         waypoints.flag_waypoints_changed = False
#         ptr = WaypointIndices() # Resets the pointers

#         # dubins path parameters
#         dubins_path = DubinsParameters(
#             p_s=waypoints.get_ned(ptr.previous),
#             chi_s=waypoints.course.item(ptr.previous),
#             p_e=waypoints.get_ned(ptr.current),
#             chi_e=waypoints.course.item(ptr.current),
#             R=radius)
#         (path, hs) = construct_dubins_circle_start(desired_airspeed=desired_airspeed, dubins_path=dubins_path)
#         if inHalfSpace(mav_pos, hs):
#             manager_state = 1
#         else:
#             manager_state = 2

#     # state machine for dubins path
#     if manager_state == 1:
#         # follow start orbit until out of H1
#         if not inHalfSpace(mav_pos, hs):
#             manager_state = 2
#     if manager_state == 2:
#         # follow start orbit until cross into H1
#         if inHalfSpace(mav_pos, hs):
#             (path, hs) = construct_dubins_line(desired_airspeed=desired_airspeed, dubins_path=dubins_path)
#             manager_state = 3
#     if manager_state == 3:
#         # follow start orbit until cross into H2
#         if inHalfSpace(mav_pos, hs):
#             (path, hs) = construct_dubins_circle_end(desired_airspeed=desired_airspeed, dubins_path=dubins_path)
#             manager_state = 4
#     if manager_state == 4:
#         # follow end orbit until out of H3
#         if not inHalfSpace(mav_pos, hs):
#             manager_state = 5

#     if manager_state == 5:
#         # follow end orbit until cross into H3
#         if inHalfSpace(mav_pos, hs):
#             ptr.increment_pointers(waypoints.num_waypoints)
#             dubins_path = DubinsParameters(
#                 p_s=waypoints.get_ned(ptr.previous),
#                 chi_s=waypoints.course.item(ptr.previous),
#                 p_e=waypoints.get_ned(ptr.current),
#                 chi_e=waypoints.course.item(ptr.current),
#                 R=radius)
#             (path, hs) = construct_dubins_circle_start(desired_airspeed=desired_airspeed, dubins_path=dubins_path)
#             if inHalfSpace(mav_pos, hs):
#                 manager_state = 1
#             else:
#                 manager_state = 2

#     return (path, hs, ptr, manager_state, dubins_path)

# def construct_dubins_circle_start(desired_airspeed: float, dubins_path: DubinsParameters) \
#     -> tuple[MsgPath, HalfSpaceParams]:
#     """ Create the starting orbit for the dubin's path

#     Args:
#         waypoints: The waypoints to be followed
#         ptr: The indices of the waypoints being used for the path
#         dubins_Path: The parameters that make-up the Dubin's path between waypoints

#     Returns:
#         path: The first circle of the Dubin's path
#         hs: The halfspace for switching to the next waypoint (H_1)
#     """
#     # Create the orbit
#     path = MsgPath()
#     path.type = 'orbit'
#     path.plot_updated = False
#     path.airspeed = desired_airspeed
#     path.orbit_radius = dubins_path.radius
#     path.orbit_center = dubins_path.center_s
#     if dubins_path.dir_s == 1:
#         path.orbit_direction = 'CW'
#     else:
#         path.orbit_direction = 'CCW'

#     # Define the switching halfspace
#     hs = HalfSpaceParams(normal=dubins_path.n1, point=dubins_path.r1)

#     return (path, hs)


def calculate_lsr_helper() -> None:
    """ This function seemingly does nothing. However, if you run it in the debug mode, you should be able to check the function for intermediary values.
    """
    # Create the Dubins Points
    p_s = np.array([[1.], [2.], [3.]]) # Starting point
    chi_s = .123 # Starting orientation
    p_e = np.array([[100.], [300.], [3.]]) # Ending point
    chi_e = 6.1 # Ending orienation
    radius = 20.2 # Minimum radius
    pnts = DubinsPoints(p_s=p_s,
                        chi_s=chi_s,
                        p_e=p_e,
                        chi_e = chi_e,
                        radius=radius)

    # Calculate the dubins parameters associated with the path
    #   center start: [[  3.47833982], [-18.04738965], [  3.        ]]
    #   center end: [[103.67968259], [319.86202246], [  3.        ]]
    #   distance between centers: 352.4529470469377
    #   vartheta: 1.2825231546102556
    #   vartheta2: 1.455918592259827
    #   dubins path length: 555.5819296883817
    #   q1: [[0.39231858], [0.9198294 ], [0.        ]]
    #   q3: [[ 0.98326844], [-0.1821625 ], [ 0.        ]]
    #   z1: [[-15.10221412], [-10.1225543 ], [  3.        ]]
    #   z2: [[122.26023653], [311.93718711], [  3.        ]]
    #   z3: [[100.], [300.], [  3.]]
    params = calculate_lsr(points=pnts)
    print(params)


def calculate_lsl_helper() -> None:
    """ This function seemingly does nothing. However, if you run it in the debug mode, you should be able to check the function for intermediary values.
    """
    # Create the Dubins Points
    p_s = np.array([[1.], [2.], [3.]]) # Starting point
    chi_s = .123 # Starting orientation
    p_e = np.array([[100.], [300.], [3.]]) # Ending point
    chi_e = 6.1 # Ending orienation
    radius = 20.2 # Minimum radius
    pnts = DubinsPoints(p_s=p_s,
                        chi_s=chi_s,
                        p_e=p_e,
                        chi_e = chi_e,
                        radius=radius)

    # Calculate the dubins parameters associated with the path
    #   center start: [[  3.47833982], [-18.04738965], [  3.        ]]
    #   center end: [[ 96.32031741], [280.13797754], [  3.        ]]
    #   vartheta: 1.2689534850322113
    #   dubins path length: 445.40985892152224
    #   q1: [[0.29728024], [0.95479027], [0.        ]]
    #   q3: [[ 0.98326844], [-0.1821625 ], [ 0.        ]]
    #   z1: [[-15.80842364], [-12.04232884], [  3.        ]]
    #   z2: [[ 77.03355395], [286.14303835], [  3.        ]]
    #   z3: [[100.], [300.], [  3.]]
    params = calculate_lsl(points=pnts)
    print(params)


def construct_dubins_line_helper() -> None:
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

    # Create the Dubins Points
    p_s = np.array([[1.], [2.], [3.]]) # Starting point
    chi_s = .123 # Starting orientation
    p_e = np.array([[100.], [300.], [3.]]) # Ending point
    chi_e = 6.1 # Ending orienation
    radius = 20.2 # Minimum radius
    dub_pnts = DubinsPoints(p_s=p_s,
                            chi_s=chi_s,
                            p_e=p_e,
                            chi_e = chi_e,
                            radius=radius)

    # Calculate the dubins parameters associated with the path
    params = calculate_lsl(points=dub_pnts)
    dubins_path = DubinsParameters()
    dubins_path.set(vals=params)


    # Construct a dubins line
    #   path:
    #       .type = 'line'
    #       .plot_updated = False
    #       .airspeed = 35.
    #       .line_origin = [[-15.80842364], [-12.04232884], [  3.        ]]
    #       .line_direction = [[0.29728024], [0.95479027], [0.        ]]
    #   hs:
    #       .point = [[ 77.03355395], [286.14303835], [  3.        ]]
    #       .normal = [[0.29728024], [0.95479027], [0.        ]]
    desired_airspeed = get_airspeed(waypoints, ptr)
    path, hs = construct_dubins_line(desired_airspeed=desired_airspeed,
                                     dubins_path=dubins_path)
    print(path, hs)

def construct_dubins_circle_end_helper() -> None:
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

    # Create the Dubins Points
    p_s = np.array([[1.], [2.], [3.]]) # Starting point
    chi_s = .123 # Starting orientation
    p_e = np.array([[100.], [300.], [3.]]) # Ending point
    chi_e = 6.1 # Ending orienation
    radius = 20.2 # Minimum radius
    dub_pnts = DubinsPoints(p_s=p_s,
                            chi_s=chi_s,
                            p_e=p_e,
                            chi_e = chi_e,
                            radius=radius)

    # Calculate the dubins parameters associated with the path
    params = calculate_lsl(points=dub_pnts)
    dubins_path = DubinsParameters(p_s=p_s,
                                   chi_s=chi_s,
                                   p_e=p_e,
                                   chi_e=chi_e,
                                   R=radius)
    dubins_path.set(vals=params)

    # Construct a dubins line
    #   path:
    #       .type = 'orbit'
    #       .plot_updated = False
    #       .airspeed = 35.
    #       .orbit_radius = 20.2
    #       .orbit_center = [[ 96.32031741], [280.13797754], [  3.        ]]
    #       .orbit_direction = 'CCW'
    #   hs:
    #       .point = [[100.], [300.], [  3.]]
    #       .normal = [[ 0.98326844], [-0.1821625 ], [ 0.        ]]
    desired_airspeed = get_airspeed(waypoints, ptr)
    path, hs = construct_dubins_circle_end(desired_airspeed=desired_airspeed,
                                           dubins_path=dubins_path)
    print(path, hs)

if __name__ == "__main__":
    calculate_lsr_helper()
    #calculate_lsl_helper()
    #construct_dubins_line_helper()
    #construct_dubins_circle_end_helper()
