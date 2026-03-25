"""path_follower.py implements a class for following a path with a mav
"""
from math import cos, sin

import numpy as np
from mav_sim.message_types.msg_autopilot import MsgAutopilot
from mav_sim.message_types.msg_path import MsgPath
from mav_sim.message_types.msg_state import MsgState
from mav_sim.tools.wrap import wrap


class PathFollower:
    """Class for path following
    """
    def __init__(self) -> None:
        """Initialize path following class
        """
        self.chi_inf = np.radians(50)  # approach angle for large distance from straight-line path
        self.k_path = 0.01 #0.05  # path gain for straight-line path following
        self.k_orbit = 1.# 10.0  # path gain for orbit following
        self.gravity = 9.8
        self.autopilot_commands = MsgAutopilot()  # message sent to autopilot

    def update(self, path: MsgPath, state: MsgState) -> MsgAutopilot:
        """Update the control for following the path

        Args:
            path: path to be followed
            state: current state of the mav

        Returns:
            autopilot_commands: Commands to autopilot for following the path
        """
        if path.type == 'line':
            self.autopilot_commands = follow_straight_line(path=path, state=state, k_path=self.k_path, chi_inf=self.chi_inf)
        elif path.type == 'orbit':
            self.autopilot_commands = follow_orbit(path=path, state=state, k_orbit=self.k_orbit, gravity=self.gravity)
        return self.autopilot_commands

def follow_straight_line(path: MsgPath, state: MsgState, k_path: float, chi_inf: float) -> MsgAutopilot:
    """Calculate the autopilot commands for following a straight line

    Args:
        path: straight-line path to be followed
        state: current state of the mav
        k_path: convergence gain for converging to the path
        chi_inf: Angle to take towards path when at an infinite distance from the path

    Returns:
        autopilot_commands: the commands required for executing the desired line
    """
    # Initialize the output
    autopilot_commands = MsgAutopilot()

    # Create autopilot commands here
    r = path.line_origin
    q = path.line_direction
    p = np.array(
        [
            [state.north],
            [state.east],
            [-state.altitude]
        ]
    )

    e_p = p - r
    e_p[2] = 0


    n = np.array([
        [q[1].item()],
        [-q[0].item()],
        [0],
    ])

    s = e_p - np.dot(e_p.flatten(), n.flatten()) * n

    s_n = s[0].item()
    s_e = s[1].item()

    q_n = q[0].item()
    q_e = q[1].item()
    q_d = q[2].item()

    chi_q = np.atan2(q_e, q_n)
    chi_q = wrap(chi_q, state.chi)

    rot_p_i = np.array([
        [np.cos(chi_q), np.sin(chi_q), 0],
        [-np.sin(chi_q), np.cos(chi_q), 0],
        [0, 0, 1],
    ])

    e_p_i = rot_p_i @ e_p
    e_py = e_p_i[1].item()
    r_d = r[2].item()

    chi_c = chi_q - chi_inf * (2 / np.pi) * np.atan(k_path * e_py) 
    h_d = -r_d - np.sqrt(s_n**2 + s_e**2)*(q_d / (np.sqrt(q_n**2 + q_e**2)))
    h_c = h_d

    autopilot_commands.course_command = chi_c.item()
    autopilot_commands.altitude_command = h_c
    autopilot_commands.phi_feedforward = 0.
    autopilot_commands.airspeed_command = state.Va

    return autopilot_commands


def follow_orbit(path: MsgPath, state: MsgState, k_orbit: float, gravity: float) -> MsgAutopilot:
    """Calculate the autopilot commands for following a circular path

    Args:
        path: circular orbit to be followed
        state: current state of the mav
        k_orbit: Convergence gain for reducing error to orbit
        gravity: Gravity constant

    Returns:
        autopilot_commands: the commands required for executing the desired orbit
    """

    # Initialize the output
    autopilot_commands = MsgAutopilot()

    return autopilot_commands
