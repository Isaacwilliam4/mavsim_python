"""
autopilot block for mavsim_python
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
        12/21 - GND
"""
from typing import Optional

import mav_sim.parameters.control_parameters as AP
import numpy as np
from mav_sim.chap6.pd_control_with_rate import PDControlWithRate
from mav_sim.chap6.pi_control import PIControl
from mav_sim.chap6.tf_control import TFControl
from mav_sim.message_types.msg_autopilot import MsgAutopilot
from mav_sim.message_types.msg_delta import MsgDelta
from mav_sim.message_types.msg_state import MsgState

# from mav_sim.tools.transfer_function import TransferFunction
from mav_sim.tools.wrap import saturate, wrap


class Autopilot:
    """Creates an autopilot for controlling the mav to desired values
    """
    def __init__(self, ts_control: float) -> None:
        """Initialize the lateral and longitudinal controllers

        Args:
            ts_control: time step for the control
        """
        roll_limit = np.pi*45 / 180
        course_angle_limit = np.pi*30 / 180
        pitch_limit = np.pi*45 / 180
        altitude_from_pitch_limit = np.pi*30 / 180
        airspeed_from_throttle_limit = 1
        yaw_damper_limit = 1

        # instantiate lateral-directional controllers (note, these should be objects, not numbers)
        self.roll_from_aileron = PDControlWithRate(AP.roll_kp, AP.roll_kd, roll_limit)
        self.course_from_roll = PIControl(AP.course_kp, AP.course_ki, ts_control, course_angle_limit)
        self.yaw_damper = TFControl(AP.yaw_damper_kr, 0, 1, 1, ts_control, ts_control, yaw_damper_limit)

        # instantiate longitudinal controllers (note, these should be objects, not numbers)
        self.pitch_from_elevator = PDControlWithRate(AP.pitch_kp, AP.pitch_kd, pitch_limit)
        self.altitude_from_pitch = PIControl(AP.altitude_kp, AP.altitude_ki, ts_control, altitude_from_pitch_limit)
        self.airspeed_from_throttle = PIControl(AP.airspeed_throttle_kp, AP.airspeed_throttle_ki, ts_control, airspeed_from_throttle_limit)
        self.commanded_state = MsgState()

    def update(self, cmd: MsgAutopilot, state: MsgState) -> tuple[MsgDelta, MsgState]:
        """Given a state and autopilot command, compute the control to the mav

        Args:
            cmd: command to the autopilot
            state: current state of the mav

        Returns:
            delta: low-level flap commands
            commanded_state: the state being commanded
        """

        # lateral autopilot
        cmd.altitude_command = saturate(cmd.altitude_command, -AP.altitude_zone, AP.altitude_zone)
        chi_c = wrap(cmd.course_command, state.chi)

        phi_c = saturate( # course hold loop, 6.1.1.2 with addition of feedforward term
             cmd.phi_feedforward + 
                self.course_from_roll.update(chi_c, state.chi), 
                -np.radians(30), np.radians(30))
        
        theta_c = self.altitude_from_pitch.update(cmd.altitude_command, state.altitude) # commanded value for theta
        delta_a = self.roll_from_aileron.update(phi_c, state.phi, 0)
        delta_r = self.yaw_damper.update(state.r)

        # longitudinal autopilot
        ydot = 0
        pitch = self.pitch_from_elevator.update(theta_c, state.theta, ydot)

        delta_e = self.pitch_from_elevator.update(0, 0, 0)
        delta_t = self.airspeed_from_throttle.update(cmd.airspeed_command, state.Va)
        delta_t = saturate(delta_t, 0.0, 1.0)

        # construct control outputs and commanded states
        delta = MsgDelta(elevator=delta_e,
                         aileron=delta_a,
                         rudder=delta_r,
                         throttle=delta_t)
        
        self.commanded_state.altitude = cmd.altitude_command
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = phi_c
        self.commanded_state.theta = theta_c
        self.commanded_state.chi = cmd.course_command
        return delta, self.commanded_state.copy()
