"""
mav_dynamics
    - this file implements the dynamic equations of motion for MAV using Euler coordinates
    - use Euler angles for the attitude state

part of mavsimPy
    - Beard & McLain, PUP, 2012
    - Update history:
        12/17/2018 - RWB
        1/14/2019 - RWB
        12/21 - GND
        12/22 - GND
"""
import mav_sim.parameters.aerosonde_parameters as MAV
import numpy as np
from mav_sim.chap3.mav_dynamics import IND, ForceMoments
from mav_sim.tools import types
from mav_sim.tools.rotations import Euler2Quaternion, Euler2Rotation, Quaternion2Euler


# Indexing constants for state using Euler representation
class StateIndicesEuler:
    """Constant class for easy access of state indices
    """
    NORTH: int  = 0  # North position
    EAST: int   = 1  # East position
    DOWN: int   = 2  # Down position
    U: int      = 3  # body-x velocity
    V: int      = 4  # body-y velocity
    W: int      = 5  # body-z velocity
    PHI: int    = 6  # Roll angle (about x-axis)
    THETA: int  = 7  # Pitch angle (about y-axis)
    PSI: int    = 8  # Yaw angle (about z-axis)
    P: int      = 9 # roll rate - body frame - i
    Q: int      = 10 # pitch rate - body frame - j
    R: int      = 11 # yaw rate - body frame - k
    VEL: list[int] = [U, V, W] # Body velocity indices
    ANG_VEL: list[int] = [P, Q, R] # Body rotational velocities
    NUM_STATES: int = 12 # Number of states
IND_EULER = StateIndicesEuler()

# Conversion functions
def euler_state_to_quat_state(state_euler: types.DynamicStateEuler) -> types.DynamicState:
    """Converts an Euler state representation to a quaternion state representation

    Args:
        state_euler: The state vector to be converted to a quaternion representation

    Returns:
        state_quat: The converted state
    """
    # Create the quaternion from the euler coordinates
    e = Euler2Quaternion(phi=state_euler[IND_EULER.PHI], theta=state_euler[IND_EULER.THETA], psi=state_euler[IND_EULER.PSI])

    # Copy over data
    state_quat = np.zeros((IND.NUM_STATES,1))
    state_quat[IND.NORTH] = state_euler.item(IND_EULER.NORTH)
    state_quat[IND.EAST] = state_euler.item(IND_EULER.EAST)
    state_quat[IND.DOWN] = state_euler.item(IND_EULER.DOWN)
    state_quat[IND.U] = state_euler.item(IND_EULER.U)
    state_quat[IND.V] = state_euler.item(IND_EULER.V)
    state_quat[IND.W] = state_euler.item(IND_EULER.W)
    state_quat[IND.E0] = e.item(0)
    state_quat[IND.E1] = e.item(1)
    state_quat[IND.E2] = e.item(2)
    state_quat[IND.E3] = e.item(3)
    state_quat[IND.P] = state_euler.item(IND_EULER.P)
    state_quat[IND.Q] = state_euler.item(IND_EULER.Q)
    state_quat[IND.R] = state_euler.item(IND_EULER.R)

    return state_quat

def quat_state_to_euler_state(state_quat: types.DynamicState) -> types.DynamicStateEuler:
    """Converts a quaternion state representation to an Euler state representation

    Args:
        state_quat: The state vector to be converted

    Returns
        state_euler: The converted state
    """
    # Create the quaternion from the euler coordinates
    phi, theta, psi = Quaternion2Euler(state_quat[IND.QUAT])

    # Copy over data
    state_euler = np.zeros((IND_EULER.NUM_STATES,1))
    state_euler[IND_EULER.NORTH] = state_quat.item(IND.NORTH)
    state_euler[IND_EULER.EAST]  = state_quat.item(IND.EAST)
    state_euler[IND_EULER.DOWN]  = state_quat.item(IND.DOWN)
    state_euler[IND_EULER.U]     = state_quat.item(IND.U)
    state_euler[IND_EULER.V]     = state_quat.item(IND.V)
    state_euler[IND_EULER.W]     = state_quat.item(IND.W)
    state_euler[IND_EULER.PHI]   = phi
    state_euler[IND_EULER.THETA] = theta
    state_euler[IND_EULER.PSI]   = psi
    state_euler[IND_EULER.P]     = state_quat.item(IND.P)
    state_euler[IND_EULER.Q]     = state_quat.item(IND.Q)
    state_euler[IND_EULER.R]     = state_quat.item(IND.R)

    return state_euler


class DynamicStateEuler:
    """Struct for the dynamic state
    """
    def __init__(self, state: types.DynamicStateEuler ) -> None:
        self.north: float     # North position
        self.east: float      # East position
        self.down: float      # Down position
        self.u: float         # body-x velocity
        self.v: float         # body-y velocity
        self.w: float         # body-z velocity
        self.phi: float       # roll angle (about x-axis)
        self.theta: float     # pitch angle (about y-axis)
        self.psi: float       # yaw angle (about z-axis)
        self.p: float         # roll rate - body frame - i
        self.q: float         # pitch rate - body frame - j
        self.r: float         # yaw rate - body frame - k

        self.extract_state(state)

    def extract_state(self, state: types.DynamicStateEuler) ->None:
        """Initializes the state variables

        Args:
            state: State from which to extract the state values

        """
        self.north = state.item(IND_EULER.NORTH)
        self.east =  state.item(IND_EULER.EAST)
        self.down =  state.item(IND_EULER.DOWN)
        self.u =     state.item(IND_EULER.U)
        self.v =     state.item(IND_EULER.V)
        self.w =     state.item(IND_EULER.W)
        self.phi =   state.item(IND_EULER.PHI)
        self.theta = state.item(IND_EULER.THETA)
        self.psi =   state.item(IND_EULER.PSI)
        self.p =     state.item(IND_EULER.P)
        self.q =     state.item(IND_EULER.Q)
        self.r =     state.item(IND_EULER.R)

    def convert_to_numpy(self) -> types.DynamicStateEuler:
        """Converts the state to a numpy object
        """
        output = np.empty( (IND_EULER.NUM_STATES,1) )
        output[IND_EULER.NORTH, 0] = self.north
        output[IND_EULER.EAST, 0] = self.east
        output[IND_EULER.DOWN, 0] = self.down
        output[IND_EULER.U, 0] = self.u
        output[IND_EULER.V, 0] = self.v
        output[IND_EULER.W, 0] = self.w
        output[IND_EULER.PHI, 0] = self.phi
        output[IND_EULER.THETA, 0] = self.theta
        output[IND_EULER.PSI, 0] = self.psi
        output[IND_EULER.P, 0] = self.p
        output[IND_EULER.Q, 0] = self.q
        output[IND_EULER.R, 0] = self.r

        return output

def derivatives_euler(state: types.DynamicStateEuler, forces_moments: types.ForceMoment) -> types.DynamicStateEuler:
    """Implements the dynamics xdot = f(x, u) where u is the force/moment vector

    Args:
        state: Current state of the vehicle
        forces_moments: 6x1 array containing [fx, fy, fz, Mx, My, Mz]^T

    Returns:
        Time derivative of the state ( f(x,u), where u is the force/moment vector )
    """
    s = np.sin
    c = np.cos
    t = np.tan

    _state = DynamicStateEuler(state)
    _forces_moments = ForceMoments(forces_moments)

    force_vec = np.array([
        [_forces_moments.fx],
        [_forces_moments.fy],
        [_forces_moments.fz],
    ])



    # --- NED DT
    ned_motion_mat = np.array([
        [c(_state.theta)*c(_state.phi), s(_state.phi)*s(_state.theta)*c(_state.phi) - c(_state.phi)*s(_state.phi), c(_state.phi)*s(_state.theta)*c(_state.phi) + s(_state.phi)*s(_state.phi)],
        [c(_state.theta)*s(_state.phi), s(_state.phi)*s(_state.theta)*s(_state.phi) + c(_state.phi)*c(_state.phi), c(_state.phi)*s(_state.theta)*s(_state.phi) - s(_state.phi)*c(_state.phi)],
        [-s(_state.theta), s(_state.phi)*c(_state.theta), c(_state.phi)*c(_state.theta)]
    ])

    uvw_vec = np.array([
        [_state.u],
        [_state.v],
        [_state.w]
        ])
    

    ned_dt = ned_motion_mat@uvw_vec
    ned_dt_flat = ned_dt.flatten()
    # ---

    # --- UVW DT
    uvw_mot_vec = np.array([
        [_state.r*_state.v - _state.q*_state.w],
        [_state.p*_state.w - _state.r*_state.u],
        [_state.q*_state.u - _state.p*state[IND_EULER.V]]
    ])

    moments_vec = (1/_forces_moments.m) * force_vec

    uvw_dt = uvw_mot_vec + moments_vec
    uvw_dt_flat = uvw_dt.flatten()
    # ---

    # --- psi_theta_phi dt
    pqr_vec = np.array([
        [_state.p],
        [_state.q],
        [_state.r]
    ])

    psi_theta_phi_motion_mat = np.array([
        [1, s(_state.phi)*t(_state.theta), c(_state.phi)*t(_state.theta)],
        [0, c(_state.phi), -s(_state.phi)],
        [0, s(_state.phi)/c(_state.theta), c(_state.phi)/c(_state.theta)],
    ])

    psi_theta_phi_dt = psi_theta_phi_motion_mat@pqr_vec
    psi_theta_phi_dt_flat = psi_theta_phi_dt.flatten()
    # ---

    # pqr dt
    pqr_mot_vec = np.array([
        [MAV.gamma1*_state.p*_state.q - MAV.gamma2*_state.q*_state.r],
        [MAV.gamma5*_state.p*_state.r - MAV.gamma6*(_state.p^2-_state.r^2)],
        [MAV.gamma7*_state.p*_state.q - MAV.gamma1*_state.q*_state.r]
    ])

    pqr_sum_vec = np.array([
        [MAV.gamma3*_forces_moments.l + MAV.gamma4*_forces_moments.n],
        [(1/MAV.Jy)*_forces_moments.m],
        [MAV.gamma4*_forces_moments.l + MAV.gamma8*_forces_moments.n]
    ])

    pqr_dt = pqr_mot_vec + pqr_sum_vec
    pqr_dt_flat = pqr_dt.flatten()
    #

    # collect the derivative of the states
    x_dot = np.empty( (state[IND_EULER.NUM_STATES],1) )
    x_dot[state[IND_EULER.NORTH]] = ned_dt_flat[0]
    x_dot[state[IND_EULER.EAST]] = ned_dt_flat[1]
    x_dot[state[IND_EULER.DOWN]] = ned_dt_flat[2]
    x_dot[_state.u] = uvw_dt_flat[0]
    x_dot[state[IND_EULER.V]] = uvw_dt_flat[1]
    x_dot[_state.w] = uvw_dt_flat[2]
    x_dot[_state.phi] = psi_theta_phi_dt_flat[0]
    x_dot[_state.theta] = psi_theta_phi_dt_flat[1]
    x_dot[_state.phi] = psi_theta_phi_dt_flat[2]
    x_dot[_state.p] = pqr_dt_flat[0]
    x_dot[_state.q] = pqr_dt_flat[1]
    x_dot[_state.r] = pqr_dt_flat[2]

    return x_dot
