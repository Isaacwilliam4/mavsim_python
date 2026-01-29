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
from mav_sim.tools.rotations import Euler2Quaternion, Quaternion2Euler

s = np.sin
c = np.cos
t = np.tan

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
    
def get_ned_dt(state: DynamicStateEuler) -> tuple[float, float, float]:
    """Gets the ned_dt values

    Args:
        state (types.DynamicStateEuler): state
    Returns:
        tuple: tuple containing (n_dt, e_dt, d_dt)
    """

    # --- NED DT
    ned_motion_mat = np.array([
        [c(state.theta)*c(state.psi), s(state.phi)*s(state.theta)*c(state.psi) - c(state.phi)*s(state.psi), c(state.phi)*s(state.theta)*c(state.psi) + s(state.phi)*s(state.psi)],
        [c(state.theta)*s(state.psi), s(state.phi)*s(state.theta)*s(state.psi) + c(state.phi)*c(state.psi), c(state.phi)*s(state.theta)*s(state.psi) - s(state.phi)*c(state.psi)],
        [-s(state.theta), s(state.phi)*c(state.theta), c(state.phi)*c(state.theta)]
    ])

    uvw_vec = np.array([
        [state.u],
        [state.v],
        [state.w]
        ])

    ned_dt = ned_motion_mat@uvw_vec
    ned_dt_flat = ned_dt.flatten()

    return ned_dt_flat[0], ned_dt_flat[1], ned_dt_flat[2]



def get_phi_theta_psi_dt(state: DynamicStateEuler) -> tuple[float, float, float]:
    """Gets the phi_theta_psi_dt values

    Args:
        state (types.DynamicStateEuler): state
    Returns:
        tuple: tuple containing (phi_dt, theta_dt, psi_dt)
    """

    pqr_vec = np.array([
        [state.p],
        [state.q],
        [state.r]
    ])

    psi_theta_phi_motion_mat = np.array([
        [1, s(state.phi)*t(state.theta), c(state.phi)*t(state.theta)],
        [0, c(state.phi), -s(state.phi)],
        [0, s(state.phi)/c(state.theta), c(state.phi)/c(state.theta)],
    ])

    phi_theta_psi_dt = psi_theta_phi_motion_mat@pqr_vec
    phi_theta_psi_dt_flat = phi_theta_psi_dt.flatten()

    return phi_theta_psi_dt_flat[0], phi_theta_psi_dt_flat[1], phi_theta_psi_dt_flat[2]

def get_pqr_dt(state: DynamicStateEuler, forces_moments: ForceMoments) -> tuple[float, float, float]:
    """Gets the pqr_dt values

    Args:
        state (types.DynamicStateEuler): state
        forces_moments (types.ForceMoment): force moments
    Returns:
        tuple: tuple containing (p_dt, q_dt, r_dt)
    """

    pqr_mot_vec = np.array([
        [MAV.gamma1*state.p*state.q - MAV.gamma2*state.q*state.r],
        [MAV.gamma5*state.p*state.r - MAV.gamma6*(state.p**2 -state.r**2)],
        [MAV.gamma7*state.p*state.q - MAV.gamma1*state.q*state.r]
    ])

    pqr_sum_vec = np.array([
        [MAV.gamma3*forces_moments.l + MAV.gamma4*forces_moments.n],
        [(1/MAV.Jy)*forces_moments.m],
        [MAV.gamma4*forces_moments.l + MAV.gamma8*forces_moments.n]
    ])

    pqr_dt = pqr_mot_vec + pqr_sum_vec
    pqr_dt_flat = pqr_dt.flatten()

    return pqr_dt_flat[0], pqr_dt_flat[1], pqr_dt_flat[2]

def get_uvw_dt(state: DynamicStateEuler, forces_moments: ForceMoments) -> tuple[float, float, float]:
    """Gets the uvw_dt values

    Args:
        state (types.DynamicStateEuler): state
        forces_moments (types.ForceMoment): force moments
    Returns:
        tuple: tuple containing (u_dt, v_dt, w_dt)
    """

    force_vec = np.array([
        [forces_moments.fx],
        [forces_moments.fy],
        [forces_moments.fz],
    ])

    uvw_mot_vec = np.array([
        [state.r*state.v - state.q*state.w],
        [state.p*state.w - state.r*state.u],
        [state.q*state.u - state.p*state.v]
    ])

    force_accel_vec = (1/MAV.mass) * force_vec

    uvw_dt = uvw_mot_vec + force_accel_vec
    uvw_dt_flat = uvw_dt.flatten()

    return uvw_dt_flat[0], uvw_dt_flat[1], uvw_dt_flat[2]

def derivatives_euler(state: types.DynamicStateEuler, forces_moments: types.ForceMoment) -> types.DynamicStateEuler:
    """Implements the dynamics xdot = f(x, u) where u is the force/moment vector

    Args:
        state: Current state of the vehicle
        forces_moments: 6x1 array containing [fx, fy, fz, Mx, My, Mz]^T

    Returns:
        Time derivative of the state ( f(x,u), where u is the force/moment vector )
    """

    _state = DynamicStateEuler(state)
    _forces_moments = ForceMoments(forces_moments)

    # --- NED DT
    n_dt, e_dt, d_dt = get_ned_dt(_state)
    # ---

    # --- UVW DT
    u_dt, v_dt, w_dt = get_uvw_dt(_state, _forces_moments)
    # ---

    # --- psi_theta_phi dt
    phi_dt, theta_dt, psi_dt = get_phi_theta_psi_dt(_state)
    # ---

    # pqr dt
    p_dt, q_dt, r_dt = get_pqr_dt(_state, _forces_moments)
    #

    # collect the derivative of the states
    x_dot = np.empty( (IND_EULER.NUM_STATES,1) )
    x_dot[IND_EULER.NORTH] = n_dt
    x_dot[IND_EULER.EAST] = e_dt
    x_dot[IND_EULER.DOWN] = d_dt
    x_dot[IND_EULER.U] = u_dt
    x_dot[IND_EULER.V] = v_dt
    x_dot[IND_EULER.W] = w_dt
    x_dot[IND_EULER.PHI] = phi_dt
    x_dot[IND_EULER.THETA] = theta_dt
    x_dot[IND_EULER.PSI] = psi_dt
    x_dot[IND_EULER.P] = p_dt
    x_dot[IND_EULER.Q] = q_dt
    x_dot[IND_EULER.R] = r_dt

    return x_dot
