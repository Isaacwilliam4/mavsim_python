from mav_sim.tools import types
import numpy as np
import mav_sim.parameters.aerosonde_parameters as MAV
from mav_sim.tools.rotations import Euler2Quaternion, Euler2Rotation, Quaternion2Euler


s = np.sin
c = np.cos
t = np.tan

def get_ned_dt_quat(state: types.DynamicState) -> tuple[float, float, float]:
    """Gets the ned_dt values

    Args:
        state (types.DynamicStateEuler): state
    Returns:
        tuple: tuple containing (n_dt, e_dt, d_dt)
    """

    quat = np.array([
        [state.e0],
        [state.e1],
        [state.e2],
        [state.e3],
    ])

    phi, theta, psi = Quaternion2Euler(quat)

    # --- NED DT
    ned_motion_mat = np.array([
        [c(theta)*c(psi), s(phi)*s(theta)*c(psi) - c(phi)*s(psi), c(phi)*s(theta)*c(psi) + s(phi)*s(psi)],
        [c(theta)*s(psi), s(phi)*s(theta)*s(psi) + c(phi)*c(psi), c(phi)*s(theta)*s(psi) - s(phi)*c(psi)],
        [-s(theta), s(phi)*c(theta), c(phi)*c(theta)]
    ])

    uvw_vec = np.array([
        [state.u],
        [state.v],
        [state.w]
        ])

    ned_dt = ned_motion_mat@uvw_vec
    ned_dt_flat = ned_dt.flatten()

    return ned_dt_flat[0], ned_dt_flat[1], ned_dt_flat[2]

def get_ned_dt(state: types.DynamicStateEuler) -> tuple[float, float, float]:
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

def get_uvw_dt(state: types.DynamicStateEuler, forces_moments: types.ForceMoment) -> tuple[float, float, float]:
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

def get_phi_theta_psi_dt(state: types.DynamicStateEuler) -> tuple[float, float, float]:
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

def get_pqr_dt(state: types.DynamicStateEuler, forces_moments: types.ForceMoment) -> tuple[float, float, float]:
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

def get_quaternion_dt(state: types.DynamicState) -> tuple[float, float, float, float]:
    """gets the quaternion dt

    Args:
        state (DynamicState): dynamic state of the mav
    Returns:
        tuple: tuple with (e0, e1, e2, e3)
    """

    quat_mat = 0.5*np.array([
        [0, -state.p, -state.q, -state.r],
        [state.p, 0, state.r, -state.q],
        [state.q, -state.r, 0, state.p],
        [state.r, state.q, -state.p, 0]
    ])

    quat_vec = np.array([
        [state.e0],
        [state.e1],
        [state.e2],
        [state.e3],
    ])

    res = quat_mat@quat_vec

    return res[0], res[1], res[2], res[3] 