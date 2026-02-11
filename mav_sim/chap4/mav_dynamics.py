"""
mavDynamics
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state

part of mavPySim
    - Beard & McLain, PUP, 2012
    - Update history:
        12/20/2018 - RWB
"""
from typing import Optional, cast

import mav_sim.parameters.aerosonde_parameters as MAV
import numpy as np

# load mav dynamics from previous chapter
from mav_sim.chap3.mav_dynamics import IND, DynamicState, ForceMoments, derivatives
from mav_sim.message_types.msg_delta import MsgDelta

# load message types
from mav_sim.message_types.msg_state import MsgState
from mav_sim.tools import types
from mav_sim.tools.rotations import Quaternion2Euler, Quaternion2Rotation


class MavDynamics:
    """Implements the dynamics of the MAV using vehicle inputs and wind
    """

    def __init__(self, Ts: float, state: Optional[DynamicState] = None):
        self._ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        # We will also need a variety of other elements that are functions of the _state and the wind.
        # self.true_state is a 19x1 vector that is estimated and used by the autopilot to control the aircraft:
        # true_state = [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        if state is None:
            self._state = DynamicState().convert_to_numpy()
        else:
            self._state = state.convert_to_numpy()

        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec

        # update velocity data
        (self._Va, self._alpha, self._beta, self._wind) = update_velocity_data(self._state)

        # Update forces and moments data
        self._forces = np.array([[0.], [0.], [0.]]) # store forces to avoid recalculation in the sensors function (ch 7)
        self._moments = np.array([[0.], [0.], [0.]]) # store moments to avoid recalculation
        forces_moments_vec = forces_moments(self._state, MsgDelta(), self._Va, self._beta, self._alpha)
        self._forces[0] = forces_moments_vec.item(0)
        self._forces[1] = forces_moments_vec.item(1)
        self._forces[2] = forces_moments_vec.item(2)
        self._moments[0] = forces_moments_vec.item(3)
        self._moments[1] = forces_moments_vec.item(4)
        self._moments[2] = forces_moments_vec.item(5)


        # initialize true_state message
        self.true_state = MsgState()
        self._update_true_state()

    @property
    def forces(self) -> types.Vector:
        """Getter for the forces variable"""
        return self._forces

    @property
    def moments(self) -> types.Vector:
        """Getter for the moments variable"""
        return self._moments

    @property
    def ts_simulation(self)->float:
        """Getter for the ts_simulation"""
        return self._ts_simulation

    @ts_simulation.setter
    def ts_simulation(self, value: float) -> None:
        """Sets the time step as long as it is greater than zero"""
        if value > 0:
            self._ts_simulation = value



    ###################################
    # public functions
    def update(self, delta: MsgDelta, wind: types.WindVector, time_step: Optional[float] = None) -> None:
        """
        Integrate the differential equations defining dynamics, update sensors

        Args:
            delta : (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind: the wind vector in inertial coordinates
        """
        # get forces and moments acting on rigid bod
        forces_moments_vec = forces_moments(self._state, delta, self._Va, self._beta, self._alpha)
        self._forces[0] = forces_moments_vec.item(0)
        self._forces[1] = forces_moments_vec.item(1)
        self._forces[2] = forces_moments_vec.item(2)
        self._moments[0] = forces_moments_vec.item(3)
        self._moments[1] = forces_moments_vec.item(4)
        self._moments[2] = forces_moments_vec.item(5)

        # Get the timestep
        if time_step is None:
            time_step = self._ts_simulation

        # Integrate ODE using Runge-Kutta RK4 algorithm
        k1 = derivatives(self._state, forces_moments_vec)
        k2 = derivatives(self._state + time_step/2.*k1, forces_moments_vec)
        k3 = derivatives(self._state + time_step/2.*k2, forces_moments_vec)
        k4 = derivatives(self._state + time_step*k3, forces_moments_vec)
        self._state += time_step/6 * (k1 + 2*k2 + 2*k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(IND.E0)
        e1 = self._state.item(IND.E1)
        e2 = self._state.item(IND.E2)
        e3 = self._state.item(IND.E3)
        norm_e = np.sqrt(e0**2+e1**2+e2**2+e3**2)
        self._state[IND.E0][0] = self._state.item(IND.E0)/norm_e
        self._state[IND.E1][0] = self._state.item(IND.E1)/norm_e
        self._state[IND.E2][0] = self._state.item(IND.E2)/norm_e
        self._state[IND.E3][0] = self._state.item(IND.E3)/norm_e

        # update the airspeed, angle of attack, and side slip angles using new state
        (self._Va, self._alpha, self._beta, self._wind) = update_velocity_data(self._state, wind)

        # update the message class for the true state
        self._update_true_state()

    def external_set_state(self, new_state: types.DynamicState) -> None:
        """Loads a new state
        """
        self._state = new_state

    def get_state(self) -> types.DynamicState:
        """Returns the state
        """
        return self._state

    def get_struct_state(self) ->DynamicState:
        '''Returns the current state in a struct format

        Outputs:
            DynamicState: The latest state of the mav
        '''
        return DynamicState(self._state)

    def get_fm_struct(self) -> ForceMoments:
        '''Returns the latest forces and moments calculated in dynamic update'''
        force_moment = np.zeros((6,1))
        force_moment[0:3] = self._forces
        force_moment[3:6] = self._moments
        return ForceMoments(force_moment= force_moment)

    def get_euler(self) -> tuple[float, float, float]:
        '''Returns the roll, pitch, and yaw Euler angles based upon the state'''
        # Get Euler angles
        phi, theta, psi = Quaternion2Euler(self._state[IND.QUAT])

        # Return angles
        return (phi, theta, psi)

    ###################################
    # private functions
    def _update_true_state(self) -> None:
        """ update the class structure for the true state:

        [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        """
        quat = self._state[IND.QUAT]
        phi, theta, psi = Quaternion2Euler(quat)
        pdot = Quaternion2Rotation(quat) @ self._state[IND.VEL]
        self.true_state.north = self._state.item(IND.NORTH)
        self.true_state.east = self._state.item(IND.EAST)
        self.true_state.altitude = -self._state.item(IND.DOWN)
        self.true_state.Va = self._Va
        self.true_state.alpha = self._alpha
        self.true_state.beta = self._beta
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = cast(float, np.linalg.norm(pdot))
        if self.true_state.Vg != 0.:
            self.true_state.gamma = np.arcsin(pdot.item(2) / self.true_state.Vg)
        else:
            self.true_state.gamma = 0.
        self.true_state.chi = np.arctan2(pdot.item(1), pdot.item(0))
        self.true_state.p = self._state.item(IND.P)
        self.true_state.q = self._state.item(IND.Q)
        self.true_state.r = self._state.item(IND.R)
        self.true_state.wn = self._wind.item(0)
        self.true_state.we = self._wind.item(1)

def sigma(alpha: float):
    num = 1 + np.exp(-MAV.M*(alpha - MAV.alpha0)) + np.exp(MAV.M*(alpha + MAV.alpha0))
    den = (1 + np.exp(-MAV.M*(alpha - MAV.alpha0)))*(1 + np.exp(MAV.M*(alpha + MAV.alpha0)))
    return num / den

def get_linear_coefficient() -> float:
    """Gets the linear coefficient, C_L_alpha

    Returns:
        float: linear coefficient
    """

    num = np.pi * MAV.AR
    den = 1 + np.sqrt(1 + (MAV.AR/2)**2)

    return num / den

def get_lift(alpha: float) -> float:
    """Returns the lift of the aircraft, C_L(alpha)

    Args:
        alpha (float): the attack angle

    Returns:
        float: lift
    """
    return (1 - sigma(alpha))*(MAV.C_L_0 + MAV.C_L_alpha*alpha) + sigma(alpha)*(2*np.sign(alpha)*np.sin(alpha)**2*np.cos(alpha))

def get_drag(alpha: float) -> float:
    """Gets the drag CD(alpha)

    Args:
        alpha (float): attack angle

    Returns:
        float: The drage force.
    """

    return MAV.C_D_p + (MAV.C_L_0 + MAV.C_L_alpha*alpha)**2 / (np.pi * MAV.e * MAV.AR)

def forces_moments(state: types.DynamicState, delta: MsgDelta, Va: float, beta: float, alpha: float) -> types.ForceMoment:
    """
    Return the forces on the UAV based on the state, wind, and control surfaces

    Args:
        state: current state of the aircraft
        delta: flap and thrust commands
        Va: Airspeed
        beta: Side slip angle
        alpha: Angle of attack

    Returns:
        Forces and Moments on the UAV (in body frame) np.matrix(fx, fy, fz, Mx, My, Mz)
    """
    if Va > 0:
        st = DynamicState(state=state)

        # Extract angular rates
        p = state.item(IND.P)
        q = state.item(IND.Q)
        r = state.item(IND.R)

        f_lift = 0.5* MAV.rho * (Va**2) * MAV.S_wing * get_lift(alpha)
        f_drag = 0.5* MAV.rho * (Va**2) * MAV.S_wing * get_drag(alpha)
        C_m = MAV.C_m_0 + MAV.C_m_alpha

        force_vec = np.array([
            [f_lift],
            [f_drag]
        ])

        rot_arr = np.array([
            [np.cos(alpha), -np.sin(alpha)],
            [np.sin(alpha), np.cos(alpha)]
        ])

        res = rot_arr @ force_vec
        res = res.flatten()
        f_x, f_z = res[0], res[1]

        delta_a, delta_r, delta_e = delta.aileron, delta.rudder, delta.elevator

        m = 0.5 * MAV.rho * (Va**2) * MAV.S_wing * MAV.c * (MAV.C_m_0 + MAV.C_m_alpha*alpha + MAV.C_m_q * (MAV.c / 2*Va) * q + MAV.C_m_delta_e * delta_e)


        # Return combined vector
        force_torque_vec = np.array([
            [f_x],
            [f_y],
            [f_z],
            [l],
            [m],
            [n]
        ])
    else:
        force_torque_vec = np.array([
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.]
        ])
    
    return force_torque_vec

def gravitational_force(quat: types.Quaternion) -> types.Vector:
    """ Computes the gravitational force on the aircraft in the body frame

    Args:
        quat: 4x1 quaternion vector

    Returns:
        types.Vector: The gravitational force due to gravity
    """
    # compute gravitaional forces in body frame
    R = Quaternion2Rotation(quat).T # rotation from body to world frame
    f_g = R@np.array([[0.], [0.], [MAV.mass*MAV.gravity]])# Force of gravity in body frame
    return f_g

def lateral_aerodynamics(p: float, r: float,
                         Va: float, beta: float,
                         aileron: float, rudder: float
                         ) -> tuple[types.Vector, types.Vector]:
    """ Computes the lateral aerodynamic force and torque in the body frame, each as a 3x1 vector

    Note that the aerodynamic parameters are obtained from MAV (imported above).
    For example, to get the lateral aerodynamic coefficient for force due to side slip angle,
    you would use
        MAV.C_Y_beta

    Args:
        p: roll rate - body frame - i
        r: yaw rate - body frame - k
        Va: Airspeed
        beta: Side slip angle
        alpha: Angle of attack
        aileron: aileron command
        rudder: rudder command

    Returns:
        tuple[types.Vector, types.Vector]: (f_lat,torque_lat)
            f_lat: The lateral aerodynamic force
            torque_lat: The lateral aerodynamic torque
    """
    # intermediate variables
    delta_a, delta_r = aileron, rudder

    if Va == 0.:
        f_y = 0.0
        l = 0.0
        n = 0.0
    else:
        f_y = 0.5 * MAV.rho * (Va**2) * MAV.S_wing * (MAV.C_Y_0 + MAV.C_Y_beta*beta + MAV.C_Y_p*(MAV.b/(2*Va))*p + MAV.C_Y_r*(MAV.b/(2*Va))*r + MAV.C_Y_delta_a*delta_a + MAV.C_Y_delta_r*delta_r)
        l = 0.5 * MAV.rho * (Va**2) * MAV.S_wing * MAV.b * (MAV.C_ell_0 + MAV.C_ell_beta*beta + MAV.C_ell_p*(MAV.b/(2*Va))*p + MAV.C_ell_r*(MAV.b/(2*Va))*r + MAV.C_ell_delta_a*delta_a + MAV.C_ell_delta_r*delta_r)
        n = 0.5 * MAV.rho * (Va**2) * MAV.S_wing * MAV.b * (MAV.C_n_0 + MAV.C_n_beta*beta + MAV.C_n_p*(MAV.b/(2*Va))*p + MAV.C_n_r*(MAV.b/(2*Va))*r + MAV.C_n_delta_a*delta_a + MAV.C_n_delta_r*delta_r)


    # compute lateral forces in body frame
    f_lat = np.array([[0.],
                      [f_y],
                      [0.]  ])

    # compute lateral torques in body frame
    torque_lat = np.array([[l], [.0], [n]])

    return (f_lat, torque_lat)

def longitudinal_aerodynamics(q: float,
                         Va: float, alpha: float,
                         elevator: float
                         ) -> tuple[types.Vector, types.Vector]:
    """ Computes the longitudinal aerodynamic force and torque in the body frame, each as a 3x1 vector.

    The lift model used combines the common linear behavior with a flat plat model for stall (4.9 - 4.10)

    The drag model combines parasitic and induced drag (4.11)

    Note that the aerodynamic parameters are obtained from MAV (imported above).
    For example, to get the aerodynamic coefficient of lift due to the angle of attack,
    you would use
        MAV.C_L_alpha

    Args:
        p: roll rate - body frame - i
        r: yaw rate - body frame - k
        Va: Airspeed
        beta: Side slip angle
        alpha: Angle of attack
        aileron: aileron command
        rudder: rudder command

    Returns:
        tuple[types.Vector, types.Vector]: (f_lon,torque_lon)
            f_lon: The lateral aerodynamic force
            torque_lon: The lateral aerodynamic torque
    """

    # intermediate variables
    if Va == 0.:
        f_x = 0.0
        f_z = 0.0
        m = 0.0
    else:
        # Extract angular rates
        C_L = get_lift(alpha)
        C_D = get_drag(alpha)
        f_lift = 0.5* MAV.rho * (Va**2) * MAV.S_wing * (C_L + MAV.C_L_q*(MAV.c / (2*Va)) * q + MAV.C_L_delta_e*elevator)
        f_drag = 0.5* MAV.rho * (Va**2) * MAV.S_wing * (C_D + MAV.C_D_q*(MAV.c / (2*Va)) * q + MAV.C_D_delta_e*elevator)

        force_vec = np.array([
            [-f_drag],
            [-f_lift]
        ])

        rot_arr = np.array([
            [np.cos(alpha), -np.sin(alpha)],
            [np.sin(alpha), np.cos(alpha)]
        ])

        res = rot_arr @ force_vec
        res = res.flatten()
        f_x, f_z = res[0], res[1]

        delta_e = elevator

        m = 0.5 * MAV.rho * (Va**2) * MAV.S_wing * MAV.c * (MAV.C_m_0 + MAV.C_m_alpha*alpha + MAV.C_m_q * (MAV.c / (2*Va)) * q + MAV.C_m_delta_e * delta_e)

    # compute longitudinal forces in body frame
    f_lon = np.array([[f_x], [0.], [f_z]])

    # compute logitudinal torque in body frame (see (4.5) )
    torque_lon = np.array([[0.], [m], [0.]])

    return (f_lon, torque_lon)

def motor_thrust_torque(Va: float, delta_t: float) -> tuple[float, float]:
    """ compute thrust and torque due to propeller  (See addendum by McLain)

    Args:
        Va: Airspeed
        delta_t: Throttle command

    Returns:
        T_p: Propeller thrust
        Q_p: Propeller torque
    """
    # thrust and torque due to propeller
    thrust_prop = 0.
    torque_prop = 0.
    print('mav_dynamics::motor_thrust_torque() Needs to be implemented')
    return thrust_prop, torque_prop

def update_velocity_data(state: types.DynamicState, \
    wind: types.WindVector = np.zeros((6,1))  \
    )  -> tuple[float, float, float, types.NP_MAT]:
    """Calculates airspeed, angle of attack, sideslip, and velocity wrt wind

    Args:
        state: current state of the aircraft

    Returns:
        Va: Airspeed
        alpha: Angle of attack
        beta: Side slip angle
        wind_inertial_frame: Wind vector in inertial frame
    """
    # Calculate wind
    steady_state = wind[0:3]
    gust = wind[3:6]

    # convert wind vector from world to body frame
    R = Quaternion2Rotation(state[IND.QUAT]) # rotation from body to world frame
    wind_body_frame = R.T @ steady_state  # rotate steady state wind to body frame
    wind_body_frame += gust  # add the gust
    wind_inertial_frame = R @ wind_body_frame # Wind in the world frame

    # compute airspeed
    Va = 50.

    # compute angle of attack
    alpha = 0.

    # compute sideslip angle
    beta = 0.

    # Return computed values
    print('mav_dynamics::update_velocity_data() Needs to be implemented')
    return (Va, alpha, beta, wind_inertial_frame)
