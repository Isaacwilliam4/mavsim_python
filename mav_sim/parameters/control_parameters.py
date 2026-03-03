"""control_parameter.py Defines the parameters used in control of the MAV
"""
import mav_sim.chap5.model_coef as TF
import mav_sim.parameters.aerosonde_parameters as MAV

gravity: float = MAV.gravity  # gravity constant
rho: float = MAV.rho  # density of air
sigma: float = 0.05  # low pass filter gain for derivative
Vg: float = TF.Va_trim # ground speed at trim condition

#----------roll loop-------------  Section 6.1.1.1
# get transfer function data for delta_a to phi
wn_roll: float = 10.0
zeta_roll: float = 0.707
roll_kp: float = (wn_roll**2 / TF.a_phi2) # Implement
roll_kd: float = (2 * zeta_roll * wn_roll - TF.a_phi1) / (TF.a_phi2)  # Implement

#----------course loop------------- Section 6.1.1.2
wn_course: float = wn_roll / 20.0
zeta_course: float = 1.0
course_kp: float = 2 * zeta_course * wn_course *  (Vg / gravity) # Implement
course_ki: float = wn_course**2 * (Vg / gravity) # Implement

#----------yaw damper------------- Section 6.1.1.4
yaw_damper_p_wo: float = 0.45
yaw_damper_kr: float = 0.2

#----------pitch loop------------- Section 6.1.2.1
wn_theta: float = 15.0
zeta_theta: float = 0.707
wn_theta_squared: float = wn_theta**2 # Created to avoid recomputation of the square
pitch_kp: float = (wn_theta**2 - TF.a_phi2) / (TF.a_theta3) # Implement
pitch_kd: float = (2 * zeta_theta * wn_theta - TF.a_theta1) / (TF.a_theta1) # Implement
K_theta_DC: float = (pitch_kp * TF.a_theta1) / (TF.a_theta2 + pitch_kp * TF.a_theta3) # Implement

#----------altitude loop------------- Section 6.1.2.2
wn_altitude: float = wn_theta / 30.0
zeta_altitude: float = 1.0
altitude_ki: float = (wn_altitude**2) / (K_theta_DC * TF.Va_trim) # Implement
altitude_kp: float = (2 * zeta_altitude * wn_altitude) / (K_theta_DC * TF.Va_trim) # Implement
altitude_zone: float = 10.0  # moving saturation limit around current altitude

#---------airspeed hold using throttle--------------- Section 6.1.2.3
wn_airspeed_throttle: float = 1.5
zeta_airspeed_throttle: float = 2.0
airspeed_throttle_ki: float = wn_airspeed_throttle**2 / TF.a_V2 # Implement
airspeed_throttle_kp: float = (2 * zeta_airspeed_throttle * wn_airspeed_throttle - TF.a_V1) / (TF.a_V2) # Implement
