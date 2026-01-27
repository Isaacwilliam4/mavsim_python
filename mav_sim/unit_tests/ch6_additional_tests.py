"""This module provides some additional tests for chapter 6 that can be used to help you
   debug if you are not passing the unit tests
"""

from mav_sim.chap6.autopilot import AP, Autopilot, MsgAutopilot, MsgState
from mav_sim.chap6.pd_control_with_rate import PDControlWithRate
from mav_sim.chap6.pi_control import PIControl
from mav_sim.chap6.tf_control import TFControl


def pd_control_with_rate_helper() -> None:
    """ This function seemingly does nothing. However, if you run it in the debug mode, you should be able to check the function for intermediary values.
    """

    # PD control class:
    pd = PDControlWithRate(kp=1.23, kd=4.56, limit=10.)

    # Calculate update
    #   error: 998.77
    #   u prior to saturation: 1217.8167
    #   u returned: 10.0
    u_sat = pd.update(y_ref=1000., y=1.23, ydot=2.34)

    print(u_sat)

def pi_control_helper() -> None:
    """ This function seemingly does nothing. However, if you run it in the debug mode, you should be able to check the function for intermediary values.
    """

    # PI control class
    pi = PIControl(kp=1.23, ki=4.56, limit=10.1)
    pi.integrator = 5.34
    pi.error_delay_1 = 1.2

    # Calculate update
    #   error: 96.11999999999999
    #   integrator before windup adjustment: 5.8266
    #   u unsaturated: 144.79689599999998
    #   u_sat: 10.1
    #   error_delay_1 at end of function:  96.11999999999999
    #   integrator after windup adjustment: 5.531212070175439
    u_sat = pi.update(y_ref=98.1, y=1.98)
    print(u_sat)


    # Calculate update
    #   error: 0.001000000000000112
    #   integrator before windup adjustment: 6.0118170701754385
    #   u unsaturated: 0.06134817070175453
    #   u_sat:  0.06134817070175453
    #   error_delay_1 at end of function: 0.001000000000000112
    #   integrator after windup adjustment: 6.0118170701754385
    pi.ki = 0.01
    u_sat = pi.update(y_ref=1.981, y=1.98)
    print(u_sat)

def tf_control_helper() -> None:
    """ This function seemingly does nothing. However, if you run it in the debug mode, you should be able to check the function for intermediary values.
    """

    # TF Controller
    tf = TFControl(k=.12, n0=1.2, n1=3.4, d0=1.32, d1=3.21, Ts=0.1, limit=4.3)

    # Update the controller
    #   u_unsaturated: 6.587941391941391
    #   u_saturated: 4.3
    #   self.y_delay_1: 51.98
    #   self.u_delay_1: 4.3
    u_sat = tf.update(y = 51.98)
    print(u_sat)

    # Update the controller
    #   u_unsaturated: -2.2314505494505497
    #   u_saturated: -2.2314505494505497
    #   self.y_delay_1: 0.01
    #   self.u_delay_1: -2.2314505494505497
    u_sat = tf.update(y = 0.01)
    print(u_sat)

def autopilot_helper() -> None:
    """ This function seemingly does nothing. However, if you run it in the debug mode, you should be able to check the function for intermediary values.
    """
    # Gain products
    #   AP.roll_kd*AP.roll_kp = -0.049553910174212706
    #   AP.course_ki*AP.course_kp = 1.623611098122222
    #   AP.pitch_kd*AP.pitch_kp*AP.wn_theta_squared*AP.K_theta_DC = 190.84756669551643
    #   AP.altitude_ki*AP.altitude_kp = 0.001294910430089064
    #   AP.airspeed_throttle_ki*AP.airspeed_throttle_kp = 0.1985748591270046
    t1 = AP.roll_kd*AP.roll_kp
    t2 = AP.course_ki*AP.course_kp
    t3 = AP.pitch_kd*AP.pitch_kp*AP.wn_theta_squared*AP.K_theta_DC
    t4 = AP.altitude_ki*AP.altitude_kp
    t5 = AP.airspeed_throttle_ki*AP.airspeed_throttle_kp
    print(t1, t2, t3, t4, t5)


    # Create an autopilot update
    #   commanded course angle = 1.8568146928204143
    #   commanded phi = 0.5235987755982988
    #   aileron = 0.24789008504047294
    #
    #   rudder = 0.02454477425791968
    #
    #   commanded altitude = 100.0
    #   commanded theta = 0.5235987755982988
    #   elevator = 0.7853981633974483
    #
    #   throttle = 1.0

    ap = Autopilot(ts_control=0.01)
    cmd = MsgAutopilot()            # Generate the command
    cmd.airspeed_command = 35.
    cmd.altitude_command = 500.
    cmd.course_command = 8.14
    cmd.phi_feedforward = 1.23
    st = MsgState(north=1.23,
                  east=3.45,
                  altitude=90.,
                  phi=0.2,
                  theta=1.45,
                  psi=2.5,
                  Va=25.6,
                  alpha=0.45,
                  beta=.12,
                  p = .01,
                  q = .2,
                  r = .123,
                  Vg = 24.5,
                  gamma=-.12,
                  chi = 2.8,
                  wn=.1,
                  we=5.6,
                  bx = .12,
                  by=.23,
                  bz = .45)
    delta, cmd_state = ap.update(cmd=cmd, state=st)
    print(delta, cmd_state)


    # Create an autopilot update
    #   commanded course angle = 2.7
    #   commanded phi = -0.14233925070147987
    #   aileron = -0.2609113457229402
    #
    #   rudder = 0.024434570731693053
    #
    #   commanded altitude = 91.1
    #   commanded theta = 0.07909492091608104
    #   elevator = 0.7853981633974483
    #
    #   throttle = 1.0
    cmd.airspeed_command = 34.9
    cmd.altitude_command = 91.1
    cmd.course_command = 2.7
    cmd.phi_feedforward = 0.1
    delta, cmd_state = ap.update(cmd=cmd, state=st)
    print(delta, cmd_state)




if __name__ == "__main__":
    #pd_control_with_rate_helper()
    #pi_control_helper()
    #tf_control_helper()
    autopilot_helper()
