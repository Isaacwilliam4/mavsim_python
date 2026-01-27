"""This module provides some additional tests for chapter 4 that can be used to help you
   debug if you are not passing the unit tests
"""
from mav_sim.chap4.mav_dynamics import longitudinal_aerodynamics, motor_thrust_torque


def longitudinal_aerodynamics_helper() -> None:
    """ This functions seemingly does nothing. However, if you run it in debug mode, you should be
        able to check the function for intermediary values. You should get the following values:
            CL = 0.7410005607025906
            CD = 3.7260788511899077
            F_lift = 0.07385052440528686
            F_drag = 0.27076556225480797
            f_lon = [[0.12533388054772146], [0.], [0.25111612462155364] ]
            torque_lon = [ [0.], [0.0657766245422738], [0.]]
    """

    f_lon, t_lon = longitudinal_aerodynamics(q= .123, Va = .456, alpha=-2.3, elevator=.567)
    print("f_lon = \n", f_lon)
    print("t_lon = \n", t_lon)

def motor_thrust_torque_helper() -> None:
    """ This function seemingly does nothing. However, if you run it in debug mode, you should be
        able to check the function for intermediary values. You should get the following values:
            a = 5.683924001284114e-06
            b = 0.10408774785819767
            c = -32.06670169995605
            omega_p = 303.05838902739146
            thrust_prop = 7.081339420286007
            torque_prop = 0.3525206069786054
    """
    T_p, Q_p = motor_thrust_torque(Va=12.3, delta_t=.456)
    print("T_p = ", T_p, ", Q_p = ", Q_p)

if __name__ == "__main__":
    longitudinal_aerodynamics_helper()
    motor_thrust_torque_helper()
