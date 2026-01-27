"""This module provides some additional tests for chapter 5 that can be used to help you
   debug if you are not passing the unit tests
"""
import numpy as np
from mav_sim.chap5.trim import trim_objective_fun


def trim_objective_fun_helper() -> None:
    """ This function seemingly does nothing. However, if you run it in debug mode, you should be
        able to check the function for intermediary values. You should get the following values:
            desired_trim_state_dot (i.e., f_d):
                    [   [ 0.        ]
                        [ 0.        ]
                        [-3.6807027 ]
                        [ 0.        ]
                        [ 0.        ]
                        [ 0.        ]
                        [ 0.        ]
                        [ 0.        ]
                        [ 0.03721669]
                        [ 0.        ]
                        [ 0.        ]
                        [ 0.        ]]
            forces_moment_vec:
                    [   [11351.99021794]
                        [   78.46070672]
                        [  -83.5720567 ]
                        [ -176.13431905]
                        [ -102.51488932]
                        [  -94.78716552]]
            f (dynamics based on current state and input):
                   [    [-6.43738031e+00]
                        [ 3.10095259e+00]
                        [-5.09354767e+00]
                        [ 1.02599911e+03]
                        [ 1.91327915e+01]
                        [-1.35974597e+01]
                        [-1.00656326e+02]
                        [ 4.09085613e-01]
                        [-1.11846569e+02]
                        [-3.12650814e+02]
                        [ 1.31583354e+01]
                        [-1.03745949e+02]]
            delta (difference between desired and actual):
                    [   [ 6.43738031e+00]
                        [-3.10095259e+00]
                        [ 1.41284497e+00]
                        [-1.02599911e+03]
                        [-1.91327915e+01]
                        [ 1.35974597e+01]
                        [ 1.00656326e+02]
                        [-4.09085613e-01]
                        [ 1.11883786e+02]
                        [ 3.12650814e+02]
                        [-1.31583354e+01]
                        [ 1.03745949e+02]]
            J (square of the difference): 13690027.487823917

    """
    x = np.array([[1.], [2.], [3.], [4.], [5.], [6.], [7.], [8.],
                  [9.], [10.], [11.], [12.], [13.], [14.], [15.], [16.]])
    J = trim_objective_fun(x = x, Va = 30., gamma=.123, R = 800., psi_weight=1000.)
    print("J = ", J)

if __name__ == "__main__":
    trim_objective_fun_helper()
