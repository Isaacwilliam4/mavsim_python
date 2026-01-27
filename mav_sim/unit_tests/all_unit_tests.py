""" Runs the unit tests for all chapters. Note that the unpickling struggles if the tests are not imported. That is why all tests are imported here even if they are not used directly.
"""
from mav_sim.unit_tests.ch2_transforms_tests import (  # pylint: disable=unused-import
    SimpleRotationTest,
    ThreeRotationTest,
    TransformTest,
)
from mav_sim.unit_tests.ch2_transforms_tests import run_all_tests as run_02_tests
from mav_sim.unit_tests.ch3_derivatives_test import (  # pylint: disable=unused-import
    DynamicsResults,
)
from mav_sim.unit_tests.ch3_derivatives_test import run_tests as run_03_tests
from mav_sim.unit_tests.ch4_dynamics_test import (  # pylint: disable=unused-import
    ForcesMomentsTest,
    GravitationalForceTest,
    LateralDynamicsTest,
    LongitudinalDynamicsTest,
    MotorThrustTorqueTest,
    UpdateVelocityTest,
    WindSimulationTest,
)
from mav_sim.unit_tests.ch4_dynamics_test import run_all_tests as run_04_tests
from mav_sim.unit_tests.ch5_dynamics_test import (  # pylint: disable=unused-import
    TrimObjectiveFunTest,
    VariableBoundsTest,
    VelocityConstraintPartialTest,
    VelocityConstraintTest,
)
from mav_sim.unit_tests.ch5_dynamics_test import run_auto_tests as run_05_tests
from mav_sim.unit_tests.ch6_feedback_control_test import (  # pylint: disable=unused-import
    AutopilotTest,
    PDControlWithRateTest,
    PIControlTest,
    TFControlTest,
)
from mav_sim.unit_tests.ch6_feedback_control_test import run_all_tests as run_06_tests
from mav_sim.unit_tests.ch7_sensors_test import (
    run_all_tests as run_07_tests,  # pylint: disable=unused-import
)
from mav_sim.unit_tests.ch10_path_follower_test import run_all_tests as run_10_tests
from mav_sim.unit_tests.ch11_line_and_fillet_path_manager_test import (
    run_all_tests as run_11a_tests,
)
from mav_sim.unit_tests.ch11b_dubins_path_manager_test import (
    run_tests as run_11b_tests,  # pylint: disable=unused-import
)
from mav_sim.unit_tests.ch12_straight_line_rrt_test import run_tests as run_12_tests

if __name__ == '__main__':
    print("\n\nRunning Chapter 2 Unit Tests")
    run_02_tests()
    print("\n\nRunning Chapter 3 Unit Tests")
    run_03_tests()
    print("\n\nRunning Chapter 4 Unit Tests")
    run_04_tests()
    print("\n\nRunning Chapter 5 Unit Tests")
    run_05_tests()
    print("\n\nRunning Chapter 6 Unit Tests")
    run_06_tests()
    print("\n\nRunning Chapter 7 Unit Tests")
    run_07_tests()
    print("\n\nRunning Chapter 10 Unit Tests")
    run_10_tests()
    print("\n\nRunning Chapter 11a Unit Tests")
    run_11a_tests()
    print("\n\nRunning Chapter 11b Unit Tests")
    run_11b_tests()
    print("\n\nRunning Chapter 12 Unit Tests")
    run_12_tests()
