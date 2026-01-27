"""
Runs a number of fixed unit tests from pickle files for chapter 2 transforms.
"""

import os
import pickle

import numpy as np
from mav_sim.chap2.transforms import (
    rot_b_to_s,
    rot_b_to_v,
    rot_s_to_w,
    rot_v1_to_v2,
    rot_v2_to_b,
    rot_v_to_b,
    rot_v_to_v1,
    rot_x,
    rot_y,
    rot_z,
    trans_b_to_i,
    trans_i_to_b,
    trans_i_to_v,
    trans_v_to_i,
)
from mav_sim.tools import types


#############  Test structure definitions ########################
class SimpleRotationTest:
    """
    Stores a test for the rotation matrix
    """
    # Input fields
    angle: float

    # Output fields
    output: types.RotMat

    def __str__(self) -> str:
        return f"Input: {self.angle}\nDesired output:\n{self.output}"

class ThreeRotationTest:
    """
    Stores a test for the rotation matrix with three angles
    """
    # Input fields
    phi: float
    theta: float
    psi: float

    # Output fields
    output: types.RotMat

    def __str__(self) -> str:
        return f"Input: phi={self.phi}, theta={self.theta}, psi={self.psi}\nDesired output:\n{self.output}"

class TransformTest:
    """
    Stores a test for the pose transformation
    """
    # Input fields
    north: float
    east: float
    altitude: float
    phi: float
    theta: float
    psi: float
    points: types.Points

    # Output fields
    output: types.Points

    def __str__(self) -> str:
        return (f"Input:\n north={self.north}, east={self.east}, altitude={self.altitude}, "
                f"phi={self.phi}, theta={self.theta}, psi={self.psi}, points=\n{self.points}\n"
                f"Desired output:\n{self.output}")


#############  auto test run ######################
def run_all_tests() -> None:
    """
    Runs each of the auto generated tests
    """
    # Load the tests
    with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "ch02_test_archive.pkl"
        ),
        "rb",
    ) as file:
        data = pickle.load(file)

        # Run the tests
        succ = rot_x_tests(tests=data["rot_x_tests"])
        if succ:
            rot_y_tests(tests=data["rot_y_tests"])
        if succ:
            rot_z_tests(tests=data["rot_z_tests"])
        if succ:
            rot_v_to_v1_tests(tests=data["rot_v_to_v1_tests"])
        if succ:
            rot_v1_to_v2_tests(tests=data["rot_v1_to_v2_tests"])
        if succ:
            rot_v2_to_b_tests(tests=data["rot_v2_to_b_tests"])
        if succ:
            rot_b_to_s_tests(tests=data["rot_b_to_s_tests"])
        if succ:
            rot_s_to_w_tests(tests=data["rot_s_to_w_tests"])
        if succ:
            rot_v_to_b_tests(tests=data["rot_v_to_b_tests"])
        if succ:
            rot_b_to_v_tests(tests=data["rot_b_to_v_tests"])
        if succ:
            trans_i_to_v_tests(tests=data["trans_i_to_v_tests"])
        if succ:
            trans_v_to_i_tests(tests=data["trans_v_to_i_tests"])
        if succ:
            trans_i_to_b_tests(tests=data["trans_i_to_b_tests"])
        if succ:
            trans_b_to_i_tests(tests=data["trans_b_to_i_tests"])

        if not succ:
            raise ValueError("Failed test")

def rot_x_tests(tests: list[SimpleRotationTest]) -> bool:
    """
    Runs the rotation matrix tests for rotation about x-axis
    """
    print("\nRunning rot_x_tests...")
    for test in tests:
        computed_output = rot_x(test.angle)
        if computed_output.shape != test.output.shape:
            print("\n\nFailed test! -- Incorrect size of output")
            print("test: \n", test, "\nactual result: \n", computed_output)
            return False
        if not np.allclose(computed_output, test.output):
            print("\n\nFailed test!")
            print("test: \n", test, "\nactual result: \n", computed_output)
            return False
    print("rot_x_tests passed")
    return True

def rot_y_tests(tests: list[SimpleRotationTest]) -> bool:
    """
    Runs the rotation matrix tests for rotation about y-axis
    """
    print("\nRunning rot_y_tests...")
    for test in tests:
        computed_output = rot_y(test.angle)
        if computed_output.shape != test.output.shape:
            print("\n\nFailed test! -- Incorrect size of output")
            print("test: \n", test, "\nactual result: \n", computed_output)
            return False
        if not np.allclose(computed_output, test.output):
            print("\n\nFailed test!")
            print("test: \n", test, "\nactual result: \n", computed_output)
            return False
    print("rot_y_tests passed")
    return True

def rot_z_tests(tests: list[SimpleRotationTest]) -> bool:
    """
    Runs the rotation matrix tests for rotation about z-axis
    """
    print("\nRunning rot_z_tests...")
    for test in tests:
        computed_output = rot_z(test.angle)
        if computed_output.shape != test.output.shape:
            print("\n\nFailed test! -- Incorrect size of output")
            print("test: \n", test, "\nactual result: \n", computed_output)
            return False
        if not np.allclose(computed_output, test.output):
            print("\n\nFailed test!")
            print("test: \n", test, "\nactual result: \n", computed_output)
            return False
    print("rot_z_tests passed")
    return True

def rot_v_to_v1_tests(tests: list[SimpleRotationTest]) -> bool:
    """
    Runs the rotation matrix tests for rotation from v to v1
    """
    print("\nRunning rot_v_to_v1_tests...")
    for test in tests:
        computed_output = rot_v_to_v1(test.angle)
        if computed_output.shape != test.output.shape:
            print("\n\nFailed test! -- Incorrect size of output")
            print("test: \n", test, "\nactual result: \n", computed_output)
            return False
        if not np.allclose(computed_output, test.output):
            print("\n\nFailed test!")
            print("test: \n", test, "\nactual result: \n", computed_output)
            return False
    print("rot_v_to_v1_tests passed")
    return True

def rot_v1_to_v2_tests(tests: list[SimpleRotationTest]) -> bool:
    """
    Runs the rotation matrix tests for rotation from v1 to v2
    """
    print("\nRunning rot_v1_to_v2_tests...")
    for test in tests:
        computed_output = rot_v1_to_v2(test.angle)
        if computed_output.shape != test.output.shape:
            print("\n\nFailed test! -- Incorrect size of output")
            print("test: \n", test, "\nactual result: \n", computed_output)
            return False
        if not np.allclose(computed_output, test.output):
            print("\n\nFailed test!")
            print("test: \n", test, "\nactual result: \n", computed_output)
            return False
    print("rot_v1_to_v2_tests passed")
    return True

def rot_v2_to_b_tests(tests: list[SimpleRotationTest]) -> bool:
    """
    Runs the rotation matrix tests for rotation from v2 to b
    """
    print("\nRunning rot_v2_to_b_tests...")
    for test in tests:
        computed_output = rot_v2_to_b(test.angle)
        if computed_output.shape != test.output.shape:
            print("\n\nFailed test! -- Incorrect size of output")
            print("test: \n", test, "\nactual result: \n", computed_output)
            return False
        if not np.allclose(computed_output, test.output):
            print("\n\nFailed test!")
            print("test: \n", test, "\nactual result: \n", computed_output)
            return False
    print("rot_v2_to_b_tests passed")
    return True

def rot_b_to_s_tests(tests: list[SimpleRotationTest]) -> bool:
    """
    Runs the rotation matrix tests for rotation from b to s
    """
    print("\nRunning rot_b_to_s_tests...")
    for test in tests:
        computed_output = rot_b_to_s(test.angle)
        if computed_output.shape != test.output.shape:
            print("\n\nFailed test! -- Incorrect size of output")
            print("test: \n", test, "\nactual result: \n", computed_output)
            return False
        if not np.allclose(computed_output, test.output):
            print("\n\nFailed test!")
            print("test: \n", test, "\nactual result: \n", computed_output)
            return False
    print("rot_b_to_s_tests passed")
    return True

def rot_s_to_w_tests(tests: list[SimpleRotationTest]) -> bool:
    """
    Runs the rotation matrix tests for rotation from s to w
    """
    print("\nRunning rot_s_to_w_tests...")
    for test in tests:
        computed_output = rot_s_to_w(test.angle)
        if computed_output.shape != test.output.shape:
            print("\n\nFailed test! -- Incorrect size of output")
            print("test: \n", test, "\nactual result: \n", computed_output)
            return False
        if not np.allclose(computed_output, test.output):
            print("\n\nFailed test!")
            print("test: \n", test, "\nactual result: \n", computed_output)
            return False
    print("rot_s_to_w_tests passed")
    return True

def rot_v_to_b_tests(tests: list[ThreeRotationTest]) -> bool:
    """
    Runs the rotation matrix tests for rotation from v to b
    """
    print("\nRunning rot_v_to_b_tests...")
    for test in tests:
        computed_output = rot_v_to_b(psi=test.psi, theta=test.theta, phi=test.phi)
        if computed_output.shape != test.output.shape:
            print("\n\nFailed test! -- Incorrect size of output")
            print("test: \n", test, "\nactual result: \n", computed_output)
            return False
        if not np.allclose(computed_output, test.output):
            print("\n\nFailed test!")
            print("test: \n", test, "\nactual result: \n", computed_output)
            return False
    print("rot_v_to_b_tests passed")
    return True

def rot_b_to_v_tests(tests: list[ThreeRotationTest]) -> bool:
    """
    Runs the rotation matrix tests for rotation from b to v
    """
    print("\nRunning rot_b_to_v_tests...")
    for test in tests:
        computed_output = rot_b_to_v(psi=test.psi, theta=test.theta, phi=test.phi)
        if computed_output.shape != test.output.shape:
            print("\n\nFailed test! -- Incorrect size of output")
            print("test: \n", test, "\nactual result: \n", computed_output)
            return False
        if not np.allclose(computed_output, test.output):
            print("\n\nFailed test!")
            print("test: \n", test, "\nactual result: \n", computed_output)
            return False
    print("rot_b_to_v_tests passed")
    return True

def trans_i_to_v_tests(tests: list[TransformTest]) -> bool:
    """
    Runs the transformation tests for transformation from inertial to vehicle frame
    """
    print("\nRunning trans_i_to_v_tests...")
    for test in tests:
        computed_output = trans_i_to_v(pose=test, p_i=test.points)
        if computed_output.shape != test.output.shape:
            print("\n\nFailed test! -- Incorrect size of output")
            print("test: \n", test, "\nactual result: \n", computed_output)
            return False
        if not np.allclose(computed_output, test.output):
            print("\n\nFailed test!")
            print("test: \n", test, "\nactual result: \n", computed_output)
            return False
    print("trans_i_to_v_tests passed")
    return True

def trans_v_to_i_tests(tests: list[TransformTest]) -> bool:
    """
    Runs the transformation tests for transformation from vehicle to inertial frame
    """
    print("\nRunning trans_v_to_i_tests...")
    for test in tests:
        computed_output = trans_v_to_i(pose=test, p_v=test.points)
        if computed_output.shape != test.output.shape:
            print("\n\nFailed test! -- Incorrect size of output")
            print("test: \n", test, "\nactual result: \n", computed_output)
            return False
        if not np.allclose(computed_output, test.output):
            print("\n\nFailed test!")
            print("test: \n", test, "\nactual result: \n", computed_output)
            return False
    print("trans_v_to_i_tests passed")
    return True

def trans_i_to_b_tests(tests: list[TransformTest]) -> bool:
    """
    Runs the transformation tests for transformation from inertial to body frame
    """
    print("\nRunning trans_i_to_b_tests...")
    for test in tests:
        computed_output = trans_i_to_b(pose=test, p_i=test.points)
        if computed_output.shape != test.output.shape:
            print("\n\nFailed test! -- Incorrect size of output")
            print("test: \n", test, "\nactual result: \n", computed_output)
            return False
        if not np.allclose(computed_output, test.output):
            print("\n\nFailed test!")
            print("test: \n", test, "\nactual result: \n", computed_output)
            return False
    print("trans_i_to_b_tests passed")
    return True

def trans_b_to_i_tests(tests: list[TransformTest]) -> bool:
    """
    Runs the transformation tests for transformation from body to inertial frame
    """
    print("\nRunning trans_b_to_i_tests...")
    for test in tests:
        computed_output = trans_b_to_i(pose=test, p_b=test.points)
        if computed_output.shape != test.output.shape:
            print("\n\nFailed test! -- Incorrect size of output")
            print("test: \n", test, "\nactual result: \n", computed_output)
            return False
        if not np.allclose(computed_output, test.output):
            print("\n\nFailed test!")
            print("test: \n", test, "\nactual result: \n", computed_output)
            return False
    print("trans_b_to_i_tests passed")
    return True

if __name__ == "__main__":
    run_all_tests()
