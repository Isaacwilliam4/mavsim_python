"""ch3_derivatives_tests.py: Implements some basic tests for the dirivatives funciton."""

import os
import pickle

import numpy as np
from mav_sim.chap3.mav_dynamics import StateIndices, derivatives
from mav_sim.chap3.mav_dynamics_euler import derivatives_euler
from mav_sim.tools import types

IND = StateIndices()

#############  Test structure definitions ########################
class DynamicsResults:
    """Stores the results for the dynamics"""
    def __init__(self) -> None:
        # Inputs
        self.state: types.NP_MAT
        self.forces_moments: types.ForceMoment

        # Output
        self.xdot: types.NP_MAT

    def __str__(self) -> str:
        """Create a string from the class"""
        return f"Inputs:\nDynamicsResults(state={self.state}, forces_moments={self.forces_moments}, \nDesired Output:\n xdot={self.xdot})"

def euler_derivatives_test(tests: list[DynamicsResults]) -> bool:
    """Test the Euler derivatives, display invalid results
    """
    # Evaluate the results
    print("\nStarting derivatives_euler test")
    success = True
    for test in tests:
        # Generate the output
        xdot = derivatives_euler(state=test.state, forces_moments=test.forces_moments)

        # Compare the output with the result
        if xdot.shape != test.xdot.shape: # Test to make ensure that the shapes are the same
            print("\n\nFailed test!")
            print("state = \n", test.state, ", forces/moments = \n", test.forces_moments, ", expected: \n", test.xdot, ", received: \n", xdot)
            success = False
            break
        if not np.allclose(xdot, test.xdot):
            print("\n\nFailed test!")
            print("state = \n", test.state, ", forces/moments = \n", test.forces_moments, ", expected: \n", test.xdot, ", received: \n", xdot)
            success = False

    # Indicate success
    if success:
        print("Passed test on derivatives_euler")
    return success

def quat_derivatives_test(tests: list[DynamicsResults]) -> bool:
    """Test the Euler derivatives, display invalid results
    """
    # Evaluate the results
    print("\nStarting derivatives test")
    success = True
    for test in tests:
        # Generate the output
        xdot = derivatives(state=test.state, forces_moments=test.forces_moments)

        if xdot.shape != test.xdot.shape: # Test to make ensure that the shapes are the same
            print("\n\nFailed test!")
            print("state = \n", test.state, ", forces/moments = \n", test.forces_moments, ", expected: \n", test.xdot, ", received: \n", xdot)
            success = False
            break
        if not np.allclose(xdot, test.xdot):
            print("\n\nFailed test!")
            print("state = \n", test.state, ", forces/moments = \n", test.forces_moments, ", expected: \n", test.xdot, ", received: \n", xdot)
            success = False

    # Indicate success
    if success:
        print("Passed test on derivatives")
    return success

def run_tests()->None:
    """Runs all of the tests
    """
    # Load the tests
    with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "ch03_test_archive.pkl"
        ),
        "rb",
    ) as file:
        data = pickle.load(file)

        # Run the tests
        succ = euler_derivatives_test(tests=data["euler"]) #and \
                 #quat_derivatives_test(tests=data["quat"])

        if not succ:
            raise ValueError("Tests failed")

if __name__ == "__main__":
    # Run tests
    run_tests()
