"""ch10_path_follower_test.py: Implements some basic tests for the chapter 10 material."""
# pylint: disable=too-many-lines

import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
from mav_sim.chap10.path_follower import follow_orbit, follow_straight_line


#############  auto test run ######################
def follow_straight_line_test(test_cases: List[Tuple[Dict[str, Any], Any]]) -> bool:
    """Tests the follow_straight_line function."""
    print("Starting follow_straight_line test")
    succ = True
    for test_case_it in test_cases:
        calculated_output = follow_straight_line(**test_case_it[0])

        # Test typing
        good_type = True
        if(not isinstance(calculated_output.airspeed_command, float) or
           not isinstance(calculated_output.course_command, float) or
           not isinstance(calculated_output.altitude_command, float) or
           not isinstance(calculated_output.phi_feedforward, float)
        ):
            good_type = False # type: ignore[unreachable]

        # Test values
        if (
            (
                1e-12
                < np.abs(calculated_output.to_array() - test_case_it[1].to_array())
            ).any()
            or np.isnan(calculated_output.to_array()).any()
            or np.isinf(calculated_output.to_array()).any()
            or not good_type
        ):
            print("Failed test!")
            if not good_type:
                print("Output has incorrect type!")
            print("Calculated output:")
            print(calculated_output)
            print("Expected output:")
            print(test_case_it[1])
            print("Difference:")
            print(calculated_output.to_array() - test_case_it[1].to_array())
            succ = False
            break
    print("End of test\n")
    return succ

def follow_orbit_test(test_cases: List[Tuple[Dict[str, Any], Any]]) -> bool:
    """Tests the follow_orbit function."""
    print("Starting follow_orbit test")
    succ = True
    for test_case_it in test_cases:
        calculated_output = follow_orbit(**test_case_it[0])

        # Test typing
        good_type = True
        if(not isinstance(calculated_output.airspeed_command, float) or
           not isinstance(calculated_output.course_command, float) or
           not isinstance(calculated_output.altitude_command, float) or
           not isinstance(calculated_output.phi_feedforward, float)
        ):
            good_type = False # type: ignore[unreachable]

        # Test values
        if (
            (
                1e-12
                < np.abs(calculated_output.to_array() - test_case_it[1].to_array())
            ).any()
            or np.isnan(calculated_output.to_array()).any()
            or np.isinf(calculated_output.to_array()).any()
            or not good_type
        ):
            print("Failed test!")
            if not good_type:
                print("Output has incorrect type!")
            print("Calculated output:")
            print(calculated_output)
            print("Expected output:")
            print(test_case_it[1])
            print("Difference:")
            print(calculated_output.to_array() - test_case_it[1].to_array())
            succ = False
            break
    print("End of test\n")
    return succ

def run_all_tests() -> None:
    """Run all tests."""
    # Open archive
    with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "ch10_test_archive.pkl"
        ),
        "rb",
    ) as file:
        tests_archive = pickle.load(file)
    # Run tests
    succ = follow_straight_line_test(tests_archive["follow_straight_line"])
    if succ:
        succ = follow_orbit_test(tests_archive["follow_orbit"])

    if not succ:
        raise ValueError("Failed test")

if __name__ == "__main__":
    run_all_tests()
