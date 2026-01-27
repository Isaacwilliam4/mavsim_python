"""00_Puthon_debugger.py Provides and example of a typical python code executable structure and provides you experience with debug features
"""
import numpy as np


def debug_this_function() -> None:
    """This function needs debugging. You might see the issue right away, but make sure you use the debugger to gain experience
    """
    # Define two column vectors
    v1 = np.array([[1], [2], [3]])
    v2 = np.array([[4], [5], [6]])

    # Use matrix multiplication to get the results of the dot product between the two matrices
    # Run in debug mode and use different features of the debug tool
    res = v1.T@v2
    print("resulting multiplication: ", res)


if __name__ == "__main__":
    # This is the entry point for an executable
    debug_this_function()