"""msg_sim_params.py defines the parameters needed for simulation
"""
# pylint: disable=too-many-arguments

from typing import Optional


class MsgSimParams:
    """
    Provides parameters for running a simulation
    """
    def __init__(self, ts_simulation: float = 0.01, start_time: float = 0., end_time: float = 100., \
            ts_plotting: float = 0.1, ts_control: Optional[float] = None ) -> None:
        """ Store parameters
        """
        self.ts_simulation: float = ts_simulation   # smallest time step for simulation
        self.start_time: float = start_time         # start time for simulation
        self.end_time: float = end_time             # end time for simulation
        self.ts_plotting: float = ts_plotting       # refresh rate for plots

        self.ts_control: float = ts_simulation      # sample rate for the controller
        if ts_control is not None:
            self.ts_control = ts_control
