import numpy as np
import time

from pymgrit.core.application import Application
from pymgrit.core.vector import Vector


class VectorTime(Vector):
    """
    Vector class for the pseudo-problem
    """

    def __init__(self, size):
        super().__init__()
        self.size = size
        self.value = np.ones(size)

    def __add__(self, other):
        tmp = VectorTime(len(self.value))
        tmp.set_values(self.get_values()+other.get_values())
        return tmp

    def __sub__(self, other):
        tmp = VectorTime(len(self.value))
        tmp.set_values(self.get_values()+other.get_values())
        return tmp

    def __mul__(self, other):
        tmp = VectorTime(len(self.value))
        tmp.set_values(self.get_values()*other)
        return tmp

    def norm(self):
        return 1

    def clone(self):
        return VectorTime(self.size)

    def clone_zero(self):
        return VectorTime(self.size)

    def clone_rand(self):
        return VectorTime(self.size)

    def set_values(self, value):
        self.value = value

    def get_values(self):
        return self.value

    def pack(self):
        return self.value

    def unpack(self, value):
        self.value = value


class PymgritAppTime(Application):
    """
    Pseudo-problem
    """
    def __init__(self: object, runtime: float, rank: int, size: int, level: int, print_timings: bool, *args, **kwargs) -> None:
        """
        Constructor

        :param self:
        :param runtime: Sleep time per time integration
        :param rank: Process rank
        :param size: Size of problem
        :param level: Level
        :param print_timings: Print timestamps
        """
        super().__init__(*args, **kwargs)
        self.size = size
        self.rank = rank
        self.vector_template = VectorTime(self.size)  # Set the class to be used for each time point
        self.vector_t_start = VectorTime(self.size)  # Set the initial condition
        self.runtime = runtime
        self.init_time = time.time()
        self.level = level
        self.print_timing = print_timings

    def step(self, u_start: VectorTime, t_start: float, t_stop: float) -> VectorTime:
        """
        Time integrations, sleeps only

        :param u_start: Initial approximation
        :param t_start: Start time
        :param t_stop: End time
        :return:
        """
        start = time.time() - self.init_time
        tmp = VectorTime(self.size)
        time.sleep(self.runtime)
        if self.print_timing:
            print("Model | Rank:", self.rank,
                  "| Start:", start,
                  "| Stop:", time.time() - self.init_time,
                  "| Type:", "Step_pymgrit_level_" + str(self.level),
                  "| T_start:", t_start,
                  "| t_stop:", t_stop,
                  "| size:", self.size)
        return tmp
