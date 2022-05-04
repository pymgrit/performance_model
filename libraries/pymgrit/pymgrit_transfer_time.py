import time

from pymgrit_app_time import VectorTime
from pymgrit.core.grid_transfer import GridTransfer  # Parent grid transfer class

from mpi4py import MPI


class PymgritTransferTime(GridTransfer):
    """
    Pseudo-problem spatial transfer
    Sleeps only
    """
    def __init__(self, res_sleep, pro_sleep, level, print_timings, size):
        """
        Constructor

        :param res_sleep: Sleep time restriction
        :param pro_sleep: Sleep time prolongation
        :param level: Level
        :param print_timings: Print timestamps
        :param size: Problem size
        """
        super().__init__()
        self.res_sleep = res_sleep
        self.pro_sleep = pro_sleep
        self.init_time = time.time()
        self.level = level
        self.print_timings = print_timings
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.size = size

    def restriction(self, u: VectorTime) -> VectorTime:
        """
        Spatial restriction (sleeps only)

        :param u: Fine approximation
        :return: Coarse approximation
        """
        start = time.time() - self.init_time
        tmp = VectorTime(size=self.size[self.level + 1])
        time.sleep(self.res_sleep)
        if self.print_timings:
            print("Model | Rank:", self.rank,
                  "| Start:", start,
                  "| Stop:", time.time() - self.init_time,
                  "| Type:", "Res_pymgrit_level_" + str(self.level),
                  "| size:", self.size[self.level], "->", self.size[self.level + 1])
        return tmp

    def interpolation(self, u: VectorTime) -> VectorTime:
        """
        Spatial interpolation (sleeps only)

        :param u: Coarse approximation
        :return: Fine approximation
        """
        start = time.time() - self.init_time
        tmp = VectorTime(size=self.size[self.level])
        time.sleep(self.pro_sleep)
        if self.print_timings:
            print("Model | Rank:", self.rank,
                  "| Start:", start,
                  "| Stop:", time.time() - self.init_time,
                  "| Type:", "Pro_pymgrit_level_" + str(self.level),
                  "| size:", self.size[self.level + 1], "->", self.size[self.level])
        return tmp
