import time

from pySDC.core.Errors import TransferError
from pySDC.core.SpaceTransfer import space_transfer
from pySDC.implementations.datatype_classes.mesh import mesh

from mpi4py import MPI


class PysdcTransfer(space_transfer):
    """
    Pseudo-problem transfer (sleeps only)
    """
    def __init__(self, fine_prob, coarse_prob, params):
        """
        Initialization routine
        """

        # invoke super initialization
        super().__init__(fine_prob, coarse_prob, params)
        self.level = fine_prob.level
        self.res_sleep = params['res_sleep'][self.level]
        self.pro_sleep = params['pro_sleep'][self.level]
        self.init_time = time.time()
        self.print_timings = params['print_timings']
        self.rank = MPI.COMM_WORLD.Get_rank()

    def restrict(self, F):
        """
        Restriction (sleeps only)

        :param F:
        :return:
        """
        if isinstance(F, mesh):
            start = time.time() - self.init_time
            G = self.coarse_prob.dtype_u(self.coarse_prob.init)
            time.sleep(self.res_sleep)
            if self.print_timings:
                print("Model | Rank:", self.rank,
                      "| Start:", start,
                      "| Stop:", time.time() - self.init_time,
                      "| Type:", "Res_pysdc_level_" + str(self.level),
                      "| size:", len(G))
        else:
            raise TransferError('Wrong data type for prolongation, got %s' % type(F))
        return G

    def prolong(self, G):
        """
        Prolongation (sleeps only)

        :param G:
        :return:
        """
        if isinstance(G, mesh):
            start = time.time() - self.init_time
            F = self.fine_prob.dtype_u(self.fine_prob.init)
            time.sleep(self.pro_sleep)
            if self.print_timings:
                print("Model | Rank:", self.rank,
                      "| Start:", start,
                      "| Stop:", time.time() - self.init_time,
                      "| Type:", "Pro_pysdc_level_" + str(self.level),
                      "| size:", len(F))
        else:
            raise TransferError('Wrong data type for prolongation, got %s' % type(G))
        return F
