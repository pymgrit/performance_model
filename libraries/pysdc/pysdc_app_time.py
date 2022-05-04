import numpy as np
import time

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh

from mpi4py import MPI

class PysdcAppTime(ptype):

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type for solution
            dtype_f: mesh data type for RHS
        """

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'sleep_time', 'level', 'sleep_f', 'print_timings']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(init=(problem_params['nvars'], None, np.dtype('float64')),
                         dtype_u=dtype_u, dtype_f=dtype_f, params=problem_params)
        self.init_time = time.time()
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.nvars = problem_params['nvars']
        self.level = problem_params['level']
        self.print_timings = problem_params['print_timings']
        self.sleep_time = problem_params['sleep_time']
        self.sleep_f = problem_params['sleep_f']
        self.name_step_op = "Step_pysdc_level_" + str(self.level)
        self.name_f_op = "F_pysdc_level_" + str(self.level)

    def eval_f(self, u, t):
        """
        Eval f (sleeps only)
        """
        start = time.time() - self.init_time
        f = self.dtype_f(self.init)
        time.sleep(self.sleep_f)
        if self.print_timings:
            print("Model | Rank:", self.rank,
                  "| Start:", start,
                  "| Stop:", time.time() - self.init_time,
                  "| Type:", self.name_f_op,
                  "| t_stop:", t,
                  "| size:", len(u))
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Pseudo-problem (sleeps only)
        """
        start = time.time() - self.init_time
        me = self.dtype_u(self.init)
        time.sleep(self.sleep_time)
        if self.print_timings:
            print("Model | Rank:", self.rank,
                  "| Start:", start,
                  "| Stop:", time.time() - self.init_time,
                  "| Type:", self.name_step_op,
                  "| t_stop:", t,
                  "| size:", len(u0))
        return me

    def initial(self, t):
        """
        Initial value
        """

        me = self.dtype_u(self.init)
        me[:] = np.random.rand(self.params.nvars)
        return me
