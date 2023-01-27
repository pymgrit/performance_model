import ast
import sys
import numpy as np
from pymgrit.heat.heat_1d import Heat1D
from pymgrit.core.mgrit import Mgrit

from mpi4py import MPI


def rhs(x, t):
    """
    Right-hand side of 1D heat equation example problem at a given space-time point (x,t),
      -sin(pi*x)(sin(t) - a*pi^2*cos(t)),  a = 1

    Note: exact solution is np.sin(np.pi * x) * np.cos(t)
    :param x: spatial grid point
    :param t: time point
    :return: right-hand side of 1D heat equation example problem at point (x,t)
    """

    return - np.sin(np.pi * x) * (np.sin(t) - 1 * np.pi ** 2 * np.cos(t))


def init_cond(x):
    """
    Initial condition of 1D heat equation example,
      u(x,0)  = sin(pi*x)

    :param x: spatial grid point
    :return: initial condition of 1D heat equation example problem
    """
    return np.sin(np.pi * x)


def run(nt, coarsening, nested_iter, cf_iter, iters, conv_crit, cycle_type):
    rank = MPI.COMM_WORLD.Get_rank()
    problem = [Heat1D(x_start=0, x_end=1, nx=4097, a=1, init_cond=init_cond, rhs=rhs, t_start=0, t_stop=2, nt=nt)]
    transfer = []
    for i in range(len(coarsening) - 1):
        problem.append(Heat1D(t_interval=problem[-1].t[::coarsening[i]],
                              x_start=0, x_end=1, nx=4097, a=1, init_cond=init_cond, rhs=rhs))

    MPI.COMM_WORLD.barrier()
    mgrit = MgritNew(problem=problem,
                     tol=-1,
                     max_iter=iters,
                     nested_iteration=nested_iter,
                     cf_iter=cf_iter,
                     conv_crit=conv_crit,
                     cycle_type=cycle_type)
    info = mgrit.solve()
    for i in range(len(coarsening)):
        if len(mgrit.problem[i].save) > 0:
            for key, value in mgrit.problem[i].save.items():
                print(f'{key}: len {len(value)} '
                      f'level {i} avg {np.average(value)} '
                      f'min {np.min(value)} '
                      f'max {np.max(value)}')


def main():
    stdin = sys.argv
    if len(stdin) == 13:
        nt = int(stdin[1])
        coarsening = ast.literal_eval(stdin[2])
        cf_iter = ast.literal_eval(stdin[3])
        nested_iter = int(stdin[4])
        iters = int(stdin[5])
        conv_crit = int(stdin[6])
        size = ast.literal_eval(stdin[7])
        sleep_step = ast.literal_eval(stdin[8])
        sleep_pro = ast.literal_eval(stdin[9])
        sleep_res = ast.literal_eval(stdin[10])
        print_timings = int(stdin[11])
        cycle_type = str(stdin[12])
    else:
        nt = 17
        coarsening = [2, 2, 1]
        cf_iter = [2, 1, 0]
        nested_iter = 1
        iters = 2
        conv_crit = 1
        cycle_type = 'V'

    run(nt=nt,
        coarsening=coarsening,
        cf_iter=cf_iter,
        nested_iter=nested_iter,
        iters=iters,
        conv_crit=conv_crit,
        cycle_type=cycle_type)


if __name__ == '__main__':
    main()
