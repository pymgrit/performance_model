import ast
import sys
import numpy as np

from pymgrit.heat.heat_1d import Heat1D
from pymgrit.core.grid_transfer_copy import GridTransferCopy

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


def run(nt, coarsening):
    rank = MPI.COMM_WORLD.Get_rank()
    problem = [Heat1D(x_start=0, x_end=1, nx=4097, a=1, init_cond=init_cond, rhs=rhs, t_start=0, t_stop=2, nt=nt)]
    transfer = []
    for i in range(len(coarsening) - 1):
        problem.append(
            Heat1D(t_interval=problem[-1].t[::coarsening[i]],
                   x_start=0, x_end=1, nx=4097, a=1, init_cond=init_cond, rhs=rhs))
        transfer.append(GridTransferCopy())
    step = [[] for i in range(len(problem))]
    res = [[] for i in range(len(problem) - 1)]
    pro = [[] for i in range(len(problem) - 1)]
    import time
    times = 5
    MPI.COMM_WORLD.barrier()
    for i in range(len(problem)):
        for j in range(times):
            start = time.time()
            problem[i].step(u_start=problem[i].vector_t_start, t_start=problem[i].t[0], t_stop=problem[i].t[1])
            runtime = time.time() - start
            step[i].append(runtime)
        if i != len(problem) - 1:
            for j in range(times):
                start = time.time()
                transfer[i].restriction(problem[i].vector_t_start)
                runtime = time.time() - start
                res[i].append(runtime)
            for j in range(times):
                start = time.time()
                transfer[i].interpolation(problem[i + 1].vector_t_start)
                runtime = time.time() - start
                pro[i].append(runtime)

    if rank == 0:
        for i in range(len(problem)):
            print('Measured costs: Step on level', i, '| max:', max(step[i]), '| min:', min(step[i]), '| avg:',
                  np.mean(step[i]))
            if len(problem[i].save) > 0:
                for key, value in problem[i].save.items():
                    print(f'{key}: len {len(value)} '
                          f'level {i} avg {np.average(value)} '
                          f'min {np.min(value)} '
                          f'max {np.max(value)}')

        for i in range(len(problem) - 1):
            print('Measured costs: Res on level', i, '| max:', max(res[i]), '| min:', min(res[i]), '| avg:',
              np.mean(res[i]))

        for i in range(len(problem) - 1):
            print('Measured costs: Pro on level', i, '| max:', max(pro[i]), '| min:', min(pro[i]), '| avg:',
              np.mean(pro[i]))


MPI.COMM_WORLD.barrier()


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
        iters = 3
        conv_crit = 1
        size = [100, 50, 10]
        sleep_step = [0.1, 0.1, 0.1]
        sleep_pro = [0.001, 0.001, 0.001]
        sleep_res = [0.001, 0.001, 0.001]
        print_timings = False
        cycle_type = 'V'

    run(nt=nt,
        coarsening=coarsening)


if __name__ == '__main__':
    main()
