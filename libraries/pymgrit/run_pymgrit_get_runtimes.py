import ast
import sys
import numpy as np

from pymgrit_app_time import PymgritAppTime
from pymgrit_transfer_time import PymgritTransferTime

from mpi4py import MPI


def run(nt, coarsening, nested_iter, cf_iter, sleep_step, iters, conv_crit, size, res_sleep, pro_sleep, print_timings,
        cycle_type):
    rank = MPI.COMM_WORLD.Get_rank()
    problem = [PymgritAppTime(t_start=0,
                              t_stop=nt - 1,
                              nt=nt,
                              runtime=sleep_step[0],
                              size=size[0],
                              rank=rank,
                              level=0,
                              print_timings=0)]
    transfer = []
    for i in range(len(coarsening) - 1):
        problem.append(
            PymgritAppTime(t_interval=problem[-1].t[::coarsening[i]],
                           runtime=sleep_step[i + 1],
                           size=size[i + 1],
                           rank=rank,
                           level=i + 1,
                           print_timings=0))
        transfer.append(PymgritTransferTime(res_sleep=res_sleep[i],
                                            pro_sleep=pro_sleep[i],
                                            level=i,
                                            print_timings=0,
                                            size=size))
    step = [[] for _ in range(len(problem))]
    res = [[] for _ in range(len(problem) - 1)]
    pro = [[] for _ in range(len(problem) - 1)]
    import time
    times = 5
    MPI.COMM_WORLD.barrier()
    if rank == 0:
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
        coarsening=coarsening,
        cf_iter=cf_iter,
        size=size,
        nested_iter=nested_iter,
        iters=iters,
        conv_crit=conv_crit,
        sleep_step=sleep_step,
        pro_sleep=sleep_pro,
        res_sleep=sleep_res,
        print_timings=print_timings,
        cycle_type=cycle_type)


if __name__ == '__main__':
    main()
