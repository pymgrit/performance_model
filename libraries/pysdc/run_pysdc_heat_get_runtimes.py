import ast
import sys
import time
import numpy as np

from mpi4py import MPI

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from HeatEquation_ND_FD import heatNd_forced
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.core.Step import step


def set_parameters_ml(intervals, num_nodes, iters, predict_type, nsweeps):
    # set time parameters
    t0 = 0.0
    Tend = 2

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = -1
    level_params['dt'] = Tend / intervals
    level_params['nsweeps'] = nsweeps

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['QI'] = 'LU'
    sweeper_params['num_nodes'] = num_nodes

    # initialize problem parameters
    ndim = 1
    problem_params = dict()
    problem_params['ndim'] = ndim
    problem_params['order'] = 2
    problem_params['nu'] = 1
    problem_params['freq'] = tuple(2 for _ in range(ndim))
    problem_params['bc'] = 'dirichlet-zero'
    if False:
        problem_params['nvars'] = [tuple(32 for _ in range(ndim)), tuple(32 for _ in range(ndim))]  # number of dofs
    else:
        problem_params['nvars'] = tuple(4097 for _ in range(ndim))  # number of dofs
    problem_params['direct_solver'] = True  # do GMRES instead of LU

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = iters  # Max iterations
    step_params['errtol'] = 1E-07

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 6

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    # controller_params['all_to_done'] = True  # can ask the controller to keep iterating all steps until the end
    controller_params['predict_type'] = predict_type  # 'fine_only', 'libpfasst_style', 'pfasst_burnin'

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = heatNd_forced  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

    return description, controller_params, t0, Tend


if __name__ == "__main__":

    stdin = sys.argv
    if len(stdin) == 12:
        intervals = int(stdin[1])
        num_nodes = ast.literal_eval(stdin[2])
        predict_type = str(stdin[3])
        if predict_type == 'None':
            predict_type = None
        nvars = ast.literal_eval(stdin[4])
        iters = int(stdin[5])
        nsweeps = ast.literal_eval(stdin[6])
        sleep_step = ast.literal_eval(stdin[7])
        sleep_pro = ast.literal_eval(stdin[8])
        sleep_res = ast.literal_eval(stdin[9])
        sleep_f = ast.literal_eval(stdin[10])
        print_timings = int(stdin[11])
    else:
        intervals = 4
        num_nodes = [3, 3, 3]
        predict_type = None
        iters = 3
        nsweeps = [1, 1, 1]

    # set MPI communicator
    comm = MPI.COMM_WORLD

    # get parameters from Part A
    description, controller_params, t0, Tend = set_parameters_ml(
        intervals=intervals,
        num_nodes=num_nodes,
        predict_type=predict_type,
        iters=iters,
        nsweeps=nsweeps,
    )

    num_level = len(num_nodes)
    probs = []
    transfer = []
    for i in range(num_level):
        probs.append(heatNd_forced(problem_params=description['problem_params']))
    for i in range(num_level - 1):
        transfer.append(mesh_to_mesh(probs[i], probs[i + 1], params=description['space_transfer_params']))

    S = step(description=description)

    sweeper = [[] for i in range(len(S.levels))]
    res = [[] for i in range(len(S.levels) - 1)]
    pro = [[] for i in range(len(S.levels) - 1)]
    res_single = [[] for i in range(len(S.levels) - 1)]
    pro_single = [[] for i in range(len(S.levels) - 1)]
    f_eval_single = [[] for i in range(len(S.levels))]

    MPI.COMM_WORLD.barrier()

    for i in range(len(S.levels)):
        L = S.levels[i]
        P = L.prob
        L.status.time = 0.1
        L.u[0] = P.u_exact(L.time)
        L.sweep.predict()
        S.status.iter = 0
        times = 5
        for j in range(times):
            start = time.time()
            L.sweep.update_nodes()
            runtime = time.time() - start
            sweeper[i].append(runtime)

            L.sweep.compute_residual()  # todo

            start = time.time()
            L.prob.eval_f(L.u[0], 0)
            runtime = time.time() - start
            f_eval_single[i].append(runtime)

    for i in range(len(S.levels)):
        if i != len(S.levels) - 1:
            for j in range(times):
                start = time.time()
                S.transfer(source=S.levels[i], target=S.levels[i + 1])
                runtime = time.time() - start
                res[i].append(runtime)

                start = time.time()
                S.transfer(source=S.levels[i + 1], target=S.levels[i])
                runtime = time.time() - start
                pro[i].append(runtime)

                start = time.time()
                transfer[i].restrict(S.levels[i].u[0])
                runtime = time.time() - start
                res_single[i].append(runtime)

                start = time.time()
                transfer[i].prolong(S.levels[i + 1].u[0])
                runtime = time.time() - start
                pro_single[i].append(runtime)
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(sweeper)
        print(f_eval_single)
        print(res)
        print(pro)
        print(pro_single)
        print(res_single)
        for i in range(len(sweeper)):
            print('Measured costs: Sweeper on level', i, '| max:', max(sweeper[i]), '| min:', min(sweeper[i]), '| avg:',
                  np.mean(sweeper[i]))

        for i in range(len(f_eval_single)):
            print('Measured costs: EvalFSingle on level', i, '| max:', max(f_eval_single[i]), '| min:',
                  min(f_eval_single[i]), '| avg:',
                  np.mean(f_eval_single[i]))

        for i in range(len(res)):
            print('Measured costs: Res on level', i, '| max:', max(res[i]), '| min:', min(res[i]), '| avg:',
                  np.mean(res[i]))

        for i in range(len(pro)):
            print('Measured costs: Pro on level', i, '| max:', max(pro[i]), '| min:', min(pro[i]), '| avg:',
                  np.mean(pro[i]))

        for i in range(len(pro_single)):
            print('Measured costs: InterpolateSingle on level', i, '| max:', max(pro_single[i]), '| min:',
                  min(pro_single[i]), '| avg:',
                  np.mean(pro_single[i]))

        for i in range(len(res_single)):
            print('Measured costs: RestrictSingle on level', i, '| max:', max(res_single[i]), '| min:',
                  min(res_single[i]), '| avg:',
                  np.mean(res_single[i]))
    MPI.COMM_WORLD.barrier()
