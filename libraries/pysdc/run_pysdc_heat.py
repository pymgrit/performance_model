import ast
import sys
import numpy

from mpi4py import MPI

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from HeatEquation_ND_FD import heatNd_forced
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh


def set_parameters_ml(intervals, num_nodes, iters,predict_type, nsweeps):
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
    ndim=1
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
        intervals = 17
        num_nodes = [5, 3,2]
        predict_type = None
        nvars = [100, 50, 50]
        iters = 10
        nsweeps = [1, 1,1]
        sleep_step = [.01, .001, 0.001]
        sleep_pro = [.0, .0, 0.0]
        sleep_res = [.0, .0, .0]
        sleep_f = [.0, .0, .0]
        print_timings = True

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

    # instantiate controllers
    controller = controller_MPI(controller_params=controller_params, description=description, comm=comm)
    # get initial values on finest level
    P = controller.S.levels[0].prob
    uinit = P.u_exact(t0)

    # call main functions to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    runtime = sort_stats(filter_stats(stats, type='timing_run'), sortby='time')[0][1]
    iter_counts = sort_stats(filter_stats(stats, type='niter'), sortby='time')
    print(f'Iterations rank {MPI.COMM_WORLD.Get_rank()}: {[item[1] for item in iter_counts]}')

    if MPI.COMM_WORLD.Get_rank() == 0:
        print('PySDC runtime:', runtime)

