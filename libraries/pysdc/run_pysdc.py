import ast
import sys

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.sweeper_classes.generic_LU import generic_implicit
from pysdc_transfer_app import PysdcTransfer
from pysdc_app_time import PysdcAppTime

from mpi4py import MPI


def set_parameters_ml(intervals, num_nodes, nvars, sleep_step, iters, sleep_res, sleep_pro, predict_type, nsweeps,
                      sleep_f, print_timings):
    # set time parameters
    t0 = 0.0
    Tend = 1

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
    problem_params = dict()
    problem_params['nvars'] = nvars
    problem_params['sleep_time'] = sleep_step
    problem_params['sleep_f'] = sleep_f
    problem_params['print_timings'] = print_timings
    problem_params['level'] = [i for i in range(len(sleep_step))]

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = iters  # Max iterations
    step_params['errtol'] = -1

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['res_sleep'] = sleep_res
    space_transfer_params['pro_sleep'] = sleep_pro
    space_transfer_params['print_timings'] = print_timings

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    # controller_params['all_to_done'] = True  # can ask the controller to keep iterating all steps until the end
    controller_params['predict_type'] = predict_type  # 'fine_only', 'libpfasst_style', 'pfasst_burnin'

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = PysdcAppTime  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = generic_implicit  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = PysdcTransfer  # pass spatial transfer class
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
        num_nodes = [2, 2,2]
        predict_type = None
        nvars = [100, 50, 50]
        iters = 3
        nsweeps = [1, 1,1]
        sleep_step = [.01, .001, 0.001]
        sleep_pro = [.0, .0, 0.0]
        sleep_res = [.0, .0, .0]
        sleep_f = [.0, .0, .0]
        print_timings = True

    # set MPI communicator
    comm = MPI.COMM_WORLD

    description, controller_params, t0, Tend = set_parameters_ml(
        intervals=intervals,
        num_nodes=num_nodes,
        predict_type=predict_type,
        nvars=nvars,
        sleep_step=sleep_step,
        iters=iters,
        sleep_res=sleep_res,
        sleep_pro=sleep_pro,
        nsweeps=nsweeps,
        sleep_f=sleep_f,
        print_timings=print_timings,
    )

    controller = controller_MPI(controller_params=controller_params, description=description, comm=comm)
    P = controller.S.levels[0].prob
    uinit = P.initial(t0)

    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    runtime = sort_stats(filter_stats(stats, type='timing_run'), sortby='time')[0][1]
    if MPI.COMM_WORLD.Get_rank() == 0:
        print('PySDC runtime:', runtime)

