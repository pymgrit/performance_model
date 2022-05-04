from parareal import Parareal
from pfasst import Pfasst
from pfasst_pysdc import PfasstPySDC
from pfasst_libpfasst import PfasstLibpfasst
from mgrit import Mgrit

if __name__ == '__main__':

    # Parareal
    parareal_model = Parareal(cost_fine=1, cost_coarse=1, nt=9,iters=3,conv_crit=1)
    parareal_model.compute()
    parareal_model.plot_dag()
    parareal_model.longest_path()
    parareal_model.compute_standard_schedule(procs=8, plot=True)

    #MGRIT
    mgrit_model = Mgrit(coarsening=[3, 1], cost_step=[2, 2], cf_iter=[1, 1], cycle_type='V', conv_crit=1,
                        placing_conv_crit=0, nt=10, iters=2, nested_iterations=True, node_cost_only=False)
    mgrit_model.compute()
    mgrit_model.plot_dag()
    mgrit_model.longest_path()

    #PFASST
    pfasst_model = Pfasst(cost_sweeper=[2, 1], cost_fas=[2, 1], cost_pro_single=[.2, .1], cost_res_single=[.2, .1],
                          cost_f_eval_single=[.2, .1],
                          cost_pro_all=[2, 1], cost_res_all=[2, 1], cost_f_eval_all=[2, 1], nsweeps=[1, 1], conv_crit=1,
                          placing_conv_crit=0, nt=10, iters=2, level=2, pfasst_style='classic',
                          predict_type='fine_only')
    pfasst_model.compute()
    pfasst_model.plot_dag()
    pfasst_model.longest_path()
    pfasst_model.compute_standard_schedule(procs=9, plot=True)

    # Libpfasst PFASST implementation
    pfasst_model = PfasstLibpfasst(cost_sweeper=[2, 1], cost_fas=[2, 1], cost_pro_single=[.2, .1],
                                   cost_res_single=[.2, .1], cost_f_eval_single=[.2, .1],
                                   cost_pro_all=[2, 1], cost_res_all=[2, 1], cost_f_eval_all=[2, 1], nsweeps=[1, 1],
                                   conv_crit=1, placing_conv_crit=0, nt=10, iters=2, level=2, pfasst_style='classic',
                                   predict_type='libpfasst_true')
    pfasst_model.compute()
    pfasst_model.plot_dag()
    pfasst_model.longest_path()
    pfasst_model.compute_standard_schedule(procs=9, plot=True)

    #PySDC PFASST implementation
    pfasst_model = PfasstPySDC(cost_sweeper=[2, 1], cost_fas=[2, 1], cost_pro_single=[.2, .1], cost_res_single=[.2, .1],
                               cost_f_eval_single=[.2, .1],
                               cost_pro_all=[2, 1], cost_res_all=[2, 1], cost_f_eval_all=[2, 1], nsweeps=[1, 1],
                               conv_crit=1, placing_conv_crit=0, nt=10, iters=2, level=2, pfasst_style='multigrid',
                               predict_type='fine_only')
    pfasst_model.compute()
    pfasst_model.plot_dag()
    pfasst_model.longest_path()
    pfasst_model.compute_standard_schedule(procs=9, plot=True)
