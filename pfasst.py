import numpy as np
from pint_task_graph import PintGraph


class Pfasst(PintGraph):
    def __init__(self, level: object, cost_sweeper: object, cost_res_all: object, cost_res_single: object,
                 cost_pro_all: object, placing_conv_crit: object,
                 pfasst_style: object, cost_pro_single: object, cost_f_eval_all: object, cost_fas: object,
                 cost_f_eval_single: object, nsweeps: object,
                 predict_type: object = None, level_0_sweep_start: object = True, level_0_sweep_end: object = True,
                 *args: object, **kwargs: object) -> object:
        """
        Constructor

        :param level: Level
        :param cost_sweeper: Sweeper costs per level
        :param cost_res_all: Restriction all collocation points costs
        :param cost_res_single: Restriction single time point
        :param cost_pro_all: Prolongation all collocation points costs
        :param placing_conv_crit: Different placing of the conv criterion
        :param pfasst_style: Pfasst view (multigrid or classic)
        :param cost_pro_single: Prolongation single time point
        :param cost_f_eval_all: Evaluation all collocation points costs
        :param cost_fas: FAS costs
        :param cost_f_eval_single: Evaluation single time point
        :param nsweeps: Number of sweeps per level
        :param predict_type: Prediction variant
        :param level_0_sweep_start: Sweep at level 0 at the beginning
        :param level_0_sweep_end: Sweep at level 0 at the end
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

        # Save to local parameters
        self.L = level
        self.cost_sweeper = cost_sweeper
        self.cost_restriction_all = cost_res_all
        self.cost_restriction_single = cost_res_single
        self.cost_interpolation_all = cost_pro_all
        self.cost_interpolation_single = cost_pro_single
        self.cost_f_eval_all = cost_f_eval_all
        self.cost_f_eval_single = cost_f_eval_single
        self.cost_fas = cost_fas
        self.placing_conv_crit = placing_conv_crit
        self.pfasst_style = pfasst_style
        self.predict_type = predict_type
        self.nsweeps = nsweeps
        self.sweep_level_0_start_iteration = level_0_sweep_start
        self.sweep_level_0_end_iteration = level_0_sweep_end
        self.cost_copy = np.zeros(self.L)
        if predict_type is not None and predict_type not in ['null', 'libpfasst_style', 'fine_only', 'pfasst_burnin',
                                                             'libpfasst_true']:
            raise Exception('unknown predict type')
        else:
            if self.predict_type == 'null':
                self.predict_type = None
        if self.pfasst_style not in ['multigrid', 'classic']:
            raise Exception('unknown pfasst_style')

        self.cc = {}

    def update_cc(self, k: int) -> None:
        """
        Convergence criterion

        :param k: iteration
        """
        self.cc = {}
        if self.placing_conv_crit == 0:
            for i in range(1, self.nt):
                cc = self.create_node_name(var_name='f', var_dict=self.cr_dict(level=0, time_point=i, iteration=k,
                                                                                colloc_node='all'))
                self.cc[i] = cc
        elif self.placing_conv_crit == 1:
            for i in range(1, self.nt):
                cc = self.create_node_name(var_name='f', var_dict=self.cr_dict(level=0, time_point=i, iteration=k,
                                                                                colloc_node='all'))
                self.cc[i] = cc
        else:
            raise Exception('Unknown placing')

    def compute(self):
        """
        Computes the graph
        """
        self.predict()
        if self.placing_conv_crit == 0:
            self.update_cc(k=0)
            self.convergence_criterion(poins_with_dependencies=self.cc)
        for k in range(1, self.iterations + 1):
            self.sychronize_nodes_plot()
            self.pfasst(k=k)
            if self.placing_conv_crit == 0:
                self.update_cc(k=k)
                self.convergence_criterion(poins_with_dependencies=self.cc)

    def pfasst(self, k: int) -> None:
        """
        k'th PFASST iteration
        :param k: iteration
        """
        for level in range(0, self.L - 1):
            for i in range(1, self.nt):
                if self.pfasst_style == 'multigrid':
                    if i > 1:
                        self.copy_and_f_eval_single(op_in=['u',
                                                           self.cr_dict(iteration=k - 1, level=level, time_point=i - 1,
                                                                        colloc_node='last')],
                                                    op_out_1=['u',
                                                              self.cr_dict(iteration=k - 1, level=level, time_point=i,
                                                                           colloc_node='first')],
                                                    op_out_2=['f',
                                                              self.cr_dict(iteration=k - 1, level=level, time_point=i,
                                                                           colloc_node='first')],
                                                    level=level,
                                                    i=i)
                else:
                    if k == 1 or level > 0:
                        self.f_eval_single(
                            op_in=['u', self.cr_dict(iteration=k - 1, level=level, time_point=i, colloc_node='first')],
                            op_out=['f',
                                    self.cr_dict(iteration=k - 1, level=level, time_point=i, colloc_node='first')],
                            level=0,
                            i=i)
                if self.sweep_level_0_start_iteration or level > 0:
                    for _ in range(self.nsweeps[level]):
                        self.sdc_sweep(
                            op_in_1=['u', self.cr_dict(iteration=k - 1, level=level, time_point=i, colloc_node='all')],
                            op_in_2=['f', self.cr_dict(iteration=k - 1, level=level, time_point=i, colloc_node='all')],
                            op_in_3=None if level == 0 else ['tau',
                                                             self.cr_dict(iteration=k, level=level, time_point=i,
                                                                          colloc_node='all')],
                            op_out_1=['u', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')],
                            op_out_2=['f', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')],
                            level=level,
                            i=i)
                else:
                    self.copy(op_in=['u', self.cr_dict(iteration=k - 1, level=level, time_point=i, colloc_node='all')],
                              op_out=['u', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')],
                              level=level,
                              i=i)
                    self.copy(op_in=['f', self.cr_dict(iteration=k - 1, level=level, time_point=i, colloc_node='all')],
                              op_out=['f', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')],
                              level=level,
                              i=i)
                self.restrict_all(op_in=['u', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')],
                                  op_out=['u', self.cr_dict(iteration=k - 1, level=level + 1, time_point=i,
                                                            colloc_node='all')],
                                  level=level,
                                  i=i)
                self.f_eval_all(
                    op_in=['u', self.cr_dict(iteration=k - 1, level=level + 1, time_point=i, colloc_node='all')],
                    op_out=['f', self.cr_dict(iteration=k - 1, level=level + 1, time_point=i, colloc_node='all')],
                    level=level,
                    i=i,
                    cost=self.cost_f_eval_all[level + 1])
                self.fas(op_in_1=['f', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')],
                         op_in_2=['f',
                                  self.cr_dict(iteration=k - 1, level=level + 1, time_point=i, colloc_node='all')],
                         op_in_3=None if level == 0 else ['tau', self.cr_dict(iteration=k, level=level, time_point=i,
                                                                              colloc_node='all')],
                         op_out=['tau', self.cr_dict(iteration=k, level=level + 1, time_point=i, colloc_node='all')],
                         level=level,
                         i=i)
        if self.placing_conv_crit == 1:
            self.update_cc(iteration=k)
            self.convergence_criterion(poins_with_dependencies=self.cc)

        # Coarsest level
        for i in range(1, self.nt):
            if i > 1:
                self.copy_and_f_eval_single(
                    op_in=['u', self.cr_dict(iteration=k, level=self.L - 1, time_point=i - 1, colloc_node='last')],
                    op_out_1=['u', self.cr_dict(iteration=k - 1, level=self.L - 1, time_point=i, colloc_node='first')],
                    op_out_2=['f', self.cr_dict(iteration=k - 1, level=self.L - 1, time_point=i, colloc_node='first')],
                    level=self.L - 1,
                    i=i)
            for _ in range(self.nsweeps[self.L - 1]):
                self.sdc_sweep(
                    op_in_1=['u', self.cr_dict(iteration=k - 1, level=self.L - 1, time_point=i, colloc_node='all')],
                    op_in_2=['f', self.cr_dict(iteration=k - 1, level=self.L - 1, time_point=i, colloc_node='all')],
                    op_in_3=['tau', self.cr_dict(iteration=k, level=self.L - 1, time_point=i, colloc_node='all')],
                    op_out_1=['u', self.cr_dict(iteration=k, level=self.L - 1, time_point=i, colloc_node='all')],
                    op_out_2=['v', self.cr_dict(iteration=k, level=self.L - 1, time_point=i, colloc_node='all')],
                    level=self.L - 1,
                    i=i)

        for level in range(self.L - 2, -1, -1):
            for i in range(1, self.nt):
                self.interpolate_and_correct_all(
                    op_in_1=['u', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')],
                    op_in_2=['u', self.cr_dict(iteration=k, level=level + 1, time_point=i, colloc_node='all')],
                    op_in_3=['u', self.cr_dict(iteration=k - 1, level=level + 1, time_point=i, colloc_node='all')],
                    op_out=['v', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')],
                    level=level,
                    i=i, )
                self.f_eval_all(op_in=['v', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')],
                                op_out=['f', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')],
                                level=level,
                                i=i,
                                cost=self.cost_f_eval_all[level] if self.pfasst_style == 'classic' else
                                self.cost_f_eval_all[level] - self.cost_f_eval_single[level])
                if i > 1:
                    if self.pfasst_style == 'multigrid':
                        self.copy_and_f_eval_single(
                            op_in=['v', self.cr_dict(iteration=k, level=level, time_point=i - 1, colloc_node='last')],
                            op_out_1=['v',
                                      self.cr_dict(iteration=k - 1, level=level, time_point=i, colloc_node='first')],
                            op_out_2=['f',
                                      self.cr_dict(iteration=k - 1, level=level, time_point=i, colloc_node='first')],
                            level=level,
                            i=i,
                            v=True)
                    else:
                        self.copy_and_error_correction(
                            op_in_1=['v',
                                     self.cr_dict(iteration=k, level=level, time_point=i - 1, colloc_node='last')],
                            op_in_2=['v',
                                     self.cr_dict(iteration=k, level=level + 1, time_point=i, colloc_node='first')],
                            op_out_1=['v', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='first')],
                            op_out_2=['f', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='first')],
                            level=level,
                            i=i)
            for i in range(1, self.nt):
                if level > 0 or self.sweep_level_0_end_iteration:
                    for _ in range(self.nsweeps[level]):
                        self.sdc_sweep(
                            op_in_1=['v', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')],
                            op_in_2=['f', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')],
                            op_in_3=None if level == 0 else ['tau',
                                                             self.cr_dict(iteration=k, level=level, time_point=i,
                                                                          colloc_node='all')],
                            op_out_1=['u', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')],
                            op_out_2=['v', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')],
                            level=level,
                            i=i)
                else:
                    self.copy(op_in=['v', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')],
                              op_out=['u', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')],
                              level=level,
                              i=i)

    def sdc_sweep(self, op_in_1: list, op_in_2: list, op_in_3: list, op_out_1: list, op_out_2: list, level: int,
                  i: int) -> None:
        """
        Models a sdc sweep

        :param op_in_1: Data dependency
        :param op_in_2: Data dependency
        :param op_in_3: Data dependency
        :param op_out_1: New data
        :param op_out_2: New data
        :param level: Level
        :param i: time point
        """
        pred = self.create_node_name(var_name=op_in_1[0], var_dict=op_in_1[1])
        pred += self.create_node_name(var_name=op_in_2[0], var_dict=op_in_2[1])
        if op_in_3 is not None:
            pred += self.create_node_name(var_name=op_in_3[0], var_dict=op_in_3[1])

        set_val = self.create_node_name(var_name=op_out_1[0], var_dict=op_out_1[1])
        set_val += self.create_node_name(var_name=op_out_2[0], var_dict=op_out_2[1])

        self.add_node(name="S|" + str(level),
                      predecessors=pred,
                      set_values=set_val,
                      cost=self.cost_sweeper[level],
                      point=i,
                      description='pfasst_sweeper' + str(level))

    def restrict_all(self, op_in: list, op_out: list, level: int, i: int) -> None:
        """
        Models restriction of all collocation nodes

        :param op_in: Data dependency
        :param op_out: New data
        :param level: Level
        :param i: iteration
        """
        self.add_node(name="R|" + str(level),
                      predecessors=self.create_node_name(var_name=op_in[0], var_dict=op_in[1]),
                      set_values=self.create_node_name(var_name=op_out[0], var_dict=op_out[1]),
                      cost=self.cost_restriction_all[level],
                      point=i,
                      description='pfasst_res_all' + str(level))

    def restrict_single(self, op_in: list, op_out: list, level: int, i: int) -> None:
        """
        Models restriction of one point

        :param op_in: Data dependency
        :param op_out: New data
        :param level: Level
        :param i: iteration
        """
        self.add_node(name="r|" + str(level),
                      predecessors=self.create_node_name(var_name=op_in[0], var_dict=op_in[1]),
                      set_values=self.create_node_name(var_name=op_out[0], var_dict=op_out[1]),
                      cost=self.cost_restriction_single[level],
                      point=i,
                      description='pfasst_res_single' + str(level))

    def f_eval_all(self, op_in: list, op_out: list, level: int, i: int, cost: float) -> None:
        """
        Models evaluation of all collocation nodes

        :param op_in: Data dependency
        :param op_out: New data
        :param level: Level
        :param i: iteration
        :param cost: Operation costs
        """
        self.add_node(name="FE|" + str(level),
                      predecessors=self.create_node_name(var_name=op_in[0], var_dict=op_in[1]),
                      set_values=self.create_node_name(var_name=op_out[0], var_dict=op_out[1]),
                      cost=cost,
                      point=i,
                      description='pfasst_f_eval_all' + str(level))

    def f_eval_single(self, op_in: list, op_out: list, level: int, i: int) -> None:
        """
        Models evaluation of one collocation nodes

        :param op_in:
        :param op_out:
        :param level:
        :param i:
        """
        self.add_node(name="fe|" + str(level),
                      predecessors=self.create_node_name(var_name=op_in[0], var_dict=op_in[1]),
                      set_values=self.create_node_name(var_name=op_out[0], var_dict=op_out[1]),
                      cost=self.cost_f_eval_single[level],
                      point=i,
                      description='pfasst_f_eval_single' + str(level))

    def fas(self, op_in_1: list, op_in_2: list, op_in_3: list, op_out: list, level: int, i: int) -> None:
        """
        Models FAS

        :param op_in_1: Data dependency
        :param op_in_2: Data dependency
        :param op_in_3: Data dependency
        :param op_out: New data
        :param level: Level
        :param i: iteration
        """
        pred = self.create_node_name(var_name=op_in_1[0], var_dict=op_in_1[1])
        pred += self.create_node_name(var_name=op_in_2[0], var_dict=op_in_2[1])
        if op_in_3 is not None:
            pred += self.create_node_name(var_name=op_in_3[0], var_dict=op_in_3[1])

        self.add_node(name="FAS|" + str(level),
                      predecessors=pred,
                      set_values=self.create_node_name(var_name=op_out[0], var_dict=op_out[1]),
                      cost=self.cost_fas[level],
                      point=i,
                      description='pfasst_fas_' + str(level))

    def copy(self, op_in: list, op_out: list, level: int, i: int) -> None:
        """
        Models a copy

        :param op_in: Data dependency
        :param op_out: New data
        :param level: Level
        :param i: iteration
        """
        self.add_node(name="C|" + str(level),
                      predecessors=self.create_node_name(var_name=op_in[0], var_dict=op_in[1]),
                      set_values=self.create_node_name(var_name=op_out[0], var_dict=op_out[1]),
                      cost=self.cost_copy[level],
                      point=i,
                      description='pfasst_copy_' + str(level))

    def copy_and_f_eval_single(self, op_in: list, op_out_1: list, op_out_2: list, level: int, i: int) -> None:
        """
        Models copy with evaluation

        :param op_in: Data dependency
        :param op_out_1: New data
        :param op_out_2: New data
        :param level: Level
        :param i: iteration
        """
        self.add_node(name="c|" + str(level),
                      predecessors=self.create_node_name(var_name=op_in[0], var_dict=op_in[1]),
                      set_values=self.create_node_name(var_name=op_out_1[0], var_dict=op_out_1[1]),
                      cost=self.cost_copy[level],
                      point=i,
                      description='pfasst_copy_and_f_eval_single' + str(level))
        self.add_node(name="f|" + str(level),
                      predecessors=self.create_node_name(var_name=op_out_1[0], var_dict=op_out_1[1]),
                      set_values=self.create_node_name(var_name=op_out_2[0], var_dict=op_out_2[1]),
                      cost=self.cost_f_eval_single[level],
                      point=i,
                      description='pfasst_commu_and_f_eval_single' + str(level))

    def interpolate_and_correct_all(self, op_in_1: list, op_in_2: list, op_in_3: list, op_out: list, level: int,
                                    i: int) -> None:
        """
        Models interpolation and correction of all collocation nodes

        :param op_in_1: Data dependency
        :param op_in_2: Data dependency
        :param op_in_3: Data dependency
        :param op_out: New data
        :param level: Level
        :param i: iteration
        """
        pred = self.create_node_name(var_name=op_in_1[0], var_dict=op_in_1[1])
        pred += self.create_node_name(var_name=op_in_2[0], var_dict=op_in_2[1])
        pred += self.create_node_name(var_name=op_in_3[0], var_dict=op_in_3[1])

        self.add_node(name="I|" + str(level),
                      predecessors=pred,
                      set_values=self.create_node_name(var_name=op_out[0], var_dict=op_out[1]),
                      cost=self.cost_interpolation_all[level],
                      point=i,
                      description='pfasst_pro_all' + str(level))

    def copy_and_error_correction(self, op_in_1: list, op_in_2: list, op_out_1: list, op_out_2: list, level: int,
                                  i: int) -> None:
        """
        Copy and correction

        :param op_in_1: Data dependency
        :param op_in_2: Data dependency
        :param op_out_1: New data
        :param op_out_2: New data
        :param level: Level
        :param i: iteration
        """
        self.add_node(name="c|" + str(level),
                      predecessors=self.create_node_name(var_name=op_in_1[0], var_dict=op_in_1[1]),
                      set_values=self.create_node_name(var_name=op_out_1[0], var_dict=op_out_1[1]),
                      cost=self.cost_copy[level],
                      point=i,
                      description='pfasst_copy_single' + str(level))
        self.add_node(name="r|" + str(level),
                      predecessors=self.create_node_name(var_name=op_out_1[0], var_dict=op_out_1[1]),
                      set_values=self.create_node_name(var_name='tmp', var_dict=op_out_1[1]),
                      cost=self.cost_restriction_single[level],
                      point=i,
                      description='pfasst_res_single' + str(level))
        pre = self.create_node_name(var_name='tmp', var_dict=op_out_1[1])
        pre += self.create_node_name(var_name=op_out_1[0], var_dict=op_out_1[1])
        pre += self.create_node_name(var_name=op_in_2[0], var_dict=op_in_2[1])
        self.add_node(name="i|" + str(level),
                      predecessors=pre,
                      set_values=self.create_node_name(var_name=op_out_1[0], var_dict=op_out_1[1]),
                      cost=self.cost_interpolation_single[level],
                      point=i,
                      description='pfasst_pro_single' + str(level))
        self.add_node(name="f|" + str(level),
                      predecessors=self.create_node_name(var_name=op_out_1[0], var_dict=op_out_1[1]),
                      set_values=self.create_node_name(var_name=op_out_2[0], var_dict=op_out_2[1]),
                      cost=self.cost_f_eval_single[level],
                      point=i,
                      description='pfasst_f_eval_single' + str(level))

    def predict(self) -> None:
        """
        Predictor
        """
        for i in range(1, self.nt):
            self.add_node(name="c0|",
                          predecessors=['u_0'],
                          set_values=self.create_node_name(var_name='u',
                                                           var_dict=self.cr_dict(iteration=0, level=0, time_point=i,
                                                                                 colloc_node='first')),
                          cost=self.cost_copy[0],
                          point=i,
                          description='Set first point of every time step to initial value')
            self.add_node(name="C0|",
                          predecessors=['0'],
                          set_values=self.create_node_name(var_name='u',
                                                           var_dict=self.cr_dict(iteration=0, level=0, time_point=i,
                                                                                 colloc_node='last')),
                          cost=self.cost_copy[0],
                          point=i,
                          description='Set last point of every time step to 0')
            self.add_node(name="C0|",
                          predecessors=['0'],
                          set_values=self.create_node_name(var_name='f',
                                                           var_dict=self.cr_dict(iteration=0, level=0, time_point=i,
                                                                                 colloc_node='all')),
                          cost=self.cost_copy[0],
                          point=i,
                          description='Set f to 0')
        if self.predict_type == 'fine_only':
            level = 0
            for i in range(1, self.nt):
                self.f_eval_all(op_in=['u', self.cr_dict(iteration=0, level=0, time_point=i, colloc_node='all')],
                                op_out=['f', self.cr_dict(iteration=0, level=0, time_point=i, colloc_node='all')],
                                level=level,
                                i=i,
                                cost=self.cost_f_eval_all[level])
                self.sdc_sweep(op_in_1=['u', self.cr_dict(iteration=0, level=0, time_point=i, colloc_node='all')],
                               op_in_2=['f', self.cr_dict(iteration=0, level=0, time_point=i, colloc_node='all')],
                               op_in_3=None,
                               op_out_1=['u', self.cr_dict(iteration=0, level=0, time_point=i, colloc_node='all')],
                               op_out_2=['f', self.cr_dict(iteration=0, level=0, time_point=i, colloc_node='all')],
                               level=level,
                               i=i)
        elif self.predict_type == 'libpfasst_true':
            for i in range(1, self.nt):
                self.f_eval_single(op_in=['u', self.cr_dict(iteration=0, level=0, time_point=i, colloc_node='first')],
                                   op_out=['f', self.cr_dict(iteration=0, level=0, time_point=i, colloc_node='first')],
                                   level=0,
                                   i=i)
            for level in range(0, self.L - 1):
                for i in range(1, self.nt):
                    self.restrict_single(
                        op_in=['u', self.cr_dict(iteration=0, level=level, time_point=i, colloc_node='first')],
                        op_out=['u', self.cr_dict(iteration=0, level=level + 1, time_point=i, colloc_node='first')],
                        level=level,
                        i=i)
                    self.restrict_all(
                        op_in=['u', self.cr_dict(iteration=0, level=level, time_point=i, colloc_node='all')],
                        op_out=['u', self.cr_dict(iteration=0, level=level + 1, time_point=i, colloc_node='all')],
                        level=level,
                        i=i)
                    self.f_eval_all(
                        op_in=['u', self.cr_dict(iteration=0, level=level + 1, time_point=i, colloc_node='all')],
                        op_out=['f', self.cr_dict(iteration=0, level=level + 1, time_point=i, colloc_node='all')],
                        level=level,
                        i=i,
                        cost=self.cost_f_eval_all[level + 1])
                    self.fas(op_in_1=['f', self.cr_dict(iteration=0, level=level, time_point=i, colloc_node='all')],
                             op_in_2=['f',
                                      self.cr_dict(iteration=0, level=level + 1, time_point=i, colloc_node='all')],
                             op_in_3=None if level == 0 else ['tau',
                                                              self.cr_dict(iteration=0, level=level, time_point=i,
                                                                           colloc_node='all')],
                             op_out=['tau',
                                     self.cr_dict(iteration=0, level=level + 1, time_point=i, colloc_node='all')],
                             level=level,
                             i=i)
            level = self.L - 1

            # burnin
            for j in range(2, self.nt):
                for i in range(self.nt - 1, j - 1, -1):
                    self.copy_and_f_eval_single(
                        op_in=['u', self.cr_dict(iteration=0, level=self.L - 1, time_point=i, colloc_node='last')],
                        op_out_1=['u', self.cr_dict(iteration=0, level=self.L - 1, time_point=i, colloc_node='first')],
                        op_out_2=['f', self.cr_dict(iteration=0, level=self.L - 1, time_point=i, colloc_node='first')],
                        level=self.L - 1,
                        i=i)
                    self.sdc_sweep(
                        op_in_1=['u', self.cr_dict(iteration=0, level=level, time_point=i, colloc_node='all')],
                        op_in_2=['f', self.cr_dict(iteration=0, level=level, time_point=i, colloc_node='all')],
                        op_in_3=['tau', self.cr_dict(iteration=0, level=level, time_point=i, colloc_node='all')],
                        op_out_1=['u', self.cr_dict(iteration=0, level=level, time_point=i, colloc_node='all')],
                        op_out_2=['f', self.cr_dict(iteration=0, level=level, time_point=i, colloc_node='all')],
                        level=level,
                        i=i)
            # sweep
            for i in range(1, self.nt):
                self.copy_and_f_eval_single(
                    op_in=['u', self.cr_dict(iteration=0, level=self.L - 1, time_point=i, colloc_node='last')],
                    op_out_1=['u', self.cr_dict(iteration=0, level=self.L - 1, time_point=i, colloc_node='first')],
                    op_out_2=['f', self.cr_dict(iteration=0, level=self.L - 1, time_point=i, colloc_node='first')],
                    level=self.L - 1,
                    i=i)
                self.sdc_sweep(op_in_1=['u', self.cr_dict(iteration=0, level=level, time_point=i, colloc_node='all')],
                               op_in_2=['f', self.cr_dict(iteration=0, level=level, time_point=i, colloc_node='all')],
                               op_in_3=['tau',
                                        self.cr_dict(iteration=0, level=level, time_point=i, colloc_node='all')],
                               op_out_1=['u', self.cr_dict(iteration=0, level=level, time_point=i, colloc_node='all')],
                               op_out_2=['f', self.cr_dict(iteration=0, level=level, time_point=i, colloc_node='all')],
                               level=level,
                               i=i)

            for level in range(self.L - 2, -1, -1):
                for i in range(1, self.nt):
                    self.interpolate_and_correct_all(
                        op_in_1=['u', self.cr_dict(iteration=0, level=level, time_point=i, colloc_node='all')],
                        op_in_2=['u', self.cr_dict(iteration=0, level=level + 1, time_point=i, colloc_node='all')],
                        op_in_3=['u', self.cr_dict(iteration=0, level=level + 1, time_point=i, colloc_node='all')],
                        op_out=['v', self.cr_dict(iteration=0, level=level, time_point=i, colloc_node='all')],
                        level=level,
                        i=i)
                    self.f_eval_all(
                        op_in=['v', self.cr_dict(iteration=0, level=level, time_point=i, colloc_node='all')],
                        op_out=['f', self.cr_dict(iteration=0, level=level, time_point=i, colloc_node='all')],
                        level=level,
                        i=i,
                        cost=self.cost_f_eval_all[level])
                    if i > 1:
                        self.copy_and_error_correction(
                            op_in_1=['u', self.cr_dict(iteration=0, level=level, time_point=i, colloc_node='last')],
                            op_in_2=['u',
                                     self.cr_dict(iteration=0, level=level + 1, time_point=i, colloc_node='first')],
                            op_out_1=['v', self.cr_dict(iteration=0, level=level, time_point=i, colloc_node='first')],
                            op_out_2=['f', self.cr_dict(iteration=0, level=level, time_point=i, colloc_node='first')],
                            level=level,
                            i=i)
                    self.sdc_sweep(
                        op_in_1=['v', self.cr_dict(iteration=0, level=level, time_point=i, colloc_node='all')],
                        op_in_2=['f', self.cr_dict(iteration=0, level=level, time_point=i, colloc_node='all')],
                        op_in_3=None if level == 0 else ['tau', self.cr_dict(iteration=0, level=level, time_point=i,
                                                                             colloc_node='all')],
                        op_out_1=['u', self.cr_dict(iteration=0, level=level, time_point=i, colloc_node='all')],
                        op_out_2=['f', self.cr_dict(iteration=0, level=level, time_point=i, colloc_node='all')],
                        level=level,
                        i=i)
        elif self.predict_type == 'libpfasst_style':
            raise Exception("not implemented")
        elif self.predict_type == 'pfasst_burnin':
            raise Exception('not implemented')
        elif self.predict_type is None:
            raise Exception('not implemented')
