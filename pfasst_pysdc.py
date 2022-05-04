from pfasst import Pfasst


class PfasstPySDC(Pfasst):
    def __init__(self, *args, **kwargs):
        """
        Constructor
        """
        super().__init__(*args, **kwargs)

    def pfasst(self, k):
        """
        k'th PFASST iteration
        :param k: iteration
        """
        for level in range(0, self.L - 1):
            for i in range(1, self.nt):
                if level > 0:
                    for j in range(self.nsweeps[level]):
                        if i > 1 and j>0:
                            self.copy_and_f_eval_single(op_in=['u', self.cr_dict(iteration=k - 1, level=level, time_point=i - 1, colloc_node='last')],
                                                    op_out_1=['u', self.cr_dict(iteration=k - 1, level=level, time_point=i, colloc_node='first')],
                                                    op_out_2=['f', self.cr_dict(iteration=k - 1, level=level, time_point=i, colloc_node='first')],
                                                    level=level,
                                                    i=i)
                        self.sdc_sweep(op_in_1=['u', self.cr_dict(iteration=k - 1, level=level, time_point=i, colloc_node='all')],
                                       op_in_2=['f', self.cr_dict(iteration=k - 1, level=level, time_point=i, colloc_node='all')],
                                       op_in_3=None if level == 0 else ['tau',
                                                                        self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')],
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
                                  op_out=['u', self.cr_dict(iteration=k - 1, level=level + 1, time_point=i, colloc_node='all')],
                                  level=level,
                                  i=i)
                self.f_eval_all(op_in=['u', self.cr_dict(iteration=k - 1, level=level + 1, time_point=i, colloc_node='all')],
                                op_out=['f', self.cr_dict(iteration=k - 1, level=level + 1, time_point=i, colloc_node='all')],
                                level=level,
                                i=i,
                                cost=self.cost_f_eval_all[level + 1])
                self.fas(op_in_1=['f', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')],
                         op_in_2=['f', self.cr_dict(iteration=k - 1, level=level + 1, time_point=i, colloc_node='all')],
                         op_in_3=None if level == 0 else ['tau', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')],
                         op_out=['tau', self.cr_dict(iteration=k, level=level + 1, time_point=i, colloc_node='all')],
                         level=level,
                         i=i)
        if self.placing_conv_crit == 1:
            self.update_cc(k=k)
            self.convergence_criterion(poins_with_dependencies=self.cc)

        # Coarsest level
        for i in range(1, self.nt):
            if i > 1:
                self.copy_and_f_eval_single(op_in=['u', self.cr_dict(iteration=k, level=self.L - 1, time_point=i - 1, colloc_node='last')],
                                            op_out_1=['u', self.cr_dict(iteration=k - 1, level=self.L - 1, time_point=i, colloc_node='first')],
                                            op_out_2=['f', self.cr_dict(iteration=k - 1, level=self.L - 1, time_point=i, colloc_node='first')],
                                            level=self.L - 1,
                                            i=i)
            for _ in range(self.nsweeps[self.L - 1]):
                self.sdc_sweep(op_in_1=['u', self.cr_dict(iteration=k - 1, level=self.L - 1, time_point=i, colloc_node='all')],
                               op_in_2=['f', self.cr_dict(iteration=k - 1, level=self.L - 1, time_point=i, colloc_node='all')],
                               op_in_3=['tau', self.cr_dict(iteration=k, level=self.L - 1, time_point=i, colloc_node='all')],
                               op_out_1=['u', self.cr_dict(iteration=k, level=self.L - 1, time_point=i, colloc_node='all')],
                               op_out_2=['v', self.cr_dict(iteration=k, level=self.L - 1, time_point=i, colloc_node='all')],
                               level=self.L - 1,
                               i=i)

        for level in range(self.L - 2, -1, -1):
            for i in range(1, self.nt):
                self.interpolate_and_correct_all(op_in_1=['u', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')],
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

            for i in range(1, self.nt):
                for j in range(self.nsweeps[level]):
                    if i > 1:
                        if j==0:
                            var_in = 'v'
                        else:
                            var_in = 'v_'+str(j-1)
                        self.copy_and_f_eval_single(op_in=[var_in, self.cr_dict(iteration=k, level=level, time_point=i - 1, colloc_node='last')],
                                                    op_out_1=[var_in, self.cr_dict(iteration=k - 1, level=level, time_point=i, colloc_node='first')],
                                                    op_out_2=['f', self.cr_dict(iteration=k - 1, level=level, time_point=i, colloc_node='first')],
                                                    level=level,
                                                    i=i)
                    if j == 0:
                        var_in = 'v'
                    else:
                        var_in = 'v_' + str(j - 1)
                    self.sdc_sweep(op_in_1=[var_in, self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')],
                                       op_in_2=['f', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')],
                                       op_in_3=None if level == 0 else ['tau',
                                                                        self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')],
                                       op_out_1=['u', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')],
                                       op_out_2=['v_' + str(j), self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')],
                                       level=level,
                                       i=i)
        for i in range(1, self.nt):
            if i > 1:
                self.copy_and_f_eval_single(op_in=['u', self.cr_dict(iteration=k, level=0, time_point=i - 1, colloc_node='last')],
                                            op_out_1=['u', self.cr_dict(iteration=k, level=0, time_point=i, colloc_node='first')],
                                            op_out_2=['f', self.cr_dict(iteration=k, level=0, time_point=i, colloc_node='first')],
                                            level=0,
                                            i=i)


    def predict(self):
        """
        Predictor
        """
        for i in range(1, self.nt):
            self.add_node(name="c0|",
                          predecessors=['u_0'],
                          set_values=self.create_node_name(var_name='u',
                                                           var_dict=self.cr_dict(iteration=0, level=0, time_point=i, colloc_node='first')),
                          cost=self.cost_copy[0],
                          point=i,
                          description='Set first point of every time step to initial value')
            self.add_node(name="C0|",
                          predecessors=['0'],
                          set_values=self.create_node_name(var_name='u',
                                                           var_dict=self.cr_dict(iteration=0, level=0, time_point=i, colloc_node='last')),
                          cost=self.cost_copy[0],
                          point=i,
                          description='Set last point of every time step to 0')
            self.add_node(name="C0|",
                          predecessors=['0'],
                          set_values=self.create_node_name(var_name='f',
                                                           var_dict=self.cr_dict(iteration=0, level=0, time_point=i, colloc_node='all')),
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
                if i > 1:
                    self.copy_and_f_eval_single(op_in=['u', self.cr_dict(iteration=0, level=0, time_point=i, colloc_node='last')],
                                                op_out_1=['u', self.cr_dict(iteration=0, level=0, time_point=i, colloc_node='first')],
                                                op_out_2=['f', self.cr_dict(iteration=0, level=0, time_point=i, colloc_node='first')],
                                                level=level,
                                                i=i)
        elif self.predict_type == 'libpfasst_style':
            raise Exception("not implemented")
        elif self.predict_type == 'pfasst_burnin':
            raise Exception('not implemented')
        elif self.predict_type is None:
            raise Exception('not implemented')
