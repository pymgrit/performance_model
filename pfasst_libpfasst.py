from pfasst import Pfasst


class PfasstLibpfasst(Pfasst):
    def __init__(self, *args, **kwargs):
        """
        Constructor
        """
        super().__init__(*args, **kwargs)
        if self.iterations == 0:
            raise Exception('not implemented')

    def pfasst(self, k):
        """
        k'th PFASST iteration
        :param k: iteration
        """
        for level in range(0, self.L - 1):
            for i in range(1, self.nt):
                if k == 1 or level > 0:
                    self.f_eval_single(
                        op_in=['u', self.cr_dict(iteration=k - 1, level=level, time_point=i, colloc_node='first')],
                        op_out=['f', self.cr_dict(iteration=k - 1, level=level, time_point=i, colloc_node='first')],
                        level=level,
                        i=i)
                if self.sweep_level_0_start_iteration or level > 0:
                    for j in range(self.nsweeps[level]):
                        if j == 0:
                            op_in_1 = ['u', self.cr_dict(iteration=k - 1, level=level, time_point=i, colloc_node='all')]
                            op_in_2 = ['f', self.cr_dict(iteration=k - 1, level=level, time_point=i, colloc_node='all')]
                        else:
                            op_in_1 = ['tmp_fr_u' + str(j - 1),
                                       self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')]
                            op_in_2 = ['tmp_fr_f' + str(j - 1),
                                       self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')]

                        if j == self.nsweeps[level] - 1:
                            op_out_1 = ['u', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')]
                            op_out_2 = ['f', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')]
                        else:
                            op_out_1 = ['tmp_fr_u' + str(j),
                                        self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')]
                            op_out_2 = ['tmp_fr_f' + str(j),
                                        self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')]
                        self.sdc_sweep(op_in_1=op_in_1,
                                       op_in_2=op_in_2,
                                       op_in_3=None if level == 0 else ['tau',
                                                                        self.cr_dict(iteration=k, level=level,
                                                                                     time_point=i, colloc_node='all')],
                                       op_out_1=op_out_1,
                                       op_out_2=op_out_2,
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
                         op_in_2=['f', self.cr_dict(iteration=k - 1, level=level + 1, time_point=i, colloc_node='all')],
                         op_in_3=None if level == 0 else ['tau', self.cr_dict(iteration=k, level=level, time_point=i,
                                                                              colloc_node='all')],
                         op_out=['tau', self.cr_dict(iteration=k, level=level + 1, time_point=i, colloc_node='all')],
                         level=level,
                         i=i)
        if self.placing_conv_crit == 1:
            self.update_cc(k=k)
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
            for j in range(self.nsweeps[self.L - 1]):
                if j == 0:
                    op_in_1 = ['u', self.cr_dict(iteration=k - 1, level=self.L - 1, time_point=i, colloc_node='all')]
                    op_in_2 = ['f', self.cr_dict(iteration=k - 1, level=self.L - 1, time_point=i, colloc_node='all')]
                else:
                    op_in_1 = ['tmp_cl_u' + str(j - 1),
                               self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')]
                    op_in_2 = ['tmp_cl_f' + str(j - 1),
                               self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')]

                if j == self.nsweeps[self.L - 1] - 1:
                    op_out_1 = ['u', self.cr_dict(iteration=k, level=self.L - 1, time_point=i, colloc_node='all')]
                    op_out_2 = ['f', self.cr_dict(iteration=k, level=self.L - 1, time_point=i, colloc_node='all')]
                else:
                    op_out_1 = ['tmp_cl_u' + str(j),
                                self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')]
                    op_out_2 = ['tmp_cl_f' + str(j),
                                self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')]
                self.sdc_sweep(op_in_1=op_in_1,
                               op_in_2=op_in_2,
                               op_in_3=None if level == 0 else ['tau',
                                                                self.cr_dict(iteration=k, level=self.L - 1,
                                                                             time_point=i, colloc_node='all')],
                               op_out_1=op_out_1,
                               op_out_2=op_out_2,
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
                    self.copy_and_error_correction(
                        op_in_1=['u', self.cr_dict(iteration=k, level=level, time_point=i - 1, colloc_node='last')],
                        op_in_2=['u', self.cr_dict(iteration=k, level=level + 1, time_point=i, colloc_node='first')],
                        op_out_1=['v', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='first')],
                        op_out_2=['f', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='first')],
                        level=level,
                        i=i)
            for i in range(1, self.nt):
                if level > 0 or self.sweep_level_0_end_iteration:
                    for j in range(self.nsweeps[level]):
                        if j == 0:
                            op_in_1 = ['v', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')]
                            op_in_2 = ['f', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')]
                        else:
                            op_in_1 = ['tmp_ba_u' + str(j - 1),
                                       self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')]
                            op_in_2 = ['tmp_ba_f' + str(j - 1),
                                       self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')]

                        if j == self.nsweeps[level] - 1:
                            op_out_1 = ['u', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')]
                            op_out_2 = ['f', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')]
                        else:
                            op_out_1 = ['tmp_ba_u' + str(j),
                                        self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')]
                            op_out_2 = ['tmp_ba_f' + str(j),
                                        self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')]
                        self.sdc_sweep(op_in_1=op_in_1,
                                       op_in_2=op_in_2,
                                       op_in_3=None if level == 0 else ['tau',
                                                                        self.cr_dict(iteration=k, level=level,
                                                                                     time_point=i, colloc_node='all')],
                                       op_out_1=op_out_1,
                                       op_out_2=op_out_2,
                                       level=level,
                                       i=i)
                else:
                    self.copy(op_in=['v', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')],
                              op_out=['u', self.cr_dict(iteration=k, level=level, time_point=i, colloc_node='all')],
                              level=level,
                              i=i)
        if k == self.iterations:
            for i in range(1, self.nt):
                if self.iterations == 0:
                    self.f_eval_single(op_in=['u', self.cr_dict(iteration=self.iterations, level=0, time_point=i,
                                                                colloc_node='first')],
                                       op_out=['f', self.cr_dict(iteration=self.iterations, level=0, time_point=i,
                                                                 colloc_node='first')],
                                       level=0,
                                       i=i)
                self.sdc_sweep(
                    op_in_1=['u', self.cr_dict(iteration=self.iterations, level=0, time_point=i, colloc_node='all')],
                    op_in_2=['f', self.cr_dict(iteration=self.iterations, level=0, time_point=i, colloc_node='all')],
                    op_in_3=None,
                    op_out_1=['u', self.cr_dict(iteration=self.iterations, level=0, time_point=i, colloc_node='all')],
                    op_out_2=['v', self.cr_dict(iteration=self.iterations, level=0, time_point=i, colloc_node='all')],
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
        if self.predict_type == 'libpfasst_true':
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
                             op_in_2=['f', self.cr_dict(iteration=0, level=level + 1, time_point=i, colloc_node='all')],
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
                               op_in_3=['tau', self.cr_dict(iteration=0, level=level, time_point=i, colloc_node='all')],
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
