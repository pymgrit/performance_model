import numpy as np
from pint_task_graph import PintGraph


class Mgrit(PintGraph):
    def __init__(self,
                 coarsening,
                 cost_step,
                 cf_iter,
                 cycle_type,
                 placing_conv_crit,
                 nested_iterations=False,
                 skip_down=False,
                 cost_res=None,
                 cost_pro=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # MGRIT parameter
        self.coarsening = coarsening
        self.cf_iter = cf_iter
        self.cycle_type = cycle_type
        self.L = len(self.coarsening)
        self.nested_iterations = nested_iterations
        self.skip_down = skip_down
        self.placing_conv_crit = placing_conv_crit

        if nested_iterations and skip_down:
            print('Combination of nested iterations and skip down not really useful')

        # MGRIT specific costs
        if cost_res is None:
            self.cost_res = np.zeros(self.L)
        else:
            self.cost_res = cost_res
        if cost_pro is None:
            self.cost_pro = np.zeros(self.L)
        else:
            self.cost_pro = cost_pro
        self.time_steppers = cost_step

        # Dummies for datastructures etc
        self.f_blocks_per_level = []
        self.points_per_level = []
        self.c_points_per_level = []

        #Compute time hierarchy
        self.pre_and_after_info_per_level = [{} for _ in range(self.L)]
        for level in range(self.L):
            if level == 0:
                self.points_per_level.append(np.linspace(0, self.nt - 1, self.nt, dtype=int))
            else:
                self.points_per_level.append(np.copy(self.c_points_per_level[-1]))
            self.c_points_per_level.append(np.copy(self.points_per_level[-1][::self.coarsening[level]]))
            f_points_per_level = np.sort(np.array(list(
                set(self.points_per_level[-1].tolist()) - set(self.c_points_per_level[-1].tolist()))))
            tmp = self.consecutive(f_points_per_level, stepsize=np.prod(self.coarsening[:level]))
            tmp.reverse()
            self.f_blocks_per_level.append(tmp)
            for j in range(len(self.points_per_level[level])):
                pre = self.points_per_level[level][j - 1] if j != 0 else -99
                self.pre_and_after_info_per_level[level][self.points_per_level[level][j]] = pre

        # Initialize approximation at time points
        for level in range(self.L):
            for i in self.points_per_level[level]:
                set_val = self.create_node_name(var_name='u', var_dict=self.cr_dict(level=level, time_point=i))
                set_val += self.create_node_name(var_name='v', var_dict=self.cr_dict(level=level, time_point=i))
                set_val += self.create_node_name(var_name='g', var_dict=self.cr_dict(level=level, time_point=i))
                if i == 0:
                    self.add_node(name="Q|" + str(level),set_values=set_val,predecessors=['u_0'],cost=0,point=i)
                else:
                    self.add_node(name="Q|" + str(level), set_values=set_val, predecessors=['0'], cost=0, point=i)

        # Setup convergence criterion
        self.cc = {}
        if self.placing_conv_crit == 0:
            for i in self.c_points_per_level[0]:
                if i > 0:
                    cc = self.create_node_name(var_name='u', var_dict=self.cr_dict(level=0, time_point=i))
                    cc += self.create_node_name(var_name='u', var_dict=self.cr_dict(level=0, time_point=i - 1))
                    self.cc[i] = cc
        elif self.placing_conv_crit == 1:
            for i in self.c_points_per_level[0]:
                if i > 0:
                    self.cc[i] = self.create_node_name(var_name='r', var_dict=self.cr_dict(level=0, time_point=i))
        else:
            raise Exception('Unknown placing')

    def f_relax(self, level: int) -> None:
        """
        F-relaxation

        :param level: Level
        """
        for block in self.f_blocks_per_level[level]:
            for i in block:
                pre = self.pre_and_after_info_per_level[level][i]
                if level == 0:
                    self.add_node(name="F|" + str(level),
                                  predecessors=self.create_node_name(var_name='u',
                                                                     var_dict=self.cr_dict(level=level,
                                                                                           time_point=pre)),
                                  set_values=self.create_node_name(var_name='u',
                                                                   var_dict=self.cr_dict(level=level, time_point=i)),
                                  cost=self.time_steppers[level],
                                  point=i,
                                  description='mgrit_step_' + str(level))
                else:
                    pre = self.create_node_name(var_name='u', var_dict=self.cr_dict(level=level, time_point=pre))
                    pre += self.create_node_name(var_name='g', var_dict=self.cr_dict(level=level, time_point=i))
                    self.add_node(name="F|" + str(level),
                                  predecessors=pre,
                                  set_values=self.create_node_name(var_name='u',
                                                                   var_dict=self.cr_dict(level=level, time_point=i)),
                                  cost=self.time_steppers[level],
                                  point=i,
                                  description='mgrit_step_' + str(level))

    def c_relax(self, level: int) -> None:
        """
        C-relaxation

        :param level: Level
        """
        for i in self.c_points_per_level[level]:
            pre = self.pre_and_after_info_per_level[level][i]
            if i != 0:
                if level == 0:
                    self.add_node(name="C|" + str(level),
                                  predecessors=self.create_node_name(var_name='u',
                                                                     var_dict=self.cr_dict(level=level,
                                                                                           time_point=pre)),
                                  set_values=self.create_node_name(var_name='u',
                                                                   var_dict=self.cr_dict(level=level, time_point=i)),
                                  cost=self.time_steppers[level],
                                  point=i,
                                  description='mgrit_step_' + str(level))
                else:
                    pred = self.create_node_name(var_name='u', var_dict=self.cr_dict(level=level, time_point=pre))
                    pred += self.create_node_name(var_name='g', var_dict=self.cr_dict(level=level, time_point=i))
                    self.add_node(name="C|" + str(level),
                                  predecessors=pred,
                                  set_values=self.create_node_name(var_name='u',
                                                                   var_dict=self.cr_dict(level=level, time_point=i)),
                                  cost=self.time_steppers[level],
                                  point=i,
                                  description='mgrit_step_' + str(level))

    def restrict(self, level: int) -> None:
        """
        Restrict C-points to next coarser level

        :param level: Level
        """
        for i in self.c_points_per_level[level]:
            self.add_node(name="R|" + str(level),
                          predecessors=self.create_node_name(var_name='u',
                                                             var_dict=self.cr_dict(level=level, time_point=i)),
                          set_values=self.create_node_name(var_name='u',
                                                           var_dict=self.cr_dict(level=level + 1, time_point=i)),
                          cost=self.cost_res[level],
                          point=i,
                          description='mgrit_res_' + str(level))
            self.add_node(name="R|" + str(level),
                          predecessors=['u_' + str(level + 1) + '_' + str(i)],
                          set_values=["v_" + str(level + 1) + "_" + str(i)],
                          cost=0,
                          point=i,
                          description='mgrit_copy_' + str(level))

    def residual(self, level: int) -> None:
        """
        Compute residual

        :param level: Level
        """
        for i in self.c_points_per_level[level]:
            pre_fine = self.pre_and_after_info_per_level[level][i]
            if i != 0:
                pred = self.create_node_name(var_name='g', var_dict=self.cr_dict(level=level, time_point=i))
                pred += self.create_node_name(var_name='u', var_dict=self.cr_dict(level=level, time_point=pre_fine))
                pred += self.create_node_name(var_name='u', var_dict=self.cr_dict(level=level, time_point=i))
                self.add_node(name="FR|" + str(level),
                              predecessors=pred,
                              set_values=self.create_node_name(var_name='r',
                                                               var_dict=self.cr_dict(level=level, time_point=i)),
                              cost=self.time_steppers[level],
                              point=i,
                              description='mgrit_step_' + str(level))

    def fas_residual(self, level: int) -> None:
        """
        Compute FAS residual

        :param level: Level
        """
        for i in self.c_points_per_level[level]:
            pre_coarse = self.pre_and_after_info_per_level[level + 1][i]
            if i != 0:
                self.add_node(name="R|" + str(level),
                              predecessors=self.create_node_name(var_name='r',
                                                                 var_dict=self.cr_dict(level=level, time_point=i)),
                              set_values=self.create_node_name(var_name='g',
                                                               var_dict=self.cr_dict(level=level + 1, time_point=i)),
                              cost=self.cost_res[level],
                              point=i,
                              description='mgrit_res_' + str(level))
                pred = self.create_node_name(var_name='g', var_dict=self.cr_dict(level=level + 1, time_point=i))
                pred += self.create_node_name(var_name='u',
                                              var_dict=self.cr_dict(level=level + 1, time_point=pre_coarse))
                pred += self.create_node_name(var_name='u', var_dict=self.cr_dict(level=level + 1, time_point=i))
                self.add_node(name="FR|" + str(level + 1),
                              predecessors=pred,
                              set_values=self.create_node_name(var_name='g',
                                                               var_dict=self.cr_dict(level=level + 1, time_point=i)),
                              cost=self.time_steppers[level + 1],
                              point=i,
                              description='mgrit_step_' + str(level + 1))

    def coarsest_level(self, level: int) -> None:
        """
        Coarsest level

        :param level: Level
        """
        for i in self.points_per_level[level]:
            pre = self.pre_and_after_info_per_level[level][i]
            if i != 0:
                pred = self.create_node_name(var_name='u', var_dict=self.cr_dict(level=level, time_point=pre))
                pred += self.create_node_name(var_name='g', var_dict=self.cr_dict(level=level, time_point=i))
                self.add_node(name="TS|" + str(level),
                              predecessors=pred,
                              set_values=self.create_node_name(var_name='u',
                                                               var_dict=self.cr_dict(level=level, time_point=i)),
                              cost=self.time_steppers[level],
                              point=i,
                              description='mgrit_step_' + str(level))

    def consecutive(self, data: np.ndarray, stepsize: int = 1) -> np.ndarray:
        """
        Auxiliary function to obtain blocks of F points

        :param data: List of time points
        :param stepsize: Stepsize between points
        :return: F-blocks
        """
        return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)

    def error_correction(self, level: int) -> None:
        """
        Error correction

        :param level: Current level
        """
        for i in self.c_points_per_level[level]:
            if i != 0:
                pred = self.create_node_name(var_name='u', var_dict=self.cr_dict(level=level + 1, time_point=i))
                pred += self.create_node_name(var_name='v', var_dict=self.cr_dict(level=level + 1, time_point=i))
                pred += self.create_node_name(var_name='u', var_dict=self.cr_dict(level=level, time_point=i))
                self.add_node(name="E|" + str(level),
                              predecessors=pred,
                              set_values=self.create_node_name(var_name='u',
                                                               var_dict=self.cr_dict(level=level, time_point=i)),
                              cost=self.cost_pro[level],
                              point=i,
                              description='mgrit_pro_' + str(level))

    def nested_iteration_interpolation(self, level: int) -> None:
        """
        Interpolation within nested iterations

        :param level: Current level
        """
        for i in self.c_points_per_level[level]:
            pre = self.pre_and_after_info_per_level[level][i]
            if i != 0:
                self.add_node(name="I|" + str(level),
                              predecessors=self.create_node_name(var_name='u',
                                                                 var_dict=self.cr_dict(level=level, time_point=i)),
                              set_values=self.create_node_name(var_name='u',
                                                               var_dict=self.cr_dict(level=level - 1, time_point=i)),
                              cost=self.cost_pro[level],
                              point=i,
                              description='mgrit_pro_' + str(level))

    def restrict_for_skip_down(self) -> None:
        """
        Restriction within skip down

        """
        for level in range(0, self.L - 1):
            for i in self.c_points_per_level[level]:
                self.add_node(name="R|" + str(level),
                              predecessors=self.create_node_name(var_name='u',
                                                                 var_dict=self.cr_dict(level=level, time_point=i)),
                              set_values=self.create_node_name(var_name='u',
                                                               var_dict=self.cr_dict(level=level + 1, time_point=i)),
                              cost=self.cost_res[level],
                              point=i,
                              description='mgrit_res_' + str(level))

    def compute(self) -> None:
        """
        Computes the graph

        """
        if self.nested_iterations:
            self.restrict_for_skip_down()
            for level in range(self.L - 1, 0, -1):
                self.mgrit(level=level, first_f=True, it=-1, cycle_type='V')
                self.nested_iteration_interpolation(level - 1)
        for it in range(self.iterations):
            if it == 0 and self.skip_down:
                self.restrict_for_skip_down()
                self.mgrit(level=0, first_f=True, it=0, skip_down=True, cycle_type=self.cycle_type)
            else:
                self.mgrit(level=0, first_f=True, it=it, cycle_type=self.cycle_type)
            if self.placing_conv_crit == 0:
                self.convergence_criterion(poins_with_dependencies=self.cc)

    def mgrit(self, level: object, cycle_type: object, first_f: object, it: object,
              skip_down: object = False) -> None:
        """
        MGRIT iteration

        :param level: Current level
        :param cycle_type: Cycle type
        :param first_f: Perform first F-relaxation
        :param it: Iteration count
        :param skip_down: Skip down
        :return:
        """
        if level == self.L - 1:
            self.coarsest_level(level=level)
            return
        else:
            if not skip_down:
                if (level > 0 or (it == 0 and level == 0)) and first_f and not skip_down:
                    self.f_relax(level=level)
                for _ in range(self.cf_iter[level]):
                    self.c_relax(level=level)
                    self.f_relax(level=level)
                self.residual(level=level)
                self.restrict(level=level)
                self.fas_residual(level=level)
                if self.placing_conv_crit == 1 and level == 0:
                    self.convergence_criterion(poins_with_dependencies=self.cc)
            self.mgrit(level=level + 1, cycle_type=cycle_type, first_f=True, it=it, skip_down=skip_down)
            self.error_correction(level=level)
            self.f_relax(level=level)

            if level != 0 and cycle_type == 'F':
                self.mgrit(level=level, cycle_type='V', it=it, first_f=False)
        return
