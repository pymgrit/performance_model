from pint_task_graph import PintGraph


class Parareal(PintGraph):
    def __init__(self, cost_fine: float, cost_coarse: float, cost_copy: float = 0, cost_correction: float = 0,
                 *args: object, **kwargs: object) -> None:
        """
        Constructor

        :param cost_fine: Cost of the fine propagator
        :param cost_coarse: Cost of the coarse propagator
        :param cost_copy: Cost of a copy operation
        :param cost_correction: Cost of a correction
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.cost_coarse = cost_coarse
        self.cost_fine = cost_fine
        self.cost_correction = cost_correction
        self.cost_copy = cost_copy

    def parareal_iteration(self, k: int) -> None:
        """
        Parareal iteration

        :param k: Iteration number
        """
        # Foreach loop, starting with the highest index to simplify standard schedule
        for i in range(self.nt - 1, k - 1, -1):
            self.add_node(name="F",
                          predecessors=self.create_node_name(var_name='u',
                                                             var_dict=self.cr_dict(level=0, time_point=i - 1,
                                                                                   iteration=k - 1)),
                          set_values=self.create_node_name(var_name='hatu', var_dict=self.cr_dict(level=0, time_point=i,
                                                                                                  iteration=k - 1)),
                          cost=self.cost_fine,
                          point=i,
                          description='parareal_fine_operator')

        for i in range(k, self.nt):
            # Coarse Propagator
            self.add_node(name="G",
                          predecessors=self.create_node_name(var_name='u',
                                                             var_dict=self.cr_dict(level=0, time_point=i - 1,
                                                                                   iteration=min(k, i - 1))),
                          set_values=self.create_node_name(var_name='tildeu',
                                                           var_dict=self.cr_dict(level=0, time_point=i, iteration=k)),
                          cost=self.cost_coarse,
                          point=i,
                          description='parareal_coarse_operator')
            # add/sub
            pred = self.create_node_name(var_name='tildeu', var_dict=self.cr_dict(level=0, time_point=i, iteration=k))
            pred += self.create_node_name(var_name='hatu',
                                          var_dict=self.cr_dict(level=0, time_point=i, iteration=k - 1))
            pred += self.create_node_name(var_name='tildeu',
                                          var_dict=self.cr_dict(level=0, time_point=i, iteration=k - 1))
            self.add_node(name="+",
                          predecessors=pred,
                          set_values=self.create_node_name(var_name='u',
                                                           var_dict=self.cr_dict(level=0, time_point=i, iteration=k)),
                          cost=self.cost_correction,
                          point=i,
                          description='parareal_correction')

    def compute_initial_guess(self) -> None:
        """
        Initial guess
        """
        for i in range(1, self.nt):
            # Coarse propagator
            self.add_node(name="G",
                          predecessors=self.create_node_name(var_name='u',
                                                             var_dict=self.cr_dict(level=0, time_point=i - 1,
                                                                                   iteration=0)),
                          set_values=self.create_node_name(var_name='tildeu',
                                                           var_dict=self.cr_dict(level=0, time_point=i, iteration=0)),
                          cost=self.cost_coarse,
                          point=i,
                          description='parareal_coarse_operator')
            # Copy
            self.add_node(name="C",
                          predecessors=self.create_node_name(var_name='tildeu',
                                                             var_dict=self.cr_dict(level=0, time_point=i, iteration=0)),
                          set_values=self.create_node_name(var_name='u',
                                                           var_dict=self.cr_dict(level=0, time_point=i, iteration=0)),
                          cost=self.cost_copy,
                          point=i,
                          description='parareal_copy')

    def compute(self) -> None:
        """
        Computes the graph
        """
        # Copy initial value
        self.add_node(name="C",
                      predecessors=['u_0'],
                      set_values=self.create_node_name(var_name='u',
                                                       var_dict=self.cr_dict(level=0, time_point=0, iteration=0)),
                      cost=self.cost_copy,
                      point=0)

        self.compute_initial_guess()
        for k in range(1, self.iterations + 1):
            self.parareal_iteration(k=k)
            cc = {}
            for i in range(k, self.nt):
                tmp = self.create_node_name(var_name='u', var_dict=self.cr_dict(level=0, time_point=i, iteration=k))
                tmp += self.create_node_name(var_name='u',
                                             var_dict=self.cr_dict(level=0, time_point=i, iteration=k - 1))
                cc[i] = tmp
        self.convergence_criterion(poins_with_dependencies=cc)
