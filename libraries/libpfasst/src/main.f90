program main
    use pf_mod_mpi

    integer :: ierror

    !> Initialize MPI
    call mpi_init(ierror)
    if (ierror /= 0) &
            stop "ERROR: Can't initialize MPI."

    !> Call the  solver
    call run_pfasst()

    !> Close mpi
    call mpi_finalize(ierror)

contains
    !>  This subroutine setups and calls libpfasst
    subroutine run_pfasst()
        use pfasst        !<  This module has include statements for the main pfasst routines
        use pf_my_sweeper !<  Local module for sweeper
        use pf_my_level   !<  Local module for level
        use probin        !<  Local module reading/parsing problem parameters

        implicit none

        !>  Local variables
        type(pf_pfasst_t) :: pf       !<  the main pfasst structure
        type(pf_comm_t) :: comm     !<  the communicator (here it is mpi)
        type(pf_ndarray_t) :: y_0      !<  the initial condition
        character(256) :: pf_fname   !<  file name for input of PFASST parameters
        real*8 :: start, stop, max, min, avg

        integer :: l, j, times

        !> Read problem parameters
        call probin_init(pf_fname)

        !>  Set up communicator
        call pf_mpi_create(comm, MPI_COMM_WORLD)

        !>  Create the pfasst structure
        call pf_pfasst_create(pf, comm, fname = pf_fname)

        !> Loop over levels and set some level specific parameters
        do l = 1, pf%nlevels
            !>  Allocate the user specific level object
            allocate(my_level_t :: pf%levels(l)%ulevel)

            !>  Allocate the user specific data constructor
            allocate(pf_ndarray_factory_t :: pf%levels(l)%ulevel%factory)

            !>  Allocate the sweeper at this level
            allocate(my_sweeper_t :: pf%levels(l)%ulevel%sweeper)

            !>  Set the size of the data on this level (here just one)
            call pf_level_set_size(pf, l, [nx(l)])
        end do

        !>  Set up some pfasst stuff
        call pf_pfasst_setup(pf)

        if (get_runtimes == 1) then
            call initialize_results(pf)
            if (pf%rank == 0) then
                times = 10
                do l = pf%nlevels, 1, -1
                    call ndarray_build(y_0, [ nx(l) ])
                    call y_0%setval(4.0_pfdp)
                    call pf%levels(l)%q0%copy(y_0)
                    call ndarray_destroy(y_0)
                end do
                do l = pf%nlevels, 1, -1
                    max = 0
                    min = 10000.0
                    avg = 0
                    do j = 1, times
                        start = MPI_Wtime()
                        call pf%levels(l)%ulevel%sweeper%sweep(pf, l, pf%state%t0, pf%state%dt, 1)
                        stop = MPI_Wtime() - start
                        if (stop > max) then
                            max = stop
                        end if
                        if (stop < min) then
                            min = stop
                        end if
                        avg = avg + stop
                    end do
                    avg = avg / times
                    print *, 'Measured costs: Sweeper on level: ', l, '| max:', max, '| min:', min, '| avg:', avg
                    max = 0
                    min = 10000.0
                    avg = 0
                    do j = 1, times
                        start = MPI_Wtime()
                        call pf%levels(l)%ulevel%sweeper%evaluate_all(pf,l, pf%state%t0+dt*pf%levels(l)%nodes)
                        stop = MPI_Wtime() - start
                        if (stop > max) then
                            max = stop
                        end if
                        if (stop < min) then
                            min = stop
                        end if
                        avg = avg + stop
                    end do
                    avg = avg / times
                    print *, 'Measured costs: EvalFAll on level: ', l, '| max:', max, '| min:', min, '| avg:', avg
                    max = 0
                    min = 10000.0
                    avg = 0
                    do j = 1, times
                        start = MPI_Wtime()
                        call pf%levels(l)%ulevel%sweeper%evaluate(pf,l,pf%state%t0+dt*pf%levels(l)%nodes(1),1)
                        stop = MPI_Wtime() - start
                        if (stop > max) then
                            max = stop
                        end if
                        if (stop < min) then
                            min = stop
                        end if
                        avg = avg + stop
                    end do
                    avg = avg / times
                    print *, 'Measured costs: EvalFSingle on level: ', l, '| max:', max, '| min:', min, '| avg:', avg
                end do
                do l = pf%nlevels, 2, -1
                    max = 0
                    min = 10000.0
                    avg = 0
                    do j = 1, times
                        start = MPI_Wtime()
                        call restrict_time_space_fas(pf, pf%state%t0, pf%state%dt, l)
                        stop = MPI_Wtime() - start
                        if (stop > max) then
                            max = stop
                        end if
                        if (stop < min) then
                            min = stop
                        end if
                        avg = avg + stop
                    end do
                    avg = avg / times
                    print *, 'Measured costs: Res on level: ', l, '| max:', max, '| min:', min, '| avg:', avg
                    max = 0
                    min = 10000.0
                    avg = 0
                    do j = 1, times
                        start = MPI_Wtime()
                        call interpolate_time_space(pf, pf%state%t0, dt, l, .false.)
                        stop = MPI_Wtime() - start
                        if (stop > max) then
                            max = stop
                        end if
                        if (stop < min) then
                            min = stop
                        end if
                        avg = avg + stop
                    end do
                    avg = avg / times
                    print *, 'Measured costs: Pro on level: ', l, '| max:', max, '| min:', min, '| avg:', avg
                    max = 0
                    min = 10000.0
                    avg = 0
                    do j = 1, times
                        start = MPI_Wtime()
                        call interpolate_q0(pf, pf%levels(l), pf%levels(l-1),flags=0)
                        stop = MPI_Wtime() - start
                        if (stop > max) then
                            max = stop
                        end if
                        if (stop < min) then
                            min = stop
                        end if
                        avg = avg + stop
                    end do
                    avg = avg / times
                    print *, 'Measured costs: SingleCorrect on level: ', l, '| max:', max, '| min:', min, '| avg:', avg
                    max = 0
                    min = 10000.0
                    avg = 0
                    do j = 1, times
                        start = MPI_Wtime()
                        call restrict_ts(pf%levels(l), pf%levels(l-1), pf%levels(l)%Q, pf%levels(l-1)%Q, pf%state%t0+dt*pf%levels(l)%nodes)
                        stop = MPI_Wtime() - start
                        if (stop > max) then
                            max = stop
                        end if
                        if (stop < min) then
                            min = stop
                        end if
                        avg = avg + stop
                    end do
                    avg = avg / times
                    print *, 'Measured costs: RestrictAll on level: ', l, '| max:', max, '| min:', min, '| avg:', avg
                    max = 0
                    min = 10000.0
                    avg = 0
                    do j = 1, times
                        start = MPI_Wtime()
                        call pf%levels(l)%ulevel%restrict(pf%levels(l), pf%levels(l-1), pf%levels(l)%Q(1), pf%levels(l)%f_encap_array_c(1), pf%state%t0+dt*pf%levels(l)%nodes(1))
                        stop = MPI_Wtime() - start
                        if (stop > max) then
                            max = stop
                        end if
                        if (stop < min) then
                            min = stop
                        end if
                        avg = avg + stop
                    end do
                    avg = avg / times
                    print *, 'Measured costs: RestrictSingle on level: ', l, '| max:', max, '| min:', min, '| avg:', avg
                    max = 0
                    min = 10000.0
                    avg = 0
                    do j = 1, times
                        start = MPI_Wtime()
                        call pf%levels(l)%ulevel%interpolate(pf%levels(l),pf%levels(l-1), pf%levels(l)%cf_delta(1), pf%levels(l)%c_delta(1), pf%state%t0+dt*pf%levels(l-1)%nodes(1))
                        stop = MPI_Wtime() - start
                        if (stop > max) then
                            max = stop
                        end if
                        if (stop < min) then
                            min = stop
                        end if
                        avg = avg + stop
                    end do
                    avg = avg / times
                    print *, 'Measured costs: InterpolateSingle on level: ', l, '| max:', max, '| min:', min, '| avg:', avg
                end do
            end if
        else
            !> add some hooks for output  (using a LibPFASST hook here)
            call pf_add_hook(pf, -1, PF_POST_ITERATION, pf_echo_residual)

            !>  Output run parameters to screen
            call print_loc_options(pf, un_opt = 6)

            !>  Allocate initial consdition
            call ndarray_build(y_0, [ nx(pf%nlevels) ])

            !> Set the initial condition
            call y_0%setval(1.0_pfdp)

            !> Do the PFASST time stepping
            call pf_pfasst_run(pf, y_0, dt, 0.0_pfdp, nsteps)

            !>  Wait for everyone to be done
            call mpi_barrier(pf%comm%comm, ierror)

            !>  Deallocate initial condition and final solution
            call ndarray_destroy(y_0)
        end if

        !>  Deallocate pfasst structure
        call pf_pfasst_destroy(pf)

    end subroutine run_pfasst

end program
