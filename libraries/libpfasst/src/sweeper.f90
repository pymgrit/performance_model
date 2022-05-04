module pf_my_sweeper
    use pf_mod_ndarray
    use pf_mod_imex_sweeper
    use posix

    !>  extend the imex sweeper type with stuff we need to compute rhs
    type, extends(pf_imex_sweeper_t) :: my_sweeper_t
        integer :: nx   !  Grid size
        integer :: sleep_step
        integer :: sleep_f
        integer :: rank
        real*8 :: start
        integer :: level
    contains

        procedure :: f_eval    !  Computes the explicit rhs terms
        procedure :: f_comp    !  Does implicit solves
        procedure :: initialize  !  Bypasses base sweeper initialize

    end type my_sweeper_t

contains

    !>  Helper function to return sweeper pointer
    function as_my_sweeper(sweeper) result(r)
        class(pf_sweeper_t), intent(inout), target :: sweeper
        class(my_sweeper_t), pointer :: r
        select type(sweeper)
        type is (my_sweeper_t)
            r => sweeper
        class default
            stop
        end select
    end function as_my_sweeper

    function runtime_from(sweeper) result(r)
        class(pf_sweeper_t), intent(inout), target :: sweeper
        real*8 :: r
        select type(sweeper)
        type is (my_sweeper_t)
            r = MPI_Wtime () - sweeper%start
        class default
            stop
        end select
    end function runtime_from


    !>  Routine to initialize sweeper (bypasses imex sweeper initialize)
    subroutine initialize(this, pf, level_index)
        use probin, only : sleep_step, sleep_f
        class(my_sweeper_t), intent(inout) :: this
        type(pf_pfasst_t), intent(inout), target :: pf
        integer, intent(in) :: level_index

        integer :: nx
        integer :: rank

        !  Call the imex sweeper initialize
        call this%imex_initialize(pf, level_index)

        !>  Set variables for explicit and implicit parts
        this%implicit = .TRUE.
        this%explicit = .FALSE.

        nx = pf%levels(level_index)%lev_shape(1)  !  local convenient grid size
        !        this%sleep_step = sleep_step(level_index)  !  local convenient grid size
        !        this%sleep_f = sleep_f(level_index)  !  local convenient grid size
        !        print *, 'Bad case for piece in f_eval ', level_index, this%sleep_step, this%sleep_f

        this%rank = pf%rank
        this%start = MPI_Wtime()
        this%level = level_index

    end subroutine initialize

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! These routines must be provided for the sweeper
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! Evaluate the explicit function at y, t.
    subroutine f_eval(this, y, t, level_index, f, piece)
        use probin, only : lam1, lam2, sleep_f, print_timings
        class(my_sweeper_t), intent(inout) :: this
        class(pf_encap_t), intent(in) :: y
        class(pf_encap_t), intent(inout) :: f
        real(pfdp), intent(in) :: t
        integer, intent(in) :: level_index
        integer, intent(in) :: piece

        start = MPI_Wtime () - this%start

        rc = c_usleep(sleep_f(level_index))
        if (print_timings == 1) then
                print *, 'Model | Rank: ', this%rank, &
                        ' | Start: ', start, &
                        ' | Stop: ', MPI_Wtime () - this%start, &
                        ' | Type: F_libpfasst_level', level_index, &
                        ' | T: ', t
        end if

    end subroutine f_eval

    !> Solve for y and return f2 also.
    subroutine f_comp(this, y, t, dtq, rhs, level_index, f, piece)
        use probin, only : lam1, lam2, sleep_step, print_timings
        class(my_sweeper_t), intent(inout) :: this
        class(pf_encap_t), intent(inout) :: y
        real(pfdp), intent(in) :: t
        real(pfdp), intent(in) :: dtq
        class(pf_encap_t), intent(in) :: rhs
        integer, intent(in) :: level_index
        class(pf_encap_t), intent(inout) :: f
        integer, intent(in) :: piece

        real(pfdp), pointer :: yvec(:), rhsvec(:), fvec(:)

        start = MPI_Wtime () - this%start
        rc = c_usleep(sleep_step(level_index))
        ende = MPI_Wtime () - this%start
        if (print_timings == 1) then
            print *, 'Model | Rank: ', this%rank, &
                        ' | Start: ', start, &
                        ' | Stop: ', MPI_Wtime () - this%start, &
                        ' | Type: Step_libpfasst_level', level_index, &
                        ' | T: ', t
        end if
    end subroutine f_comp

end module pf_my_sweeper

