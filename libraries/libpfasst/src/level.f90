module pf_my_level
    use pf_mod_ndarray
    use posix

    !>  extend the generic level type by defining transfer operators
    type, extends(pf_user_level_t) :: my_level_t
    contains
        procedure :: restrict
        procedure :: interpolate
    end type my_level_t

contains

    !>  Interpolate from coarse  level to fine
    subroutine interpolate(this, f_lev, c_lev, f_vec, c_vec, t, flags)
        use pf_my_sweeper, only : my_sweeper_t, as_my_sweeper, runtime_from
        use probin, only : sleep_pro, print_timings

        class(my_sweeper_t), pointer :: sweeper_f, sweeper_c  !  fine and coarse sweepers
        class(my_level_t), intent(inout) :: this
        class(pf_level_t), intent(inout) :: f_lev, c_lev  !  fine and coarse levels
        class(pf_encap_t), intent(inout) :: f_vec, c_vec  !  fine and coarse vectors
        real(pfdp), intent(in) :: t
        integer, intent(in), optional :: flags

        sweeper_f => as_my_sweeper(f_lev%ulevel%sweeper)
        start = runtime_from(sweeper_f)
        rc = c_usleep(sleep_pro(sweeper_f%level))
        call f_vec%copy(c_vec)
        if (print_timings == 1) then
            print *, 'Model | Rank:', sweeper_f%rank, &
                    ' | Start: ', start, &
                    ' | Stop: ', runtime_from(sweeper_f), &
                    ' | Type: Pro_libpfasst_level', sweeper_f%level, &
                    ' | T: ', t
        end if

    end subroutine interpolate

    !>  Restrict from fine level to coarse
    subroutine restrict(this, f_lev, c_lev, f_vec, c_vec, t, flags)
        use pf_my_sweeper, only : my_sweeper_t, as_my_sweeper, runtime_from
        use probin, only : sleep_res, print_timings

        class(my_sweeper_t), pointer :: sweeper_f, sweeper_c  !  fine and coarse sweepers
        class(my_level_t), intent(inout) :: this
        class(pf_level_t), intent(inout) :: f_lev, c_lev  !  fine and coarse levels
        class(pf_encap_t), intent(inout) :: f_vec, c_vec  !  fine and coarse vectors
        real(pfdp), intent(in) :: t      !<  time of solution
        integer, intent(in), optional :: flags

        sweeper_f => as_my_sweeper(f_lev%ulevel%sweeper)
        start = runtime_from(sweeper_f)
        rc = c_usleep(sleep_res(sweeper_f%level))
        call c_vec%copy(f_vec)
        if (print_timings == 1) then
            print *, 'Model | Rank:', sweeper_f%rank, &
                    ' | Start: ', start, &
                    ' | Stop: ', runtime_from(sweeper_f), &
                    ' | Type: Res_libpfasst_level', sweeper_f%level, &
                    ' | T: ', t
        end if
    end subroutine restrict


end module pf_my_level
