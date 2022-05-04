#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#include "braid.h"

typedef struct _braid_App_struct {
    int rank;
    double start_time;
    int *sleep_step;
    int *sleep_res;
    int *sleep_pro;
    int *nspace;
    int print_timing;
} my_App;

typedef struct _braid_Vector_struct {
    int size;
    double *values;
} my_Vector;

void
create_vector(my_Vector **u,
              int size) {
    (*u) = (my_Vector *) malloc(sizeof(my_Vector));
    ((*u)->size) = size;
    ((*u)->values) = (double *) malloc(size * sizeof(double));
}

int
my_Step(braid_App app,
        braid_Vector ustop,
        braid_Vector fstop,
        braid_Vector u,
        braid_StepStatus status) {
    double tstart;             /* current time */
    double tstop;              /* evolve to this time*/
    int level;
    double time_taken_start = MPI_Wtime() - app->start_time;
    braid_StepStatusGetLevel(status, &level);
    braid_StepStatusGetTstartTstop(status, &tstart, &tstop);
    usleep(app->sleep_step[level]);
    if (app->print_timing) {
        printf("Model | Rank: %d"
               "| Start: %f "
               "| Stop: %f "
               "| Type: Step_xbraid_level_%d "
               "| T_start: %f "
               "| t_stop: %f "
               "| size: %d \n", app->rank, time_taken_start, MPI_Wtime() - app->start_time,
               level, tstart, tstop, app->nspace[level]);
    }
    return 0;
}

int
my_Step_get_runtime(braid_App app,
                    braid_Vector ustop,
                    braid_Vector fstop,
                    braid_Vector u,
                    int level) {
    double time_taken_start = MPI_Wtime() - app->start_time;
    usleep(app->sleep_step[level]);
    if (0) {
        printf("Model | Rank: %d"
               "| Start: %f "
               "| Stop: %f "
               "| Type: Step_xbraid_level_%d "
               "| size: %d \n", app->rank, time_taken_start, MPI_Wtime() - app->start_time,
               level, app->nspace[level]);
    }
    return 0;
}

int
my_Init(braid_App app,
        double t,
        braid_Vector *u_ptr) {
    my_Vector *u;
    int i;
    int nspace = (app->nspace[0]);

    create_vector(&u, nspace);

    for (i = 0; i < nspace; i++) {
        (u->values)[i] = ((double) braid_Rand()) / braid_RAND_MAX;
    }

    *u_ptr = u;

    return 0;
}

int
my_Clone(braid_App app,
         braid_Vector u,
         braid_Vector *v_ptr) {
    my_Vector *v;
    int size = (u->size);
    int i;

    create_vector(&v, size);
    for (i = 0; i < size; i++) {
        (v->values)[i] = (u->values)[i];
    }
    *v_ptr = v;

    return 0;
}

int
my_Free(braid_App app,
        braid_Vector u) {
    free(u->values);
    free(u);

    return 0;
}

int
my_Sum(braid_App app,
       double alpha,
       braid_Vector x,
       double beta,
       braid_Vector y) {
    int i;
    int size = (y->size);

    for (i = 0; i < size; i++) {
        (y->values)[i] = alpha * (x->values)[i] + beta * (y->values)[i];
    }

    return 0;
}

int
my_SpatialNorm(braid_App app,
               braid_Vector u,
               double *norm_ptr) {
    *norm_ptr = 1;
    return 0;
}

int
my_Access(braid_App app,
          braid_Vector u,
          braid_AccessStatus astatus) {
    return 0;
}

int
my_BufSize(braid_App app,
           int *size_ptr,
           braid_BufferStatus bstatus) {
    int size = (app->nspace[0]);
    *size_ptr = (size + 1) * sizeof(double);
    return 0;
}

int
my_BufPack(braid_App app,
           braid_Vector u,
           void *buffer,
           braid_BufferStatus bstatus) {
    double *dbuffer = buffer;
    int i, size = (u->size);

    dbuffer[0] = size;
    for (i = 0; i < size; i++) {
        dbuffer[i + 1] = (u->values)[i];
    }

    braid_BufferStatusSetSize(bstatus, (size + 1) * sizeof(double));

    return 0;
}

int
my_BufUnpack(braid_App app,
             void *buffer,
             braid_Vector *u_ptr,
             braid_BufferStatus bstatus) {
    my_Vector *u = NULL;
    double *dbuffer = buffer;
    int i, size;

    size = dbuffer[0];
    create_vector(&u, size);
    for (i = 0; i < size; i++) {
        (u->values)[i] = dbuffer[i + 1];
    }
    *u_ptr = u;

    return 0;
}

int
my_Coarsen(braid_App app,
           braid_Vector fu,
           braid_Vector *cu_ptr,
           braid_CoarsenRefStatus status) {

    int level;
    double time_taken_start = MPI_Wtime() - app->start_time;
    braid_CoarsenRefStatusGetLevel(status, &level);
    my_Vector *v;
    create_vector(&v, app->nspace[level + 1]);
    *cu_ptr = v;
    usleep(app->sleep_res[level]);
    if (app->print_timing) {
        printf("Model | Rank: %d"
               "| Start: %f "
               "| Stop: %f "
               "| Type: Res_xbraid_level_%d \n", app->rank, time_taken_start, MPI_Wtime() - app->start_time, level);
    }
    return 0;
}

int
my_Coarsen_get_runtime(braid_App app,
                       braid_Vector fu,
                       braid_Vector *cu_ptr,
                       int level) {
    double time_taken_start = MPI_Wtime() - app->start_time;
    my_Vector *v;
    create_vector(&v, app->nspace[level + 1]);
    *cu_ptr = v;
    usleep(app->sleep_res[level]);
    if (app->print_timing) {
        printf("Model | Rank: %d"
               "| Start: %f "
               "| Stop: %f "
               "| Type: Res_xbraid_level_%d \n", app->rank, time_taken_start, MPI_Wtime() - app->start_time, level);
    }
    return 0;
}

int
my_Interp(braid_App app,
          braid_Vector cu,
          braid_Vector *fu_ptr,
          braid_CoarsenRefStatus status) {

    int level;
    double time_taken_start = MPI_Wtime() - app->start_time;
    braid_CoarsenRefStatusGetLevel(status, &level);
    my_Vector *v;
    create_vector(&v, app->nspace[level]);
    *fu_ptr = v;

    usleep(app->sleep_pro[level]);
    if (app->print_timing) {
        printf("Model | Rank: %d"
               "| Start: %f "
               "| Stop: %f "
               "| Type: Pro_xbraid_level_%d \n", app->rank, time_taken_start, MPI_Wtime() - app->start_time, level);
    }
    return 0;
}

int
my_Interp_get_runtime(braid_App app,
                      braid_Vector cu,
                      braid_Vector *fu_ptr,
                      int level) {
    double time_taken_start = MPI_Wtime() - app->start_time;
    my_Vector *v;
    create_vector(&v, app->nspace[level]);
    *fu_ptr = v;
    usleep(app->sleep_pro[level]);
    if (app->print_timing) {
        printf("Model | Rank: %d"
               "| Start: %f "
               "| Stop: %f "
               "| Type: Pro_xbraid_level_%d \n", app->rank, time_taken_start, MPI_Wtime() - app->start_time, level);
    }
    return 0;
}

/*--------------------------------------------------------------------------
 * Main driver
 *--------------------------------------------------------------------------*/

int main(int argc, char *argv[]) {
    braid_Core core;
    my_App *app;
    double tstart = 0.0;
    double tstop = 100;
    int ntime = 64;
    int rank = 0;
    int j = 0;
    int arg_index = 0;
    int max_levels = 2;

    while (arg_index < argc) {
        if (strcmp(argv[arg_index], "-ml") == 0) {
            arg_index++;
            max_levels = atoi(argv[arg_index++]);
        } else {
            arg_index++;
        }
    }
    int skip_cycle = 1;
    int max_iter = 3;
    int fmg = 0;
    int *cfactor = (int *) malloc(max_levels * sizeof(int));
    int *nrelax = (int *) malloc(max_levels * sizeof(int));
    int *sleep_step = (int *) malloc(max_levels * sizeof(int));
    int *sleep_pro = (int *) malloc(max_levels * sizeof(int));
    int *sleep_res = (int *) malloc(max_levels * sizeof(int));
    int *nspace = (int *) malloc(max_levels * sizeof(int));
    int print_timing = 0;
    int get_runtime = 0;
    for (j = 0; j < max_levels; j++) {
        cfactor[j] = 2;
        nrelax[j] = 1;
        sleep_step[j] = 10000;
        sleep_pro[j] = 100000;
        sleep_res[j] = 1000;
        nspace[j] = 20 - j;
    }

    ntime = 64;
    tstop = tstart + ntime / 0.01;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    arg_index = 1;
    while (arg_index < argc) {
        if (strcmp(argv[arg_index], "-ntime") == 0) {
            arg_index++;
            ntime = atoi(argv[arg_index++]);
            tstop = tstart + ntime / 2.;
        } else if (strcmp(argv[arg_index], "-nu") == 0) {
            arg_index++;
            for (j = 0; j < max_levels; j++) {
                nrelax[j] = atoi(argv[arg_index++]);
            }
        } else if (strcmp(argv[arg_index], "-cf") == 0) {
            arg_index++;
            for (j = 0; j < max_levels; j++) {
                cfactor[j] = atoi(argv[arg_index++]);
            }
        } else if (strcmp(argv[arg_index], "-mi") == 0) {
            arg_index++;
            max_iter = atoi(argv[arg_index++]);
        } else if (strcmp(argv[arg_index], "-fmg") == 0) {
            arg_index++;
            fmg = 1;
        } else if (strcmp(argv[arg_index], "-pt") == 0) {
            arg_index++;
            print_timing = atoi(argv[arg_index++]);
        } else if (strcmp(argv[arg_index], "-ss") == 0) {
            arg_index++;
            for (j = 0; j < max_levels; j++) {
                sleep_step[j] = atoi(argv[arg_index++]);
            }
        } else if (strcmp(argv[arg_index], "-sp") == 0) {
            arg_index++;
            for (j = 0; j < max_levels; j++) {
                sleep_pro[j] = atoi(argv[arg_index++]);
            }
        } else if (strcmp(argv[arg_index], "-sr") == 0) {
            arg_index++;
            for (j = 0; j < max_levels; j++) {
                sleep_res[j] = atoi(argv[arg_index++]);
            }
        } else if (strcmp(argv[arg_index], "-ns") == 0) {
            arg_index++;
            for (j = 0; j < max_levels; j++) {
                nspace[j] = atoi(argv[arg_index++]);
            }
        } else if (strcmp(argv[arg_index], "-sc") == 0) {
            arg_index++;
            skip_cycle = atoi(argv[arg_index++]);
        } else if (strcmp(argv[arg_index], "-gr") == 0) {
            arg_index++;
            get_runtime = atoi(argv[arg_index++]);
        } else {
            arg_index++;
        }
    }

    app = (my_App *) malloc(sizeof(my_App));
    (app->rank) = rank;
    (app->nspace) = nspace;
    (app->start_time) = MPI_Wtime();
    (app->sleep_step) = sleep_step;
    (app->sleep_pro) = sleep_pro;
    (app->sleep_res) = sleep_res;
    (app->print_timing) = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    if (get_runtime == 1) {
        if (rank == 0) {
            my_Vector *u_start;
            my_Vector *u_end;
            my_Vector *f;
            int times = 5;
            double min = 10000.0;
            double max = 0;
            double avg = 0;
            double start, stop;
            for (j = 0; j < max_levels; j++) {
                create_vector(&u_start, nspace[j]);
                create_vector(&u_end, nspace[j]);
                create_vector(&f, nspace[j]);
                avg = 0;
                min = 10000;
                max = 0;
                for (int i = 0; i < times; i++) {
                    start = MPI_Wtime();
                    my_Step_get_runtime(app, u_end, f, u_start, j);
                    stop = (MPI_Wtime() - start);
                    if (stop > max)
                        max = stop;
                    if (stop < min)
                        min = stop;
                    avg += stop;
                }
                avg = avg / times;
                printf("Measured costs: Step on level %i | max: %f | min: %f, | avg: %f \n", j, max, min, avg);
                free(u_start);
                free(u_end);
                free(f);
            }

            for (j = 0; j < max_levels - 1; j++) {
                create_vector(&u_start, nspace[j]);
                avg = 0;
                max = 0;
                min = 10000;
                for (int i = 0; i < times; i++) {
                    start = MPI_Wtime();
                    my_Coarsen_get_runtime(app, u_start, &u_end, j);
                    stop = (MPI_Wtime() - start);
                    if (stop > max)
                        max = stop;
                    if (stop < min)
                        min = stop;
                    avg += stop;
                }
                avg = avg / times;
                printf("Measured costs: Res on level %i | max: %f | min: %f, | avg: %f \n", j, max, min, avg);
                free(u_start);
                free(u_end);
                create_vector(&u_start, nspace[j]);

                avg = 0;
                max = 0;
                min = 10000;
                for (int i = 0; i < times; i++) {
                    start = MPI_Wtime();
                    my_Interp_get_runtime(app, u_start, &u_end, j);
                    stop = (MPI_Wtime() - start);
                    if (stop > max)
                        max = stop;
                    if (stop < min)
                        min = stop;
                    avg += stop;
                }
                avg = avg / times;
                printf("Measured costs: Pro on level %i | max: %f | min: %f, | avg: %f \n", j, max, min, avg);
                free(u_start);
                free(u_end);
            }
        }
    } else {
        MPI_Barrier(MPI_COMM_WORLD);
        (app->start_time) = MPI_Wtime();
        (app->print_timing) = print_timing;
        /* initialize XBraid and set options */
        braid_Init(MPI_COMM_WORLD, MPI_COMM_WORLD, tstart, tstop, ntime, app,
                   my_Step, my_Init, my_Clone, my_Free, my_Sum, my_SpatialNorm,
                   my_Access, my_BufSize, my_BufPack, my_BufUnpack, &core);

        /* Set some typical Braid parameters */
        braid_SetPrintLevel(core, 2);
        braid_SetMaxLevels(core, max_levels);
        braid_SetAbsTol(core, -1);
        for (j = 0; j < max_levels; j++) {
            braid_SetCFactor(core, j, cfactor[j]);
            braid_SetNRelax(core, j, nrelax[j]);
        }
        braid_SetMaxIter(core, max_iter);
        braid_SetSpatialCoarsen(core, my_Coarsen);
        braid_SetSpatialRefine(core, my_Interp);
        braid_SetSkip(core, skip_cycle);
        if (fmg) {
            braid_SetFMG(core);
        }
        /* Run simulation, and then clean up */
        braid_Drive(core);
        braid_Destroy(core);
    }
    free(app);
    free(cfactor);
    free(nrelax);
    free(sleep_step);
    free(sleep_pro);
    free(sleep_res);
    free(nspace);
    MPI_Finalize();

    return (0);
}
