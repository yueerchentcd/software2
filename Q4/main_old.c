#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <mpi.h>

#include "poisson1d.h"
#include "jacob2d.h"
#include "decomp1d.h"
#include "decomp2d.h"

#define maxit 2000

void init_full_grid(double g[][maxn]);
void init_full_grids(double a[][maxn], double b[][maxn], double f[][maxn]);
void copy_full_grid(double a[][maxn], double b[][maxn]);

void onedinit_q2(double a[][maxn], double b[][maxn], double f[][maxn],
                 int nx, int ny, int s, int e);

void GatherGrid(double g[][maxn], double a[][maxn], int nx, int ny, MPI_Comm comm);
void write_grid(const char *fname, double x[][maxn], int nx, int ny);

double exact_u(double x, double y);
double bc_left(double y);
double bc_right(double y);
double bc_bottom(double x);
double bc_top(double x);

int main(int argc, char **argv)
{
    double a[maxn][maxn], b[maxn][maxn], f[maxn][maxn];
    double ggrid[maxn][maxn];

    int nx, ny;
    int myid, nprocs;
    int s, e, it;

    /* new 2D decomposition indices */
    int sx, ex, sy, ey;
    int local_nx, local_ny;

    double glob_diff, ldiff;
    double t1, t2;
    double tol = 1.0E-11;

    double local_sse, global_sse;
    double local_maxerr, global_maxerr;
    double err, x, y, h, rmse;

    MPI_Comm cart_comm;
    int dims[2], periods[2], coords[2], reorder;
    int nbr_up, nbr_down, nbr_left, nbr_right;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (myid == 0) {
        nx = 31;

        if (argc > 2) {
            fprintf(stderr, "Usage: mpirun -np <nproc> %s <nx>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (argc == 2) {
            nx = atoi(argv[1]);
        }

        if (nx > maxn - 2) {
            fprintf(stderr, "grid size too large for maxn; increase maxn in poisson1d.h\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    ny = nx;

    init_full_grids(a, b, f);
    init_full_grid(ggrid);

    dims[0] = 0;
    dims[1] = 0;
    MPI_Dims_create(nprocs, 2, dims);

    periods[0] = 0;
    periods[1] = 0;
    reorder = 0;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);
    MPI_Cart_coords(cart_comm, myid, 2, coords);

    MPI_Cart_shift(cart_comm, 0, 1, &nbr_up, &nbr_down);
    MPI_Cart_shift(cart_comm, 1, 1, &nbr_left, &nbr_right);

    /* 2D decomposition */
    MPE_Decomp2d(nx, ny, dims, coords, &sx, &ex, &sy, &ey);
    local_nx = local_size_1d(sx, ex);
    local_ny = local_size_1d(sy, ey);

    /*
      TEMPORARY:
      We still keep the old 1D variables s,e alive so the old 1D Q3 code
      continues to compile. For now we map them to the x-direction block.
      This will be removed when we replace the 1D communication/sweep with 2D versions.
    */
    s = sx;
    e = ex;

    if (myid == 0) {
        printf("========================================\n");
        printf("Q4 run: parallel Poisson solver\n");
        printf("Grid size            : %d x %d\n", nx, ny);
        printf("Processes used       : %d\n", nprocs);
        printf("Process grid         : %d x %d\n", dims[0], dims[1]);
        printf("Jacobi iterations    : %d\n", maxit);
        printf("========================================\n");
    }

    MPI_Barrier(cart_comm);
    printf("rank %d coords=(%d,%d) x:[%d,%d] y:[%d,%d] local=(%d,%d) nbrs U/D/L/R=(%d,%d,%d,%d)\n",
           myid, coords[0], coords[1], sx, ex, sy, ey, local_nx, local_ny,
           nbr_up, nbr_down, nbr_left, nbr_right);
    MPI_Barrier(cart_comm);

    /*
      TEMPORARY:
      still using old Q3 initialisation, which is only correct for the 1D-style layout.
      We keep it for now so this stage compiles and runs while we verify the 2D decomposition.
    */
    onedinit_q2(a, b, f, nx, ny, s, e);

    MPI_Barrier(cart_comm);
    t1 = MPI_Wtime();

    glob_diff = 1.0e100;

    for (it = 0; it < maxit; it++) {
        exchangi1(a, nx, s, e, cart_comm, nbr_left, nbr_right);
        sweep1d(a, f, nx, s, e, b);

        exchangi1(b, nx, s, e, cart_comm, nbr_left, nbr_right);
        sweep1d(b, f, nx, s, e, a);

        ldiff = griddiff(a, b, nx, s, e);
        MPI_Allreduce(&ldiff, &glob_diff, 1, MPI_DOUBLE, MPI_SUM, cart_comm);

        if (myid == 0 && it % 100 == 0) {
            printf("iter %4d: global diff = %.12e\n", it, glob_diff);
        }

        if (glob_diff < tol) {
            if (myid == 0) {
                printf("Iterative solve converged at iteration %d\n", it);
            }
            break;
        }
    }

    MPI_Barrier(cart_comm);
    t2 = MPI_Wtime();

    h = 1.0 / ((double)(nx + 1));
    local_sse = 0.0;
    local_maxerr = 0.0;

    for (int i = s; i <= e; i++) {
        x = i * h;
        for (int j = 1; j <= ny; j++) {
            y = j * h;
            err = fabs(a[i][j] - exact_u(x, y));
            local_sse += err * err;
            if (err > local_maxerr) {
                local_maxerr = err;
            }
        }
    }

    MPI_Reduce(&local_sse, &global_sse, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
    MPI_Reduce(&local_maxerr, &global_maxerr, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    GatherGrid(ggrid, a, nx, ny, cart_comm);

    if (myid == 0) {
        char fname[64];
        snprintf(fname, sizeof(fname), "q4_grid_%d.dat", nx);

        rmse = sqrt(global_sse / (double)(nx * ny));

        printf("Finished after %d iterations\n", it);
        printf("Run took %.6lf s\n", t2 - t1);
        printf("Max abs error = %.12e\n", global_maxerr);
        printf("RMSE          = %.12e\n", rmse);

        write_grid(fname, ggrid, nx, ny);
        printf("Gathered grid written to %s\n", fname);
    }

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}

double exact_u(double x, double y)
{
    return y / (((1.0 + x) * (1.0 + x)) + y * y);
}

double bc_left(double y)
{
    return y / (1.0 + y * y);
}

double bc_right(double y)
{
    return y / (4.0 + y * y);
}

double bc_bottom(double x)
{
    (void)x;
    return 0.0;
}

double bc_top(double x)
{
    return 1.0 / (((1.0 + x) * (1.0 + x)) + 1.0);
}

void onedinit_q2(double a[][maxn], double b[][maxn], double f[][maxn],
                 int nx, int ny, int s, int e)
{
    int i, j;
    double h, x, y;

    h = 1.0 / ((double)(nx + 1));

    for (i = s - 1; i <= e + 1; i++) {
        for (j = 0; j <= ny + 1; j++) {
            a[i][j] = 0.0;
            b[i][j] = 0.0;
            f[i][j] = 0.0;
        }
    }

    for (i = s; i <= e; i++) {
        x = i * h;
        a[i][0]      = bc_bottom(x);
        b[i][0]      = bc_bottom(x);
        a[i][ny + 1] = bc_top(x);
        b[i][ny + 1] = bc_top(x);
    }

    if (s == 1) {
        for (j = 1; j <= ny; j++) {
            y = j * h;
            a[0][j] = bc_left(y);
            b[0][j] = bc_left(y);
        }
        a[0][0]      = exact_u(0.0, 0.0);
        b[0][0]      = exact_u(0.0, 0.0);
        a[0][ny + 1] = exact_u(0.0, 1.0);
        b[0][ny + 1] = exact_u(0.0, 1.0);
    }

    if (e == nx) {
        for (j = 1; j <= ny; j++) {
            y = j * h;
            a[nx + 1][j] = bc_right(y);
            b[nx + 1][j] = bc_right(y);
        }
        a[nx + 1][0]      = exact_u(1.0, 0.0);
        b[nx + 1][0]      = exact_u(1.0, 0.0);
        a[nx + 1][ny + 1] = exact_u(1.0, 1.0);
        b[nx + 1][ny + 1] = exact_u(1.0, 1.0);
    }
}

void init_full_grid(double g[][maxn])
{
    int i, j;
    const double junkval = -5.0;

    for (i = 0; i < maxn; i++) {
        for (j = 0; j < maxn; j++) {
            g[i][j] = junkval;
        }
    }
}

void init_full_grids(double a[][maxn], double b[][maxn], double f[][maxn])
{
    int i, j;
    const double junkval = -5.0;

    for (i = 0; i < maxn; i++) {
        for (j = 0; j < maxn; j++) {
            a[i][j] = junkval;
            b[i][j] = junkval;
            f[i][j] = junkval;
        }
    }
}

void copy_full_grid(double a[][maxn], double b[][maxn])
{
    int i, j;
    for (i = 0; i < maxn; ++i) {
        for (j = 0; j < maxn; ++j) {
            a[i][j] = b[i][j];
        }
    }
}

void GatherGrid(double g[][maxn], double a[][maxn], int nx, int ny, MPI_Comm comm)
{
    double **subgrid;
    double *tmp;
    int myid, nprocs;
    int *gridsizes;
    int s, e;
    int localcols;
    int subgrid_index;
    MPI_Status status;
    int i, j, k;

    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &nprocs);

    if (nprocs == 1) {
        copy_full_grid(g, a);
        return;
    }

    if (myid == 0) {
        MPE_Decomp1d(nx, nprocs, myid, &s, &e);

        for (i = s - 1; i < e + 2; ++i) {
            for (j = 0; j < (ny + 2); ++j) {
                g[i][j] = a[i][j];
            }
        }

        gridsizes = (int *)calloc(nprocs, sizeof(int));

        for (i = 1; i < nprocs; ++i) {
            MPE_Decomp1d(nx, nprocs, i, &s, &e);
            gridsizes[i] = (e - s + 1) + 2;

            subgrid = (double **)malloc(gridsizes[i] * sizeof(double *));
            tmp = (double *)malloc(gridsizes[i] * (ny + 2) * sizeof(double));

            for (j = 0; j < gridsizes[i]; ++j) {
                subgrid[j] = &tmp[j * (ny + 2)];
            }

            MPI_Recv(tmp, gridsizes[i] * (ny + 2), MPI_DOUBLE, i, i, comm, &status);

            for (k = s; k < e + 2; ++k) {
                for (j = 0; j < (ny + 2); ++j) {
                    g[k][j] = subgrid[k - s + 1][j];
                }
            }

            free(subgrid);
            free(tmp);
        }

        free(gridsizes);

    } else {
        MPE_Decomp1d(nx, nprocs, myid, &s, &e);
        localcols = (e - s + 1) + 2;

        subgrid = (double **)malloc(localcols * sizeof(double *));
        tmp = (double *)malloc(localcols * (ny + 2) * sizeof(double));

        for (i = 0; i < localcols; ++i) {
            subgrid[i] = &tmp[i * (ny + 2)];
        }

        for (i = s - 1; i < e + 2; ++i) {
            subgrid_index = i - s + 1;
            for (j = 0; j < ny + 2; ++j) {
                subgrid[subgrid_index][j] = a[i][j];
            }
        }

        MPI_Send(tmp, localcols * (ny + 2), MPI_DOUBLE, 0, myid, comm);

        free(subgrid);
        free(tmp);
    }
}

void write_grid(const char *fname, double x[][maxn], int nx, int ny)
{
    FILE *fp;
    int i, j;

    if (fname == NULL) {
        for (j = ny + 1; j >= 0; j--) {
            for (i = 0; i < nx + 2; i++) {
                printf("%lf ", x[i][j]);
            }
            printf("\n");
        }
        return;
    }

    fp = fopen(fname, "w");
    if (!fp) {
        fprintf(stderr, "Error: can't open file %s\n", fname);
        exit(4);
    }

    for (j = ny + 1; j >= 0; j--) {
        for (i = 0; i < nx + 2; i++) {
            fprintf(fp, "%lf ", x[i][j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}
