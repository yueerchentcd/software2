#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <mpi.h>

#include "poisson1d.h"
#include "decomp2d.h"
#include "jacobi2d.h"

#define maxit 2000

void init_full_grid(double g[][maxn]);
void init_full_grids(double a[][maxn], double b[][maxn], double f[][maxn]);

void init2d_q2(double a[][maxn], double b[][maxn], double f[][maxn],
               int nx, int ny, int sx, int ex, int sy, int ey);

void GatherGrid2D(double g[][maxn], double a[][maxn],
                  int nx, int ny,
                  MPI_Comm cart_comm, const int dims[2],
                  int sx, int ex, int sy, int ey);

void write_grid(const char *fname, double x[][maxn], int nx, int ny);

double exact_u(double x, double y);
double bc_left(double y);
double bc_right(double y);
double bc_bottom(double x);
double bc_top(double x);

void set_global_boundaries(double g[][maxn], int nx, int ny);

int main(int argc, char **argv)
{
    double a[maxn][maxn], b[maxn][maxn], f[maxn][maxn];
    double ggrid[maxn][maxn];

    int nx, ny;
    int world_rank, world_size;
    int myid, nprocs;
    int sx, ex, sy, ey;
    int local_nx, local_ny;
    int it;

    double glob_diff, ldiff;
    double t1, t2;
    double tol = 1.0E-11;

    double local_sse, global_sse;
    double local_maxerr, global_maxerr;
    double err, x, y, h, rmse;

    int use_nonblocking;

    MPI_Comm cart_comm;
    int dims[2], periods[2], coords[2], reorder;
    int nbr_up, nbr_down, nbr_left, nbr_right;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    use_nonblocking = 0;

    if (world_rank == 0) {
        nx = 31;

        if (argc > 3) {
            fprintf(stderr, "Usage: mpirun -np <nproc> %s [nx] [sr|nb]\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (argc >= 2) {
            nx = atoi(argv[1]);
        }

        if (argc == 3) {
            if (strcmp(argv[2], "nb") == 0) {
                use_nonblocking = 1;
            } else if (strcmp(argv[2], "sr") == 0) {
                use_nonblocking = 0;
            } else {
                fprintf(stderr, "Second optional argument must be 'sr' or 'nb'\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        if (nx > maxn - 2) {
            fprintf(stderr, "grid size too large for maxn; increase maxn in poisson1d.h\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&use_nonblocking, 1, MPI_INT, 0, MPI_COMM_WORLD);
    ny = nx;

    init_full_grids(a, b, f);
    init_full_grid(ggrid);

    dims[0] = 0;
    dims[1] = 0;
    MPI_Dims_create(world_size, 2, dims);

    periods[0] = 0;
    periods[1] = 0;
    reorder = 0;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

    MPI_Comm_rank(cart_comm, &myid);
    MPI_Comm_size(cart_comm, &nprocs);
    MPI_Cart_coords(cart_comm, myid, 2, coords);

    MPI_Cart_shift(cart_comm, 0, 1, &nbr_up, &nbr_down);
    MPI_Cart_shift(cart_comm, 1, 1, &nbr_left, &nbr_right);

    MPE_Decomp2d(nx, ny, dims, coords, &sx, &ex, &sy, &ey);
    local_nx = local_size_1d(sx, ex);
    local_ny = local_size_1d(sy, ey);

    if (myid == 0) {
        printf("========================================\n");
        printf("Q4 run: 2D parallel Poisson solver\n");
        printf("Grid size            : %d x %d\n", nx, ny);
        printf("Processes used       : %d\n", nprocs);
        printf("Process grid         : %d x %d\n", dims[0], dims[1]);
        printf("Exchange mode        : %s\n", use_nonblocking ? "nonblocking" : "sendrecv");
        printf("Jacobi iterations    : %d\n", maxit);
        printf("========================================\n");
    }

    init2d_q2(a, b, f, nx, ny, sx, ex, sy, ey);

    MPI_Barrier(cart_comm);
    t1 = MPI_Wtime();

    glob_diff = 1.0e100;

    for (it = 0; it < maxit; it++) {
        if (use_nonblocking) {
            exchang2d_nonblocking(a, sx, ex, sy, ey, cart_comm,
                                  nbr_up, nbr_down, nbr_left, nbr_right);
        } else {
            exchang2d_sendrecv(a, sx, ex, sy, ey, cart_comm,
                               nbr_up, nbr_down, nbr_left, nbr_right);
        }
        sweep2d(a, f, nx, ny, sx, ex, sy, ey, b);

        if (use_nonblocking) {
            exchang2d_nonblocking(b, sx, ex, sy, ey, cart_comm,
                                  nbr_up, nbr_down, nbr_left, nbr_right);
        } else {
            exchang2d_sendrecv(b, sx, ex, sy, ey, cart_comm,
                               nbr_up, nbr_down, nbr_left, nbr_right);
        }
        sweep2d(b, f, nx, ny, sx, ex, sy, ey, a);

        ldiff = griddiff2d(a, b, sx, ex, sy, ey);
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

    for (int i = sx; i <= ex; i++) {
        x = i * h;
        for (int j = sy; j <= ey; j++) {
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

    GatherGrid2D(ggrid, a, nx, ny, cart_comm, dims, sx, ex, sy, ey);

    if (myid == 0) {
        char fname[64];

        if (use_nonblocking) {
            snprintf(fname, sizeof(fname), "q4_grid_%d_nb.dat", nx);
        } else {
            snprintf(fname, sizeof(fname), "q4_grid_%d_sr.dat", nx);
        }

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

    for (i = 0; i < maxn; i++) {
        for (j = 0; j < maxn; j++) {
            a[i][j] = 0.0;
            b[i][j] = 0.0;
            f[i][j] = 0.0;
        }
    }
}

void init2d_q2(double a[][maxn], double b[][maxn], double f[][maxn],
               int nx, int ny, int sx, int ex, int sy, int ey)
{
    int i, j;
    double h, x, y;

    h = 1.0 / ((double)(nx + 1));

    for (i = sx; i <= ex; i++) {
        for (j = sy; j <= ey; j++) {
            a[i][j] = 0.0;
            b[i][j] = 0.0;
            f[i][j] = 0.0;
        }
    }

    if (sx == 1) {
        for (j = sy; j <= ey; j++) {
            y = j * h;
            a[0][j] = bc_left(y);
            b[0][j] = bc_left(y);
        }
    }

    if (ex == nx) {
        for (j = sy; j <= ey; j++) {
            y = j * h;
            a[nx + 1][j] = bc_right(y);
            b[nx + 1][j] = bc_right(y);
        }
    }

    if (sy == 1) {
        for (i = sx; i <= ex; i++) {
            x = i * h;
            a[i][0] = bc_bottom(x);
            b[i][0] = bc_bottom(x);
        }
    }

    if (ey == ny) {
        for (i = sx; i <= ex; i++) {
            x = i * h;
            a[i][ny + 1] = bc_top(x);
            b[i][ny + 1] = bc_top(x);
        }
    }

    if (sx == 1 && sy == 1) {
        a[0][0] = exact_u(0.0, 0.0);
        b[0][0] = exact_u(0.0, 0.0);
    }

    if (sx == 1 && ey == ny) {
        a[0][ny + 1] = exact_u(0.0, 1.0);
        b[0][ny + 1] = exact_u(0.0, 1.0);
    }

    if (ex == nx && sy == 1) {
        a[nx + 1][0] = exact_u(1.0, 0.0);
        b[nx + 1][0] = exact_u(1.0, 0.0);
    }

    if (ex == nx && ey == ny) {
        a[nx + 1][ny + 1] = exact_u(1.0, 1.0);
        b[nx + 1][ny + 1] = exact_u(1.0, 1.0);
    }
}

void set_global_boundaries(double g[][maxn], int nx, int ny)
{
    int i, j;
    double h, x, y;

    h = 1.0 / ((double)(nx + 1));

    for (i = 0; i <= nx + 1; i++) {
        x = i * h;
        g[i][0] = bc_bottom(x);
        g[i][ny + 1] = bc_top(x);
    }

    for (j = 0; j <= ny + 1; j++) {
        y = j * h;
        g[0][j] = bc_left(y);
        g[nx + 1][j] = bc_right(y);
    }
}

void GatherGrid2D(double g[][maxn], double a[][maxn],
                  int nx, int ny,
                  MPI_Comm cart_comm, const int dims[2],
                  int sx, int ex, int sy, int ey)
{
    int myid, nprocs;
    int i, j, src;
    int coords_src[2];
    int rsx, rex, rsy, rey;
    int lx, ly, count, idx;
    double *buf;
    MPI_Status status;

    MPI_Comm_rank(cart_comm, &myid);
    MPI_Comm_size(cart_comm, &nprocs);

    lx = ex - sx + 1;
    ly = ey - sy + 1;
    count = lx * ly;

    if (myid == 0) {
        init_full_grid(g);
        set_global_boundaries(g, nx, ny);

        for (i = sx; i <= ex; i++) {
            for (j = sy; j <= ey; j++) {
                g[i][j] = a[i][j];
            }
        }

        for (src = 1; src < nprocs; src++) {
            MPI_Cart_coords(cart_comm, src, 2, coords_src);
            MPE_Decomp2d(nx, ny, dims, coords_src, &rsx, &rex, &rsy, &rey);

            lx = rex - rsx + 1;
            ly = rey - rsy + 1;
            count = lx * ly;

            buf = (double *)malloc(count * sizeof(double));
            if (buf == NULL) {
                fprintf(stderr, "rank 0: malloc failed in GatherGrid2D\n");
                MPI_Abort(cart_comm, 1);
            }

            MPI_Recv(buf, count, MPI_DOUBLE, src, 77, cart_comm, &status);

            idx = 0;
            for (i = rsx; i <= rex; i++) {
                for (j = rsy; j <= rey; j++) {
                    g[i][j] = buf[idx++];
                }
            }

            free(buf);
        }
    } else {
        buf = (double *)malloc(count * sizeof(double));
        if (buf == NULL) {
            fprintf(stderr, "rank %d: malloc failed in GatherGrid2D\n", myid);
            MPI_Abort(cart_comm, 1);
        }

        idx = 0;
        for (i = sx; i <= ex; i++) {
            for (j = sy; j <= ey; j++) {
                buf[idx++] = a[i][j];
            }
        }

        MPI_Send(buf, count, MPI_DOUBLE, 0, 77, cart_comm);

        free(buf);
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
