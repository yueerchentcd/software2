#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

static double exact_u(double x, double y) {
    return y / (((1.0 + x) * (1.0 + x)) + y * y);
}

static double bc_left(double y) {
    return y / (1.0 + y * y);
}

static double bc_right(double y) {
    return y / (4.0 + y * y);
}

static double bc_top(double x) {
    return 1.0 / (((1.0 + x) * (1.0 + x)) + 1.0);
}

static int local_nx_1d(int n, int size, int rank) {
    int base = n / size;
    int rem  = n % size;
    return base + (rank < rem ? 1 : 0);
}

static int global_x_start_1d(int n, int size, int rank) {
    int base = n / size;
    int rem  = n % size;
    return 1 + rank * base + (rank < rem ? rank : rem);
}

static double **alloc_2d(int nx, int ny) {
    double *data = (double *)calloc((size_t)nx * ny, sizeof(double));
    double **a = (double **)malloc((size_t)nx * sizeof(double *));
    if (!data || !a) {
        fprintf(stderr, "Allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < nx; i++) {
        a[i] = data + (size_t)i * ny;
    }
    return a;
}

static void free_2d(double **a) {
    if (a) {
        free(a[0]);
        free(a);
    }
}

static void run_case(int n, int max_iters, MPI_Comm world) {
    int rank, size;
    MPI_Comm cart_comm;
    int dims[1], periods[1], reorder;
    int nbrleft, nbrright;

    MPI_Comm_rank(world, &rank);
    MPI_Comm_size(world, &size);

    if (size > n) {
        if (rank == 0) {
            fprintf(stderr, "For this code, use number of processes <= grid size n.\n");
        }
        MPI_Abort(world, 1);
    }

    dims[0] = size;
    periods[0] = 0;
    reorder = 0;

    MPI_Cart_create(world, 1, dims, periods, reorder, &cart_comm);
    MPI_Cart_shift(cart_comm, 0, 1, &nbrleft, &nbrright);

    int local_nx = local_nx_1d(n, size, rank);
    int gx_start = global_x_start_1d(n, size, rank);

    int nx = local_nx + 2;   /* left ghost + interior + right ghost */
    int ny = n + 2;          /* bottom boundary + interior + top boundary */

    double h = 1.0 / (n + 1);

    double **u     = alloc_2d(nx, ny);
    double **u_new = alloc_2d(nx, ny);

    /* Set physical left/right boundaries if this process owns them */
    if (rank == 0) {
        for (int j = 0; j < ny; j++) {
            double y = j * h;
            u[0][j] = bc_left(y);
            u_new[0][j] = u[0][j];
        }
    }

    if (rank == size - 1) {
        for (int j = 0; j < ny; j++) {
            double y = j * h;
            u[local_nx + 1][j] = bc_right(y);
            u_new[local_nx + 1][j] = u[local_nx + 1][j];
        }
    }

    /* Set bottom/top boundaries for interior columns */
    for (int i = 1; i <= local_nx; i++) {
        int gi = gx_start + i - 1;   /* global interior x-index: 1..n */
        double x = gi * h;

        u[i][0] = 0.0;
        u_new[i][0] = 0.0;

        u[i][ny - 1] = bc_top(x);
        u_new[i][ny - 1] = u[i][ny - 1];
    }

    /* Also keep boundary rows consistent on physical ghost columns */
    if (rank == 0) {
        u[0][0] = 0.0;
        u_new[0][0] = 0.0;
        u[0][ny - 1] = bc_top(0.0);
        u_new[0][ny - 1] = u[0][ny - 1];
    }

    if (rank == size - 1) {
        u[local_nx + 1][0] = 0.0;
        u_new[local_nx + 1][0] = 0.0;
        u[local_nx + 1][ny - 1] = bc_top(1.0);
        u_new[local_nx + 1][ny - 1] = u[local_nx + 1][ny - 1];
    }

    double global_change = 0.0;

    for (int iter = 0; iter < max_iters; iter++) {
        /* Exchange halo columns: send only interior y values j=1..n */
        MPI_Sendrecv(&u[local_nx][1], n, MPI_DOUBLE, nbrright, 0,
                     &u[0][1],        n, MPI_DOUBLE, nbrleft,  0,
                     cart_comm, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&u[1][1],            n, MPI_DOUBLE, nbrleft,  1,
                     &u[local_nx + 1][1], n, MPI_DOUBLE, nbrright, 1,
                     cart_comm, MPI_STATUS_IGNORE);

        double local_change = 0.0;

        for (int i = 1; i <= local_nx; i++) {
            for (int j = 1; j <= n; j++) {
                /* f = 0, so Jacobi update is pure average of 4 neighbours */
                u_new[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] +
                                      u[i][j - 1] + u[i][j + 1]);

                double diff = fabs(u_new[i][j] - u[i][j]);
                if (diff > local_change) local_change = diff;
            }
        }

        MPI_Allreduce(&local_change, &global_change, 1, MPI_DOUBLE, MPI_MAX, cart_comm);

        double **tmp = u;
        u = u_new;
        u_new = tmp;
    }

    /* Compare numerical solution to analytic solution */
    double local_max_err = 0.0;
    double local_sse = 0.0;

    for (int i = 1; i <= local_nx; i++) {
        int gi = gx_start + i - 1;
        double x = gi * h;

        for (int j = 1; j <= n; j++) {
            double y = j * h;
            double uex = exact_u(x, y);
            double err = fabs(u[i][j] - uex);

            if (err > local_max_err) local_max_err = err;
            local_sse += err * err;
        }
    }

    double global_max_err = 0.0;
    double global_sse = 0.0;

    MPI_Reduce(&local_max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    MPI_Reduce(&local_sse, &global_sse, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);

    if (rank == 0) {
        double rmse = sqrt(global_sse / (double)(n * n));
        printf("========================================\n");
        printf("Q2 result for grid size n = %d\n", n);
        printf("Processes used        : %d\n", size);
        printf("Jacobi iterations     : %d\n", max_iters);
        printf("Final max update      : %.12e\n", global_change);
        printf("Max abs error         : %.12e\n", global_max_err);
        printf("RMSE                  : %.12e\n", rmse);
        printf("========================================\n");
    }

    free_2d(u);
    free_2d(u_new);
    MPI_Comm_free(&cart_comm);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 4 && rank == 0) {
        fprintf(stderr, "Warning: Q2 asks for at least 4 processors.\n");
    }

    int max_iters = 2000;

    if (argc == 1) {
        run_case(15, max_iters, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        run_case(31, max_iters, MPI_COMM_WORLD);
    } else if (argc == 2) {
        int n = atoi(argv[1]);
        run_case(n, max_iters, MPI_COMM_WORLD);
    } else {
        int n = atoi(argv[1]);
        int iters = atoi(argv[2]);
        run_case(n, iters, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
