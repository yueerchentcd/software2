#include <stdlib.h>
#include <stdio.h>

#include <mpi.h>

#include "poisson1d.h"
#include "jacobi2d.h"

/*
  Storage convention:
    x[i][j]
  with i = x-direction index, j = y-direction index.

  In C, x[i][j] is contiguous in j for fixed i.
  Therefore:
    - exchanging boundary data for fixed i and varying j is contiguous
    - exchanging boundary data for fixed j and varying i is strided

  In this code:
    - nbr_up / nbr_down correspond to decomposition in the i-direction
      (exchange contiguous columns x[sx][:] / x[ex][:])
    - nbr_left / nbr_right correspond to decomposition in the j-direction
      (exchange strided rows x[:][sy] / x[:][ey]) using MPI_Type_vector
*/

/* sendrecv version */
void exchang2d_sendrecv(double x[][maxn],
                        int sx, int ex, int sy, int ey,
                        MPI_Comm comm,
                        int nbr_up, int nbr_down,
                        int nbr_left, int nbr_right)
{
    int ny_local;
    int nx_local;
    MPI_Datatype row_type;

    ny_local = ey - sy + 1;   /* contiguous count in j-direction */
    nx_local = ex - sx + 1;   /* number of entries in strided i-direction */

    /* For fixed j and varying i, data are non-contiguous in memory */
    MPI_Type_vector(nx_local, 1, maxn, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);

    /*
      Exchange in decomposition direction associated with nbr_up/nbr_down.
      These messages are contiguous because i is fixed and j varies.
    */

    /* receive left x-ghost from nbr_up into x[sx-1][sy:ey]
       send right owned boundary x[ex][sy:ey] to nbr_down */
    MPI_Sendrecv(&x[ex][sy], ny_local, MPI_DOUBLE, nbr_down, 10,
                 &x[sx-1][sy], ny_local, MPI_DOUBLE, nbr_up,   10,
                 comm, MPI_STATUS_IGNORE);

    /* receive right x-ghost from nbr_down into x[ex+1][sy:ey]
       send left owned boundary x[sx][sy:ey] to nbr_up */
    MPI_Sendrecv(&x[sx][sy], ny_local, MPI_DOUBLE, nbr_up,   11,
                 &x[ex+1][sy], ny_local, MPI_DOUBLE, nbr_down, 11,
                 comm, MPI_STATUS_IGNORE);

    /*
      Exchange in decomposition direction associated with nbr_left/nbr_right.
      These messages are strided because j is fixed and i varies.
    */

    /* receive lower y-ghost from nbr_left into x[sx:ex][sy-1]
       send upper owned boundary x[sx:ex][ey] to nbr_right */
    MPI_Sendrecv(&x[sx][ey], 1, row_type, nbr_right, 20,
                 &x[sx][sy-1], 1, row_type, nbr_left,  20,
                 comm, MPI_STATUS_IGNORE);

    /* receive upper y-ghost from nbr_right into x[sx:ex][ey+1]
       send lower owned boundary x[sx:ex][sy] to nbr_left */
    MPI_Sendrecv(&x[sx][sy], 1, row_type, nbr_left,  21,
                 &x[sx][ey+1], 1, row_type, nbr_right, 21,
                 comm, MPI_STATUS_IGNORE);

    MPI_Type_free(&row_type);
}


/* non-blocking version */
void exchang2d_nonblocking(double x[][maxn],
                           int sx, int ex, int sy, int ey,
                           MPI_Comm comm,
                           int nbr_up, int nbr_down,
                           int nbr_left, int nbr_right)
{
    int ny_local;
    int nx_local;
    MPI_Datatype row_type;
    MPI_Request req[8];

    ny_local = ey - sy + 1;
    nx_local = ex - sx + 1;

    MPI_Type_vector(nx_local, 1, maxn, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);

    /*
      Receives:
        0: from nbr_up    into x[sx-1][sy:ey]
        1: from nbr_down  into x[ex+1][sy:ey]
        2: from nbr_left  into x[sx:ex][sy-1]
        3: from nbr_right into x[sx:ex][ey+1]
    */
    MPI_Irecv(&x[sx-1][sy], ny_local, MPI_DOUBLE, nbr_up,    10, comm, &req[0]);
    MPI_Irecv(&x[ex+1][sy], ny_local, MPI_DOUBLE, nbr_down,  11, comm, &req[1]);
    MPI_Irecv(&x[sx][sy-1], 1, row_type,         nbr_left,   20, comm, &req[2]);
    MPI_Irecv(&x[sx][ey+1], 1, row_type,         nbr_right,  21, comm, &req[3]);

    /*
      Sends:
        4: send x[ex][sy:ey]    to nbr_down
        5: send x[sx][sy:ey]    to nbr_up
        6: send x[sx:ex][ey]    to nbr_right
        7: send x[sx:ex][sy]    to nbr_left
    */
    MPI_Isend(&x[ex][sy], ny_local, MPI_DOUBLE, nbr_down,  10, comm, &req[4]);
    MPI_Isend(&x[sx][sy], ny_local, MPI_DOUBLE, nbr_up,    11, comm, &req[5]);
    MPI_Isend(&x[sx][ey], 1, row_type,         nbr_right, 20, comm, &req[6]);
    MPI_Isend(&x[sx][sy], 1, row_type,         nbr_left,  21, comm, &req[7]);

    MPI_Waitall(8, req, MPI_STATUSES_IGNORE);

    MPI_Type_free(&row_type);
}


void sweep2d(double a[][maxn], double f[][maxn],
             int nx, int ny,
             int sx, int ex, int sy, int ey,
             double b[][maxn])
{
    double h;
    int i, j;

    (void)ny;  /* square-grid coursework; h from nx is enough here */

    h = 1.0 / ((double)(nx + 1));

    for (i = sx; i <= ex; i++) {
        for (j = sy; j <= ey; j++) {
            b[i][j] = 0.25 * ( a[i-1][j] + a[i+1][j]
                             + a[i][j-1] + a[i][j+1]
                             - h * h * f[i][j] );
        }
    }
}


double griddiff2d(double a[][maxn], double b[][maxn],
                  int sx, int ex, int sy, int ey)
{
    double sum, tmp;
    int i, j;

    sum = 0.0;

    for (i = sx; i <= ex; i++) {
        for (j = sy; j <= ey; j++) {
            tmp = a[i][j] - b[i][j];
            sum += tmp * tmp;
        }
    }

    return sum;
}
