#ifndef JACOBI2D_H
#define JACOBI2D_H

#include <mpi.h>
#include "poisson1d.h"

void exchang2d_sendrecv(double x[][maxn],
                        int sx, int ex, int sy, int ey,
                        MPI_Comm comm,
                        int nbr_up, int nbr_down,
                        int nbr_left, int nbr_right);

void exchang2d_nonblocking(double x[][maxn],
                           int sx, int ex, int sy, int ey,
                           MPI_Comm comm,
                           int nbr_up, int nbr_down,
                           int nbr_left, int nbr_right);

void sweep2d(double a[][maxn], double f[][maxn],
             int nx, int ny,
             int sx, int ex, int sy, int ey,
             double b[][maxn]);

double griddiff2d(double a[][maxn], double b[][maxn],
                  int sx, int ex, int sy, int ey);

#endif
