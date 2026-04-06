nano decomp2d.c
#ifndef DECOMP2D_H
#define DECOMP2D_H

void MPE_Decomp2d(int nx, int ny,
                  const int dims[2], const int coords[2],
                  int *sx, int *ex, int *sy, int *ey);

int local_size_1d(int s, int e);

#endif
