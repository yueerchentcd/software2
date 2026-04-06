#include <stdlib.h>
#include <stdio.h>

#include "decomp1d.h"
#include "decomp2d.h"

void MPE_Decomp2d(int nx, int ny,
                  const int dims[2], const int coords[2],
                  int *sx, int *ex, int *sy, int *ey)
{
    /* x-direction decomposition */
    MPE_Decomp1d(nx, dims[0], coords[0], sx, ex);

    /* y-direction decomposition */
    MPE_Decomp1d(ny, dims[1], coords[1], sy, ey);
}

int local_size_1d(int s, int e)
{
    return e - s + 1;
}
