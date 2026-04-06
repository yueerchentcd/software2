void sweep1d(double a[][maxn], double f[][maxn], int nx,
	     int s, int e, double b[][maxn]);

void exchang1(double x[][maxn], int nx, int s, int e, MPI_Comm comm,
	      int nbrbottom, int nbrtop);

void exchang2(double x[][maxn], int nx, int s, int e, MPI_Comm comm,
	      int nbrleft, int nbrright);

void exchang3(double x[][maxn], int nx, int s, int e, MPI_Comm comm,
	      int nbrleft, int nbrright);

void exchangi1(double x[][maxn], int nx, int s, int e, MPI_Comm comm,
	       int nbrleft, int nbrright);

void nbxchange_and_sweep(double u[][maxn], double f[][maxn], int nx, int ny,
			 int s, int e, double unew[][maxn], MPI_Comm comm,
			 int nbrleft, int nbrright);

double griddiff(double a[][maxn], double b[][maxn], int nx, int s, int e);

void exchangrma1(double x[][maxn], int nx, int s, int e, MPI_Win win,
		 int nbrleft, int nbrright, MPI_Aint right_ghost_disp);

void exchangrma2(double x[][maxn], int nx, int s, int e, MPI_Win win,
		 int nbrleft, int nbrright);

void exchangpscw1(double x[][maxn], int nx, int s, int e, MPI_Win win,
		  int nbrleft, int nbrright, MPI_Aint right_ghost_disp,
		  MPI_Group nbr_group);

void exchangpscw2(double x[][maxn], int nx, int s, int e, MPI_Win win,
		  int nbrleft, int nbrright, MPI_Group nbr_group);
