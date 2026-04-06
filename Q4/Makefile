CC=mpicc
CFLAGS=-g -Wall
TARGET=poiss1d

all: $(TARGET)

$(TARGET): main.o decomp1d.o decomp2d.o jacobi2d.o
	$(CC) $(CFLAGS) -o $(TARGET) main.o decomp1d.o decomp2d.o jacobi2d.o -lm

main.o: main.c poisson1d.h decomp2d.h jacobi2d.h
	$(CC) $(CFLAGS) -c main.c

decomp1d.o: decomp1d.c decomp1d.h
	$(CC) $(CFLAGS) -c decomp1d.c

decomp2d.o: decomp2d.c decomp2d.h decomp1d.h
	$(CC) $(CFLAGS) -c decomp2d.c

jacobi2d.o: jacobi2d.c jacobi2d.h poisson1d.h
	$(CC) $(CFLAGS) -c jacobi2d.c

clean:
	rm -f *.o $(TARGET)
