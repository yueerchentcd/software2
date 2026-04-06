CC=mpicc
CFLAGS=-O2 -Wall -Wextra -std=c11
TARGET=poisson_q2

all: $(TARGET)

$(TARGET): poisson_q2.c
	$(CC) $(CFLAGS) -o $(TARGET) poisson_q2.c -lm

clean:
	rm -f $(TARGET)
