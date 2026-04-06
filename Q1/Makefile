CC = mpicc

CFLAGS = -g -Wall #-D_CC_OVERLAP

LDFLAGS = -lm

POISSOBJS = decomp1d.o jacobi.o
#POISSOBJS = decomp1d.o jacobi.o gfunc.o

EXECS = poiss1d poiss1dnb rma1poiss1d pscwpoiss1d

all: $(EXECS)

poiss1d: main.o $(POISSOBJS) 
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

poiss1dnb: mainnb.o $(POISSOBJS) 
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

rma1poiss1d: mainrma1.o decomp1d.o jacobi-rma.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

pscwpoiss1d: mainpscw.o decomp1d.o jacobi-rma.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)



# tmpedc1d: testdc1d.o decomp1d.o

## tests

tags:
	etags *.c *.h

.PHONY: clean tags tests

clean:
	$(RM) *.o $(EXECS) $(TESTS) TAGS tags
