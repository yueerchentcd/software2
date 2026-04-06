# Q4: 2D MPI Poisson Solver

## Build
make

## Run examples
mpirun -np 4  ./poiss1d 31 sr
mpirun -np 4  ./poiss1d 31 nb
mpirun -np 16 ./poiss1d 31 sr
mpirun -np 16 ./poiss1d 31 nb

## Automation
./run_q4.sh

## Features
- 2D domain decomposition
- MPI Cartesian topology
- sendrecv + nonblocking ghost exchange
- MPI_Type_vector for strided communication
- GatherGrid2D()

## Results
See Q4_output.txt
