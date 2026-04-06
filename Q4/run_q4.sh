#!/bin/bash

echo "===== Q4 runs start ====="

echo "Running: np=4, nx=15, sendrecv"
mpirun -np 4 ./poiss1d 15 sr | tee q4_np4_n15_sr.txt

echo "Running: np=4, nx=15, nonblocking"
mpirun -np 4 ./poiss1d 15 nb | tee q4_np4_n15_nb.txt

echo "Running: np=4, nx=31, sendrecv"
mpirun -np 4 ./poiss1d 31 sr | tee q4_np4_n31_sr.txt

echo "Running: np=4, nx=31, nonblocking"
mpirun -np 4 ./poiss1d 31 nb | tee q4_np4_n31_nb.txt

echo "Running: np=16, nx=31, sendrecv"
mpirun -np 16 ./poiss1d 31 sr | tee q4_np16_n31_sr.txt

echo "Running: np=16, nx=31, nonblocking"
mpirun -np 16 ./poiss1d 31 nb | tee q4_np16_n31_nb.txt

echo "===== Q4 runs finished ====="
