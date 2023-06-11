#! /usr/bin/bash -l

#SBATCH --ntasks 32 # The number of cores you need...
#SBATCH -J ohai
#SBATCH -o %J.stdout.log
#SBATCH -e %J.stderr.log
#SBATCH -p bluefield1
#SBATCH -A do009
#SBATCH --exclusive
#SBATCH -t 00:10:00

module purge
module load cosma/2018
source ~/source_gnu_neso.sh
module list -l

cd /cosma5/data/do009/dc-saun1/git/NESO-workspace/performance_workshop
OMP_NUM_THREADS=1 OMP_DISPLAY_AFFINITY=true OMP_PLACES=cores maqao lprof --mpi-command="/cosma/local/openmpi/gnu_11.1.0/4.1.4/bin/mpirun --report-bindings --map-by numa:PE=1 -n 32" -- /cosma5/data/do009/dc-saun1/git/NESO3D/solvers/Electrostatic2D3V/Electrostatic2D3V conditions.xml square_16x16.xml
