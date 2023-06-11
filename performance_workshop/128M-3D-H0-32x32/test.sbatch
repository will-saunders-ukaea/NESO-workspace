#! /usr/bin/bash -l

#SBATCH --ntasks 32 # The number of cores you need...
#SBATCH -J ohai
#SBATCH -o %J.stdout.log
#SBATCH -e %J.stderr.log
#SBATCH -p bluefield1
#SBATCH -A do009
#SBATCH --exclusive
#SBATCH -t 00:40:00

module purge
module load cosma/2018
source ~/source_gnu_neso.sh
module list -l

## I_MPI_PIN_CELL=core mpirun -n 8 python3 -c "import psutil; print(psutil.Process().cpu_affinity())"

cd /cosma5/data/do009/dc-saun1/git/NESO-workspace/performance_workshop/128M-3D-H0-32x32
export EXEC=/cosma5/data/do009/dc-saun1/git/NESO/solvers/Electrostatic2D3V/Electrostatic2D3V-H0
OMP_NUM_THREADS=1 mpirun -x OMP_DISPLAY_AFFINITY=true --report-bindings -x OMP_PLACES=cores --map-by numa:PE=1 -n 32 /cosma5/data/do009/dc-saun1/git/NESO3D/solvers/Electrostatic2D3V/Electrostatic2D3V ../conditions_128M.xml ../square_32x32.xml
