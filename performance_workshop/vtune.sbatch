#! /usr/bin/bash -l
#SBATCH --ntasks=128      # Total number of MPI processes
#SBATCH --ntasks-per-node=16   # MPI processes per node
#SBATCH --cpus-per-task=2    # OMP threads per process
#SBATCH -J ohai
#SBATCH -o %J.stdout.log
#SBATCH -e %J.stderr.log
#SBATCH -p bluefield1
#SBATCH -A do009
#SBATCH --exclusive
#SBATCH -t 00:20:00

module purge
module load cosma/2018
source ~/source_gnu_neso.sh

module load intel_comp/2022.3.0
module load vtune/2022.4.1

module list -l

## I_MPI_PIN_CELL=core mpirun -n 8 python3 -c "import psutil; print(psutil.Process().cpu_affinity())"

cd /cosma5/data/do009/dc-saun1/git/NESO-workspace/performance_workshop
OMP_NUM_THREADS=2 mpirun -x OMP_DISPLAY_AFFINITY=true --report-bindings -x OMP_PLACES=cores --map-by numa:PE=2 -n 128  vtune -collect hotspots -result-dir foo8 /cosma5/data/do009/dc-saun1/git/NESO3D/solvers/Electrostatic2D3V/Electrostatic2D3V conditions.xml square_32x32.xml
