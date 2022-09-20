#include <iostream>
#include <mpi.h>
#include <neso_particles.hpp>
#include <chrono>
#include <csignal>
#include <string>

using namespace cl;
using namespace NESO::Particles;

inline void duplicated_domain(const int N_total, const int N_steps){
  
  // Cell count in the actual domain particles live in.
  const int cell_count = 20*20;

  // time step size
  const REAL dt = 0.001;
  
  // Dimension of the space particles move in for NESO Particles
  const int ndim_space = 2;
  // Dimension of the space particles move in in phyiscal space
  const int ndim_particles = 3;
  
  // For NESO-Particles create a 2D domain with diagonally opposite corners [-1, -1] and [1,1].
  std::vector<double> extents(ndim_particles);
  extents[0] = 2;
  extents[1] = 2;
  extents[2] = 2;
  std::vector<double> origin(ndim_particles);
  origin[0] = -1.0;
  origin[1] = -1.0;
  origin[2] = 0.0;
  
  // On each MPI rank use MPI_COMM_SELF to have a communicator per simulation.
  // On each rank (aka communicator) create a 2D domain as described above such
  // that there are N copies of the domain in total where N is the number of
  // MPI ranks.
  LocalDecompositionHMesh mesh(ndim_space, origin, extents, cell_count, MPI_COMM_SELF);

  // Create a NESO compute device on each rank
  // The get_local_mpi call finds a local MPI rank to use for assigning GPUs to
  // MPI ranks.
  SYCLTarget sycl_target{0, mesh.get_comm(), get_local_mpi_rank(MPI_COMM_WORLD, 0)};
  
  // create a domain from the 2D rectangle on each rank. Note this is not
  // spatially decomposed.
  Domain domain(mesh);
  
  // Properties we want each particle to have
  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim_particles, true),
                             ParticleProp(Sym<REAL>("V"), ndim_particles),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};
  
  // Create a group of particles with the properties above.
  ParticleGroup A(domain, particle_spec, sycl_target);
  
  // Create a new ensemble of particles on each rank such that the simulations
  // on each rank are independent (needs better seeding)
  int rank;
  MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  std::mt19937 rng_pos(52234234 + rank);
  std::mt19937 rng_vel(52234231 + rank);
  
  MPICHK(MPI_Barrier(MPI_COMM_WORLD));
  if (rank == 0){
    std::cout << "Starting..." << std::endl;
  }
  
  // Compute a set of initial positions and velocities
  auto positions =
      uniform_within_extents(N_total, ndim_particles, mesh.global_extents.data(), rng_pos);
  auto velocities = NESO::Particles::normal_distribution(
      N_total, 3, 0.0, 0.5, rng_vel);

  // Allocate space to store the initial particle configuration
  ParticleSet initial_distribution(N_total, A.get_particle_spec());
  
  // populate the initial particle configuration
  for (int px = 0; px < N_total; px++) {
    for (int dimx = 0; dimx < ndim_particles; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] = positions[dimx][px] + origin[dimx];
      initial_distribution[Sym<REAL>("V")][px][dimx] = velocities[dimx][px];
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
    initial_distribution[Sym<INT>("ID")][px][0] = px;
  }
  
  // Add the initial configuration to the ParticleGroup A. This is where the
  // particles are actually added to A
  A.add_particles_local(initial_distribution);

  MPICHK(MPI_Barrier(MPI_COMM_WORLD));
  if (rank == 0){
    std::cout << "Particles Added..." << std::endl;
  }

  // Create a method to move the particles.
  auto lambda_advect = [&] {

    auto t0 = profile_timestamp();
    
    // these calls get the pointers to access the particle data in the kernel
    auto k_P = A[Sym<REAL>("P")]->cell_dat.device_ptr();
    const auto k_V = A[Sym<REAL>("V")]->cell_dat.device_ptr();
    
    // explictly get copies/references to variables we want to access in the
    // kernel
    const auto k_ndim = ndim_particles;
    const auto k_dt = dt;
    
    // get the variables required to loop over cells and particles in cells.
    const auto pl_iter_range = A.mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride = A.mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell = A.mpi_rank_dat->get_particle_loop_npart_cell();
    
    // NESO-Particles profiling
    sycl_target.profile_map.inc("Advect", "Prepare", 1, 
        profile_elapsed(t0, profile_timestamp()));
    
    // actually create and queue the advection loop/kernel
    sycl_target.queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                
                // NESO-Particles has some basic loop reordering depending if
                // -DNESO_PARTICLES_DEVICE_TYPE=GPU or
                // -DNESO_PARTICLES_DEVICE_TYPE=CPU was passed to cmake.
                NESO_PARTICLES_KERNEL_START
                
                // determine which cell and particle is being accessed. The
                // term "layer" is a referrence to how Rapaport stores
                // particles in "layers" in GPU MD simulations.
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

                // the operation we want to perform on each particle
                for (int dimx = 0; dimx < k_ndim; dimx++) {
                  k_P[cellx][dimx][layerx] += k_V[cellx][dimx][layerx] * k_dt;
                }

                // See NESO_PARTICLES_KERNEL_START comment
                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();

    // NESO-Particles profiling
    sycl_target.profile_map.inc("Advect", "Execute", 1, 
        profile_elapsed(t0, profile_timestamp()));
  };

  MPI_Barrier(sycl_target.comm_pair.comm_parent);
  if (rank == 0){
    std::cout << N_total << " Particles Distributed..." << std::endl;
  }
  
  // Create a class that writes particle data to file. Note that if cmake does
  // not find HDF5 this and the write and free calls become noops
  H5Part h5_part("trajectory_" + std::to_string(rank) + ".h5part",
      A,
      // Select which properties to write
      Sym<REAL>("P"),
      Sym<INT>("ID")
  );

  REAL T = 0.0;
  for (int stepx = 0; stepx < N_steps; stepx++) {
    
    // advect the particles forward in time
    lambda_advect();
  
    T += dt;   
    if( (stepx % 100 == 0) && (rank == 0)) {
      std::cout << stepx << std::endl;
    }

    // update the trajectory in the file
    if(stepx % 100 == 0) {
      h5_part.write();
    }   
  }
  
  // close the HDF5 file and free the mesh
  h5_part.close();
  mesh.free();
  
  // print some internal profiling information
  if (rank == 0){
    sycl_target.profile_map.print();
  }

}

int main(int argc, char **argv) {

  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cout << "ERROR: MPI_Init != MPI_SUCCESS" << std::endl;
    return -1;
  }
 
  if (argc > 2){
    
    std::string argv1 = std::string(argv[1]);
    const int N_particles = std::stoi(argv1);
    
    std::string argv2 = std::string(argv[2]);
    const int N_steps = std::stoi(argv2);

    duplicated_domain(N_particles, N_steps);

  } else {
    std::cout << "Expected number of particles and number of steps to be passed on command line." << std::endl;
  }

  if (MPI_Finalize() != MPI_SUCCESS) {
    std::cout << "ERROR: MPI_Finalize != MPI_SUCCESS" << std::endl;
    return -1;
  }

  return 0;
}
