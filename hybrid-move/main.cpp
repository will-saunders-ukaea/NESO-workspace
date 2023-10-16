#include <iostream>
#include <mpi.h>
#include <neso_particles.hpp>
#include <chrono>
#include <csignal>
#include <string>

using namespace cl;
using namespace NESO::Particles;

inline void hybrid_move_driver(const int N_total, 
  const int Nsteps_warmup = 1024,
  const int Nsteps = 2048
){

  const int ndim = 2;
  std::vector<int> dims(ndim);
  dims[0] = 16;
  dims[1] = 16;

  const double cell_extent = 1.0;
  const int subdivision_order = 1;
  const int stencil_width = 1;
  auto mesh = std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims, cell_extent,
                      subdivision_order, stencil_width);

  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);

  auto domain = std::make_shared<Domain>(mesh, cart_local_mapper);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("P_ORIG"), ndim),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  A->add_particle_dat(ParticleDat(sycl_target, ParticleProp(Sym<REAL>("FOO"), 3),
                                 domain->mesh->get_cell_count()));

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;

  std::mt19937 rng_pos(52234234 + rank);
  std::mt19937 rng_vel(52234231 + rank);
  std::mt19937 rng_rank(18241);
  
  
  int rstart, rend;
  get_decomp_1d(size, N_total, rank, &rstart, &rend);
  const int N = rend - rstart;

  int N_check = -1;
  MPICHK(MPI_Allreduce(&N, &N_check, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  NESOASSERT(N_check == N_total, "Error creating particles");


  const REAL dt = 0.001;
  const int cell_count = domain->mesh->get_cell_count();

  if (rank == 0){
    nprint("Num Particles:", N_total);
    nprint("Num Warm-up Steps:", Nsteps_warmup);
    nprint("Num Steps:", Nsteps);
  }
  
  if (N > 0){
  auto positions =
      uniform_within_extents(N, ndim, mesh->global_extents, rng_pos);
  auto velocities = NESO::Particles::normal_distribution(
      N, 3, 0.0, 0.5, rng_vel);
  std::uniform_int_distribution<int> uniform_dist(
      0, size - 1);
  ParticleSet initial_distribution(N, A->get_particle_spec());
  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] = positions[dimx][px];
      initial_distribution[Sym<REAL>("P_ORIG")][px][dimx] = positions[dimx][px];
    }
    for (int dimx = 0; dimx < 3; dimx++) {
      initial_distribution[Sym<REAL>("V")][px][dimx] = velocities[dimx][px];
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = px % cell_count;
    initial_distribution[Sym<INT>("ID")][px][0] = px;
    const auto px_rank = uniform_dist(rng_rank);
    initial_distribution[Sym<INT>("NESO_MPI_RANK")][px][0] = px_rank;
  }
  A->add_particles_local(initial_distribution);
  }
  parallel_advection_initialisation(A, 64);


  if (rank == 0){
    std::cout << "Particles Added..." << std::endl;
  }

  auto pbc = std::make_shared<CartesianPeriodic>(sycl_target, mesh, A->position_dat);
  auto ccb = std::make_shared<CartesianCellBin>(sycl_target, mesh, A->position_dat, A->cell_id_dat);

  auto lambda_advect = [&] {

    auto t0 = profile_timestamp();

    auto k_P = A->get_dat(Sym<REAL>("P"))->cell_dat.device_ptr();
    const auto k_V = A->get_dat(Sym<REAL>("V"))->cell_dat.device_ptr();
    const auto k_ndim = ndim;
    const auto k_dt = dt;

    const auto pl_iter_range = A->mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride = A->mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell = A->mpi_rank_dat->get_particle_loop_npart_cell();

    sycl_target->profile_map.inc("Advect", "Prepare", 1, profile_elapsed(t0, profile_timestamp()));
    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;
                for (int dimx = 0; dimx < k_ndim; dimx++) {
                  k_P[cellx][dimx][layerx] += k_V[cellx][dimx][layerx] * k_dt;
                }
                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();
    sycl_target->profile_map.inc("Advect", "Execute", 1, profile_elapsed(t0, profile_timestamp()));
  };

  REAL T = 0.0;
 
  pbc->execute();
  A->hybrid_move();
  ccb->execute();
  A->cell_move();  

  MPI_Barrier(sycl_target->comm_pair.comm_parent);
  if (rank == 0){
    std::cout << N_total << " Particles Distributed..." << std::endl;
  }

  H5Part h5part("traj.h5part", A, Sym<REAL>("P"), Sym<INT>("CELL_ID"));

  for (int stepx = 0; stepx < Nsteps_warmup; stepx++) {

    pbc->execute();

    A->hybrid_move();

    ccb->execute();
    A->cell_move();
    h5part.write();

    lambda_advect();

    T += dt;
    
    if( (stepx % 100 == 0) && (rank == 0)) {
      std::cout << stepx << std::endl;
    }
  }
  h5part.close();
  sycl_target->profile_map.reset();

  std::chrono::high_resolution_clock::time_point time_start = std::chrono::high_resolution_clock::now();

  for (int stepx = 0; stepx < Nsteps; stepx++) {


    pbc->execute();

    A->hybrid_move();

    ccb->execute();
    A->cell_move();

    lambda_advect();

    T += dt;   
    if( (stepx % 100 == 0) && (rank == 0)) {
      std::cout << stepx << std::endl;
    }
  }

  std::chrono::high_resolution_clock::time_point time_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_taken = time_end - time_start;
  const double time_taken_double = (double) time_taken.count();
  
  if (rank == 0){
    std::cout << "TIME TAKEN: " << time_taken_double << " PER STEP: " << time_taken_double / Nsteps << std::endl;
  }


  mesh->free();
  
  if (rank == 0){
    sycl_target->profile_map.print();
  }

}

int main(int argc, char **argv) {

  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cout << "ERROR: MPI_Init != MPI_SUCCESS" << std::endl;
    return -1;
  }
 
  if (argc > 3){

    std::vector<int> argsi(3);
    for(int ix=0 ; ix<3 ; ix++){
      std::string argv0 = std::string(argv[ix+1]);
      argsi.at(ix) = std::stoi(argv0);
    }
    hybrid_move_driver(argsi.at(0), argsi.at(1), argsi.at(2));
  }


  if (MPI_Finalize() != MPI_SUCCESS) {
    std::cout << "ERROR: MPI_Finalize != MPI_SUCCESS" << std::endl;
    return -1;
  }

  return 0;
}
