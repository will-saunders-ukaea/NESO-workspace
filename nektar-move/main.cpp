#include "nektar_interface/particle_interface.hpp"
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <SolverUtils/Driver.h>
#include <array>
#include <cmath>
#include <deque>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <vector>

using namespace std;
using namespace Nektar;
using namespace Nektar::SolverUtils;
using namespace Nektar::SpatialDomains;
using namespace NESO::Particles;

inline void hybrid_move_driver(const int N_total, char * mesh_filename) {

  int argc = 2;
  char *argv[2] = {"test_particle_geometry_interface", 
    mesh_filename};

  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;
  // Create session reader.
  session = LibUtilities::SessionReader::CreateInstance(argc, argv);
  graph = SpatialDomains::MeshGraph::Read(session);

  ParticleMeshInterface mesh(graph);
  SYCLTarget sycl_target{0, mesh.get_comm()};

  auto nektar_graph_local_mapper =
      std::make_shared<NektarGraphLocalMapperT>(sycl_target, mesh, 1.0e-10);
  Domain domain(mesh, nektar_graph_local_mapper);

  const int ndim = 2;
  const double extent[2] = {1.0, 1.0};
  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("P_ORIG"), ndim),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  ParticleGroup A(domain, particle_spec, sycl_target);

  NektarCartesianPeriodic pbc(sycl_target, graph, A.position_dat);

  CellIDTranslation cell_id_translation(sycl_target, A.cell_id_dat, mesh);

  const int rank = sycl_target.comm_pair.rank_parent;
  const int size = sycl_target.comm_pair.size_parent;

  std::mt19937 rng_pos(52234234 + rank);
  std::mt19937 rng_vel(52234231 + rank);
  std::mt19937 rng_rank(18241);

  int rstart, rend;
  get_decomp_1d(size, N_total, rank, &rstart, &rend);
  const int N = rend - rstart;

  int N_check = -1;
  MPICHK(MPI_Allreduce(&N, &N_check, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  NESOASSERT(N_check == N_total, "Error creating particles");

  const int Nsteps_warmup = 200;
  const int Nsteps = 2000;
  const REAL dt = 0.01;
  const int cell_count = domain.mesh.get_cell_count();

  if (rank == 0) {
    std::cout << "Starting..." << std::endl;
  }



  if (N > 0) {
    auto positions = uniform_within_extents(N, ndim, pbc.global_extent, rng_pos);
    auto velocities =
        NESO::Particles::normal_distribution(N, 3, 0.0, 0.5, rng_vel);
    std::uniform_int_distribution<int> uniform_dist(
        0, sycl_target.comm_pair.size_parent - 1);
    ParticleSet initial_distribution(N, A.get_particle_spec());
    for (int px = 0; px < N; px++) {
      for (int dimx = 0; dimx < ndim; dimx++) {
        const double pos_orig = positions[dimx][px] + pbc.global_origin[dimx];
        initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig;
        initial_distribution[Sym<REAL>("P_ORIG")][px][dimx] = pos_orig;
      }
      for (int dimx = 0; dimx < 3; dimx++) {
        initial_distribution[Sym<REAL>("V")][px][dimx] = velocities[dimx][px];
      }
      initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
      initial_distribution[Sym<INT>("ID")][px][0] = px;
      const auto px_rank = uniform_dist(rng_rank);
      initial_distribution[Sym<INT>("NESO_MPI_RANK")][px][0] = px_rank;
    }
    A.add_particles_local(initial_distribution);
  }
  reset_mpi_ranks(A[Sym<INT>("NESO_MPI_RANK")]);

  if (rank == 0) {
    std::cout << "Particles Added..." << std::endl;
  }

  MeshHierarchyGlobalMap mesh_heirarchy_global_map(
      sycl_target, domain.mesh, A.position_dat, A.cell_id_dat, A.mpi_rank_dat);

  auto lambda_advect = [&] {
    auto t0 = profile_timestamp();

    auto k_P = A[Sym<REAL>("P")]->cell_dat.device_ptr();
    const auto k_V = A[Sym<REAL>("V")]->cell_dat.device_ptr();
    const auto k_ndim = ndim;
    const auto k_dt = dt;

    const auto pl_iter_range = A.mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride = A.mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell = A.mpi_rank_dat->get_particle_loop_npart_cell();

    sycl_target.profile_map.inc("Advect", "Prepare", 1,
                                profile_elapsed(t0, profile_timestamp()));
    sycl_target.queue
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
    sycl_target.profile_map.inc("Advect", "Execute", 1,
                                profile_elapsed(t0, profile_timestamp()));
  };

  REAL T = 0.0;

  pbc.execute();
  mesh_heirarchy_global_map.execute();
  A.hybrid_move();
  cell_id_translation.execute();
  A.cell_move();

  MPI_Barrier(sycl_target.comm_pair.comm_parent);
  if (rank == 0) {
    std::cout << N_total << " Particles Distributed..." << std::endl;

    std::cout << "TriCount: " << graph->GetAllTriGeoms().size() << std::endl;;
    std::cout << "QuadCount: " << graph->GetAllQuadGeoms().size() << std::endl;;
  }


  H5Part h5_part("trajectory.h5part",
      A,
      Sym<REAL>("P"),
      Sym<INT>("CELL_ID"),
      Sym<INT>("NESO_MPI_RANK")
  );
  for (int stepx = 0; stepx < Nsteps_warmup; stepx++) {

    pbc.execute();
    mesh_heirarchy_global_map.execute();
    A.hybrid_move();
    cell_id_translation.execute();
    A.cell_move();
    
    if(stepx % 2 == 0){
     h5_part.write();
    }

    lambda_advect();
    
    T += dt;

    if ((stepx % 100 == 0) && (rank == 0)) {
      std::cout << stepx << std::endl;
    }
  }
  h5_part.close();

  if (rank == 0) {
    sycl_target.profile_map.print();
  }
  sycl_target.profile_map.reset();

  std::chrono::high_resolution_clock::time_point time_start =
      std::chrono::high_resolution_clock::now();

  for (int stepx = 0; stepx < Nsteps; stepx++) {

    pbc.execute();
    mesh_heirarchy_global_map.execute();
    A.hybrid_move();
    cell_id_translation.execute();
    A.cell_move();

    lambda_advect();

    T += dt;
    if ((stepx % 100 == 0) && (rank == 0)) {
      std::cout << stepx << std::endl;
    }
  }

  std::chrono::high_resolution_clock::time_point time_end =
      std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_taken = time_end - time_start;
  const double time_taken_double = (double)time_taken.count();

  if (rank == 0) {
    std::cout << "TIME TAKEN: " << time_taken_double
              << " PER STEP: " << time_taken_double / Nsteps << std::endl;
  }

  mesh.free();

  if (rank == 0) {
    sycl_target.profile_map.print();
  }
}

int main(int argc, char **argv) {

  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cout << "ERROR: MPI_Init != MPI_SUCCESS" << std::endl;
    return -1;
  }

  if (argc > 2) {

    std::string argv0 = std::string(argv[1]);
    const int N = std::stoi(argv0);
    char * mesh_filename = argv[2];

    hybrid_move_driver(N, mesh_filename);
  }

  if (MPI_Finalize() != MPI_SUCCESS) {
    std::cout << "ERROR: MPI_Finalize != MPI_SUCCESS" << std::endl;
    return -1;
  }

  return 0;
}
