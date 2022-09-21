#include "nektar_interface/particle_interface.hpp"
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/DisContField.h>
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
using namespace Nektar::MultiRegions;
using namespace NESO::Particles;


inline void hybrid_move_driver(const int N_total, int argc, char ** argv) {

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

  pbc.execute();
  mesh_heirarchy_global_map.execute();
  A.hybrid_move();
  cell_id_translation.execute();
  A.cell_move();

  
  DisContField d(session,
                graph,
                "u");

  std::cout << "N quads local: " << graph->GetAllQuadGeoms().size() << std::endl;
  std::cout << "N tris local: " << graph->GetAllTriGeoms().size() << std::endl;
  std::cout << "N dofs local: " << d.GetNcoeffs() << std::endl;





  mesh.free();

}

int main(int argc, char **argv) {

  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cout << "ERROR: MPI_Init != MPI_SUCCESS" << std::endl;
    return -1;
  }

  if (argc > 2) {

    std::string argv0 = std::string(argv[1]);
    const int N = std::stoi(argv0);
    
    std::vector<char*> targv(1);

    targv.reserve(argc);
    //targv.push_back(argv[0]);
    for(int ax=2 ; ax<argc ; ax++){
      targv.push_back(argv[ax]);
    }

    hybrid_move_driver(N, argc-1, targv.data());
  } else {
    std::cout << "Insufficient command line arguments" << std::endl;
  }

  if (MPI_Finalize() != MPI_SUCCESS) {
    std::cout << "ERROR: MPI_Finalize != MPI_SUCCESS" << std::endl;
    return -1;
  }

  return 0;
}
