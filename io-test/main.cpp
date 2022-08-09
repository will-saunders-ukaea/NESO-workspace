#include <iostream>
#include <mpi.h>
#include <neso_particles.hpp>
#include <chrono>
#include <csignal>
#include <string>

using namespace cl;
using namespace NESO::Particles;

inline void hybrid_move_driver(const int N_total){


  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  const int width = 2;
  int N = 0, offset = 0;
  const int start = width * rank;

  std::vector<double> x;
  x.reserve(width);
  for(int ix=0 ; ix<width ; ix++){
    x[ix] = (double) rank * width + ix;
  }

  
  nprint("MARCO");
  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
  hid_t file_id = H5Fcreate("foo.h5", H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);

  auto group_step = H5Gcreate(file_id, "Step#0", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  
  hsize_t dims[1] = {width};
  auto memspace = H5Screate_simple(1, dims, NULL);
  MPICHK(
    MPI_Allreduce(
      &width, &N, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD
    )
  );
  dims[0] = N;
  auto filespace = H5Screate_simple(1, dims, NULL);
  MPICHK(MPI_Scan(&width, &offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  offset -= width;
  nprint("offset", offset);
  hsize_t start_offset[1] = {static_cast<hsize_t>(offset)};
  hsize_t counts[1] = {static_cast<hsize_t>(width)};

  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start_offset, NULL, counts, NULL);
  auto dxpl = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);


  auto dset = H5Dcreate1(group_step, "x", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT);
  H5Dwrite(dset, H5T_NATIVE_DOUBLE, memspace, filespace, dxpl, x.data());
  H5Dclose(dset);

  H5Pclose(dxpl);
  H5Sclose(filespace);
  H5Sclose(memspace);
  H5Gclose(group_step);
  H5Fclose(file_id);
  H5Pclose(plist_id);

  nprint("POLO");
  /*
  const int ndim = 2;
  std::vector<int> dims(ndim);
  dims[0] = 16;
  dims[1] = 16;

  const double cell_extent = 1.0;
  const int subdivision_order = 1;
  const int stencil_width = 1;
  CartesianHMesh mesh(MPI_COMM_WORLD, ndim, dims, cell_extent,
                      subdivision_order, stencil_width);

  SYCLTarget sycl_target{0, mesh.get_comm()};

  auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);
  Domain domain(mesh, cart_local_mapper);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("P_ORIG"), ndim),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  ParticleGroup A(domain, particle_spec, sycl_target);

  A.add_particle_dat(ParticleDat(sycl_target, ParticleProp(Sym<REAL>("FOO"), 3),
                                 domain.mesh.get_cell_count()));

  std::mt19937 rng_pos(52234234);
  std::mt19937 rng_vel(52234231);
  std::mt19937 rng_rank(18241);

  const int rank = sycl_target.comm_pair.rank_parent;
  const int size = sycl_target.comm_pair.size_parent;
  
  
  int rstart, rend;
  get_decomp_1d(size, N_total, rank, &rstart, &rend);
  const int N = rend - rstart;

  int N_check = -1;
  MPICHK(MPI_Allreduce(&N, &N_check, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  NESOASSERT(N_check == N_total, "Error creating particles");


  const int Nsteps_warmup = 1024;
  const int Nsteps = 2048;
  const REAL dt = 0.001;
  const int cell_count = domain.mesh.get_cell_count();

  if (rank == 0){
    std::cout << "Starting..." << std::endl;
  }
  
  if (N > 0){
  auto positions =
      uniform_within_extents(N, ndim, mesh.global_extents, rng_pos);
  auto velocities = NESO::Particles::normal_distribution(
      N, 3, 0.0, 0.5, rng_vel);
  std::uniform_int_distribution<int> uniform_dist(
      0, sycl_target.comm_pair.size_parent - 1);
  ParticleSet initial_distribution(N, A.get_particle_spec());
  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] = positions[dimx][px];
      initial_distribution[Sym<REAL>("P_ORIG")][px][dimx] = positions[dimx][px];
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





  mesh.free();
  
  if (rank == 0){
    sycl_target.profile_map.print();
  }
  */
}

int main(int argc, char **argv) {

  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cout << "ERROR: MPI_Init != MPI_SUCCESS" << std::endl;
    return -1;
  }
 
  if (argc > 1){
    
    std::string argv0 = std::string(argv[1]);
    const int N = std::stoi(argv0);

    hybrid_move_driver(N);

  }

  if (MPI_Finalize() != MPI_SUCCESS) {
    std::cout << "ERROR: MPI_Finalize != MPI_SUCCESS" << std::endl;
    return -1;
  }

  return 0;
}
