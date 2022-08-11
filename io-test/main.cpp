#include <iostream>
#include <mpi.h>
#include <neso_particles.hpp>
#include <chrono>
#include <csignal>
#include <string>

using namespace cl;
using namespace NESO::Particles;

inline void hybrid_move_driver(const int N_total){


  //int size, rank;
  //MPI_Comm_size(MPI_COMM_WORLD, &size);
  //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //
  //const int width = 2;
  //int N = 0, offset = 0;
  //const int start = width * rank;
  //
  //std::vector<int64_t> id{width};
  //std::vector<double> x{width};
  //std::vector<double> y{width};
  //std::vector<double> z{width};
  //std::vector<double> px{width};
  //std::vector<double> py{width};
  //std::vector<double> pz{width};
  //
  //for(int ix=0 ; ix<width ; ix++){
  //  id[ix] = (int64_t) rank * width + ix;
  //  x[ix] = (double) rank * width + ix;
  //  y[ix] = (double) rank * width + ix;
  //  z[ix] = (double) rank * width + ix;
  //  px[ix] = (double) 0;
  //  py[ix] = (double) 0;
  //  pz[ix] = (double) 0;
  //}
  //
  //
  //hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  //H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
  //hid_t file_id = H5Fcreate("foo.h5part", H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  //
  //auto group_step = H5Gcreate(file_id, "Step#0", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  //
  //hsize_t dims[1] = {width};
  //auto memspace = H5Screate_simple(1, dims, NULL);
  //MPICHK(
  //  MPI_Allreduce(
  //    &width, &N, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD
  //  )
  //);
  //dims[0] = N;
  //auto filespace = H5Screate_simple(1, dims, NULL);
  //MPICHK(MPI_Scan(&width, &offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  //offset -= width;
  //hsize_t start_offset[1] = {static_cast<hsize_t>(offset)};
  //hsize_t counts[1] = {static_cast<hsize_t>(width)};
  //
  //H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start_offset, NULL, counts, NULL);
  //auto dxpl = H5Pcreate(H5P_DATASET_XFER);
  //H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);
  //
  //auto dset = H5Dcreate1(group_step, "id", H5T_NATIVE_LLONG, filespace, H5P_DEFAULT);
  //H5Dwrite(dset, H5T_NATIVE_LLONG, memspace, filespace, dxpl, id.data());
  //H5Dclose(dset);
  //dset = H5Dcreate1(group_step, "x", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT);
  //H5Dwrite(dset, H5T_NATIVE_DOUBLE, memspace, filespace, dxpl, x.data());
  //H5Dclose(dset);
  //dset = H5Dcreate1(group_step, "y", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT);
  //H5Dwrite(dset, H5T_NATIVE_DOUBLE, memspace, filespace, dxpl, y.data());
  //H5Dclose(dset);
  //dset = H5Dcreate1(group_step, "z", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT);
  //H5Dwrite(dset, H5T_NATIVE_DOUBLE, memspace, filespace, dxpl, z.data());
  //H5Dclose(dset);
  //dset = H5Dcreate1(group_step, "px", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT);
  //H5Dwrite(dset, H5T_NATIVE_DOUBLE, memspace, filespace, dxpl, px.data());
  //H5Dclose(dset);
  //dset = H5Dcreate1(group_step, "py", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT);
  //H5Dwrite(dset, H5T_NATIVE_DOUBLE, memspace, filespace, dxpl, py.data());
  //H5Dclose(dset);
  //dset = H5Dcreate1(group_step, "pz", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT);
  //H5Dwrite(dset, H5T_NATIVE_DOUBLE, memspace, filespace, dxpl, pz.data());
  //H5Dclose(dset);
  //
  //H5Pclose(dxpl);
  //H5Sclose(filespace);
  //H5Sclose(memspace);
  //H5Gclose(group_step);
  //H5Fclose(file_id);
  //H5Pclose(plist_id);

   
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


  const int Nsteps_warmup = 1;
  const REAL dt = 0.1;
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

  
  
  if (rank == 0){
    std::cout << "Particles Added..." << std::endl;
  }

  MeshHierarchyGlobalMap mesh_heirarchy_global_map(
      sycl_target, domain.mesh, A.position_dat, A.cell_id_dat, A.mpi_rank_dat);

  CartesianPeriodic pbc(sycl_target, mesh, A.position_dat);
  CartesianCellBin ccb(sycl_target, mesh, A.position_dat, A.cell_id_dat);

  auto lambda_advect = [&] {

    auto t0 = profile_timestamp();

    auto k_P = A[Sym<REAL>("P")]->cell_dat.device_ptr();
    const auto k_V = A[Sym<REAL>("V")]->cell_dat.device_ptr();
    const auto k_ndim = ndim;
    const auto k_dt = dt;

    const auto pl_iter_range = A.mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride = A.mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell = A.mpi_rank_dat->get_particle_loop_npart_cell();

    sycl_target.profile_map.inc("Advect", "Prepare", 1, profile_elapsed(t0, profile_timestamp()));
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
    sycl_target.profile_map.inc("Advect", "Execute", 1, profile_elapsed(t0, profile_timestamp()));
  };

  REAL T = 0.0;
 
  pbc.execute();
  mesh_heirarchy_global_map.execute();
  A.global_move();
  ccb.execute();
  A.cell_move();  

  MPI_Barrier(sycl_target.comm_pair.comm_parent);
  if (rank == 0){
    std::cout << N_total << " Particles Distributed..." << std::endl;
  }

  H5Part h5part("bar.h5part", A, 
      Sym<REAL>("P_ORIG"),
      Sym<REAL>("V"),
      Sym<INT>("ID"),
      Sym<INT>("NESO_MPI_RANK")
  );


  for (int stepx = 0; stepx < Nsteps_warmup; stepx++) {

    pbc.execute();

    A.hybrid_move();

    ccb.execute();
    A.cell_move();

    h5part.write();
    lambda_advect();

    T += dt;
    
    if( (stepx % 100 == 0) && (rank == 0)) {
      std::cout << stepx << std::endl;
    }
  }
  
  
  h5part.close();
  mesh.free();
  
  if (rank == 0){
    sycl_target.profile_map.print();
  }
  
  
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
