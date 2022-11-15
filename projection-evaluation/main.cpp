#include "nektar_interface/particle_interface.hpp"
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/DisContField.h>
#include <MultiRegions/ContField.h>
#include <SolverUtils/Driver.h>
#include <array>
#include <cmath>
#include <deque>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <vector>

#include "exponential_density.hpp"
#include "function_evaluation.hpp"
#include "function_projection.hpp"
#include "function_vtk_output.hpp"

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
                             ParticleProp(Sym<REAL>("Q"), 1),
                             ParticleProp(Sym<REAL>("FUNC_EVALS"), 1),
                             ParticleProp(Sym<REAL>("FUNC_EVALS_Q"), 1),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  ParticleGroup A(domain, particle_spec, sycl_target);

  NektarCartesianPeriodic pbc(sycl_target, graph, A.position_dat);

  CellIDTranslation cell_id_translation(sycl_target, A.cell_id_dat, mesh);

  const int rank = sycl_target.comm_pair.rank_parent;
  const int size = sycl_target.comm_pair.size_parent;

  std::mt19937 rng_pos(52234234 + rank);

  int rstart, rend;
  get_decomp_1d(size, N_total, rank, &rstart, &rend);
  const int N = rend - rstart;
  int N_check = -1;
  MPICHK(MPI_Allreduce(&N, &N_check, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  NESOASSERT(N_check == N_total, "Error creating particles");

  const int Nsteps = 2000;
  const int cell_count = domain.mesh.get_cell_count();

  if (rank == 0) {
    std::cout << "Starting..." << std::endl;
  }


  if (N > 0) {
    auto positions = uniform_within_extents(N, ndim, pbc.global_extent, rng_pos);
    ParticleSet initial_distribution(N, A.get_particle_spec());

    std::mt19937 rng_cell(52234234 + rank);
    std::uniform_int_distribution rand_int_dist{domain.mesh.get_cell_count()-2, domain.mesh.get_cell_count()-1};

    for (int px = 0; px < N; px++) {
      for (int dimx = 0; dimx < ndim; dimx++) {
        const double pos_orig = positions[dimx][px] + pbc.global_origin[dimx];
        initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig;
      }
      initial_distribution[Sym<INT>("CELL_ID")][px][0] = rand_int_dist(rng_cell);
      initial_distribution[Sym<INT>("ID")][px][0] = px;
    }

    A.add_particles_local(initial_distribution);
  }


  reset_mpi_ranks(A[Sym<INT>("NESO_MPI_RANK")]);

  if (rank == 0) {
    std::cout << "Particles Added..." << std::endl;
  }


  MeshHierarchyGlobalMap mesh_heirarchy_global_map(
      sycl_target, domain.mesh, A.position_dat, A.cell_id_dat, A.mpi_rank_dat);


  nprint(1);
  pbc.execute();
  nprint(2);
  mesh_heirarchy_global_map.execute();

  nprint(3);
  A.hybrid_move();

  nprint(4);
  cell_id_translation.execute();


  nprint(5);
  A.cell_move();


  nprint(6);


    auto t0 = profile_timestamp();

    const auto k_P = A[Sym<REAL>("P")]->cell_dat.device_ptr();
    auto k_Q = A[Sym<REAL>("Q")]->cell_dat.device_ptr();
    const auto k_ID = A[Sym<INT>("ID")]->cell_dat.device_ptr();

    const auto pl_iter_range = A.mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride = A.mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell = A.mpi_rank_dat->get_particle_loop_npart_cell();
    const REAL two_over_sqrt_pi = 1.1283791670955126;
    const REAL reweight = extent[0] * extent[1] / ((REAL) N_total);

    sycl_target.queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {

                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

                const INT pid = k_ID[cellx][0][layerx];
                const REAL x = k_P[cellx][0][layerx];
                const REAL y = k_P[cellx][1][layerx];
                const REAL exp_eval = two_over_sqrt_pi * exp(-(4.0 * ((x - 0.5)*(x - 0.5) + (y - 0.5)*(y - 0.5))));
                k_Q[cellx][0][layerx] = exp_eval * reweight;
                
                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();


  DisContField d(session,
                graph,
                "u");

  nprint(8);

  auto lambda_exp = [&] (const NekDouble x, const NekDouble y) {
    const REAL two_over_sqrt_pi = 1.1283791670955126;
    const REAL exp_eval = two_over_sqrt_pi * exp(-(4.0 * ((x - 0.5)*(x - 0.5) + (y - 0.5)*(y - 0.5))));
    return exp_eval;
  };

  nprint(10);

  
  // get the number of quadrature points over all elements
  const int tot_points = d.GetTotPoints();
  nprint("GetTotPoints:", tot_points);

  Array<OneD, NekDouble> x = Array<OneD, NekDouble>(tot_points);
  Array<OneD, NekDouble> y = Array<OneD, NekDouble>(tot_points);
  Array<OneD, NekDouble> f = Array<OneD, NekDouble>(tot_points);

  auto lambda_f = [&] (const NekDouble x, const NekDouble y) {
    return 10.0 * (x - 0.25) * (x - 0.75) * (y - 0.2) * (y - 0.8);
  };

  d.GetCoords(x, y);
  for(int pointx=0 ; pointx<tot_points ; pointx++){
    f[pointx] = lambda_f(x[pointx], y[pointx]);
  }
  
  Array<OneD, NekDouble> phys_f(tot_points);
  Array<OneD, NekDouble> coeffs_f((unsigned) d.GetNcoeffs());
  
  
  //Project onto expansion
  d.FwdTrans(f, coeffs_f);
  //Backward transform solution to get projected values
  d.BwdTrans(coeffs_f, phys_f);
  d.SetPhys(phys_f);

  FieldEvaluate field_evaluate(d, A, cell_id_translation);
  field_evaluate.evaluate(Sym<REAL>("FUNC_EVALS"));

  if (rank == 0) {
    std::cout << "Checking Eval..." << std::endl;
  }

  auto lambda_check_evals = [&] {

    Array<OneD, NekDouble> point(2);

    double max_err = 0.0;
    double mean_err = 0.0;

    for (int cellx = 0; cellx < cell_count; cellx++) {

      auto positions = A.position_dat->cell_dat.get_cell(cellx);
      auto func_evals = A[Sym<REAL>("FUNC_EVALS")]->cell_dat.get_cell(cellx);

      for (int rowx = 0; rowx < positions->nrow; rowx++) {

        point[0] = (*positions)[0][rowx];
        point[1] = (*positions)[1][rowx];
        
        const double eval_dat = (*func_evals)[0][rowx];
        const double eval_correct = lambda_f(point[0], point[1]);
        
        const double err = ABS(eval_dat - eval_correct);
        max_err = std::max(err, max_err);
        mean_err += err;
      }
    }

    nprint("max_err:", max_err);
    nprint("mean_err:", mean_err / N_total);
  };
  
  lambda_check_evals();
  
  
  // zero d
  for(int pointx=0 ; pointx<tot_points ; pointx++){
    f[pointx] = 0.0;
  }
  //Project onto expansion
  d.FwdTrans(f, coeffs_f);
  //Backward transform solution to get projected values
  d.BwdTrans(coeffs_f, phys_f);
  d.SetPhys(phys_f);

  nprint(11);
  FieldProject field_project(d, A, cell_id_translation);
  field_project.project(Sym<REAL>("Q"));

  write_vtk(d, "projected.vtu");

  FieldEvaluate field_evaluate_q(d, A, cell_id_translation);
  field_evaluate_q.evaluate(Sym<REAL>("FUNC_EVALS_Q"));


  H5Part h5part("exp.h5part", A, 
      Sym<REAL>("P"),
      Sym<REAL>("Q"),
      Sym<INT>("ID"),
      Sym<INT>("NESO_MPI_RANK"),
      Sym<REAL>("NESO_REFERENCE_POSITIONS"),
      Sym<REAL>("FUNC_EVALS"),
      Sym<REAL>("FUNC_EVALS_Q")
  );
  h5part.write();
  h5part.close();


  for(int pointx=0 ; pointx<tot_points ; pointx++){
    f[pointx] = lambda_exp(x[pointx], y[pointx]);
  }
  //Project onto expansion
  d.FwdTrans(f, coeffs_f);
  //Backward transform solution to get projected values
  d.BwdTrans(coeffs_f, phys_f);
  d.SetPhys(phys_f);

  write_vtk(d, "target_density.vtu");


  mesh.free();

}

int main(int argc, char **argv) {

  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cout << "ERROR: MPI_Init != MPI_SUCCESS" << std::endl;
    return -1;
  }

  if (argc > 2) {

    //std::string argv0 = std::string(argv[1]);
    //const int N = std::stoi(argv0);
    //
    //std::vector<char*> targv(1);

    //targv.reserve(argc);
    ////targv.push_back(argv[0]);
    //for(int ax=2 ; ax<argc ; ax++){
    //  targv.push_back(argv[ax]);
    //}

    //hybrid_move_driver(N, argc-1, targv.data());

    hybrid_move_driver(1000000, argc, argv);
  } else {
    std::cout << "Insufficient command line arguments" << std::endl;
  }

  if (MPI_Finalize() != MPI_SUCCESS) {
    std::cout << "ERROR: MPI_Finalize != MPI_SUCCESS" << std::endl;
    return -1;
  }

  return 0;
}



//inline void hybrid_move_driver2(const int N_total, int argc, char ** argv) {
//
//  LibUtilities::SessionReaderSharedPtr session;
//  SpatialDomains::MeshGraphSharedPtr graph;
//  // Create session reader.
//  session = LibUtilities::SessionReader::CreateInstance(argc, argv);
//  graph = SpatialDomains::MeshGraph::Read(session);
//
//
//  std::vector<int> dims(2);
//  dims[0] = 16;
//  dims[1] = 16;
//
//  const double cell_extent = 1.0;
//  const int subdivision_order = 1;
//  const int stencil_width = 1;
//  CartesianHMesh mesh(MPI_COMM_WORLD, 2, dims, cell_extent,
//                      subdivision_order, stencil_width);
//
//
//  //ParticleMeshInterface mesh(graph);
//
//
//
//
//  SYCLTarget sycl_target{0, mesh.get_comm()};
//
//  //auto nektar_graph_local_mapper =
//  //    std::make_shared<NektarGraphLocalMapperT>(sycl_target, mesh, 1.0e-10);
//  //Domain domain(mesh, nektar_graph_local_mapper);
//  
//  Domain domain(mesh);
//
//
//  const int ndim = 2;
//  const double extent[2] = {1.0, 1.0};
//  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
//                             ParticleProp(Sym<REAL>("Q"), 1),
//                             //ParticleProp(Sym<REAL>("FUNC_EVALS"), 1),
//                             //ParticleProp(Sym<REAL>("FUNC_EVALS_Q"), 1),
//                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
//                             ParticleProp(Sym<INT>("ID"), 1)};
//
//  ParticleGroup A(domain, particle_spec, sycl_target);
//
//  //NektarCartesianPeriodic pbc(sycl_target, graph, A.position_dat);
//
//  //CellIDTranslation cell_id_translation(sycl_target, A.cell_id_dat, mesh);
//
//  const int rank = sycl_target.comm_pair.rank_parent;
//  const int size = sycl_target.comm_pair.size_parent;
//
//  std::mt19937 rng_pos(52234234 + rank);
//
//  int rstart, rend;
//  get_decomp_1d(size, N_total, rank, &rstart, &rend);
//  const int N = rend - rstart;
//  int N_check = -1;
//  MPICHK(MPI_Allreduce(&N, &N_check, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
//  NESOASSERT(N_check == N_total, "Error creating particles");
//
//  const int Nsteps = 2000;
//  const int cell_count = domain.mesh.get_cell_count();
//
//  if (rank == 0) {
//    std::cout << "Starting..." << std::endl;
//  }
//
//
//  if (N > 0) {
//    //auto positions = uniform_within_extents(N, ndim, pbc.global_extent, rng_pos);
//    ParticleSet initial_distribution(N, A.get_particle_spec());
//    
//
//    //std::mt19937 rng_cell(52234234 + rank);
//    //std::uniform_int_distribution rand_int_dist{domain.mesh.get_cell_count()-2, domain.mesh.get_cell_count()-1};
//
//
//    //for (int px = 0; px < N; px++) {
//    //  for (int dimx = 0; dimx < ndim; dimx++) {
//    //    const double pos_orig = positions[dimx][px] + pbc.global_origin[dimx];
//    //    initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig;
//    //  }
//    //  //initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
//    //  initial_distribution[Sym<INT>("CELL_ID")][px][0] = rand_int_dist(rng_cell);
//    //  initial_distribution[Sym<INT>("ID")][px][0] = px;
//    //}
//
//
//    A.add_particles_local(initial_distribution);
//  }
//
//
//  /*
//  reset_mpi_ranks(A[Sym<INT>("NESO_MPI_RANK")]);
//
//  if (rank == 0) {
//    std::cout << "Particles Added..." << std::endl;
//  }
//
//
//
//  MeshHierarchyGlobalMap mesh_heirarchy_global_map(
//      sycl_target, domain.mesh, A.position_dat, A.cell_id_dat, A.mpi_rank_dat);
//
//
//  nprint(1);
//  pbc.execute();
//  nprint(2);
//  mesh_heirarchy_global_map.execute();
//
//  nprint(3);
//  A.hybrid_move();
//
//  nprint(4);
//  cell_id_translation.execute();
//
//*/
///*
//  auto lambda_cell_id = [&] {
//
//    auto t0 = profile_timestamp();
//
//    const auto k_ID = A[Sym<INT>("ID")]->cell_dat.device_ptr();
//    const auto k_CELL_ID = A[Sym<INT>("CELL_ID")]->cell_dat.device_ptr();
//
//    const auto pl_iter_range = A.mpi_rank_dat->get_particle_loop_iter_range();
//    const auto pl_stride = A.mpi_rank_dat->get_particle_loop_cell_stride();
//    const auto pl_npart_cell = A.mpi_rank_dat->get_particle_loop_npart_cell();
//
//    ErrorPropagate ep(sycl_target);
//    auto k_ep = ep.device_ptr();
//
//    sycl_target.queue
//        .submit([&](sycl::handler &cgh) {
//          cgh.parallel_for<>(
//              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
//                NESO_PARTICLES_KERNEL_START
//                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
//                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;
//                const INT pid = k_ID[cellx][0][layerx];
//                if(pid > 1000000){
//                  k_CELL_ID[cellx][0][layerx] = 99;
//                bool valid = (cellx >= 0) && (cellx < 100);
//                NESO_KERNEL_ASSERT(valid, k_ep);
//                }
//                NESO_PARTICLES_KERNEL_END
//              });
//        })
//        .wait_and_throw();
//        ep.check_and_throw("cellid kernel failed.");
//        
//  };
//  
//  lambda_cell_id();
//  */
//
//
//
//
//
//  nprint(5);
//  //A.cell_move();
//
//
//  nprint(6);
//
//
//    auto t0 = profile_timestamp();
//
//    const auto k_P = A[Sym<REAL>("P")]->cell_dat.device_ptr();
//    auto k_Q = A[Sym<REAL>("Q")]->cell_dat.device_ptr();
//    const auto k_ID = A[Sym<INT>("ID")]->cell_dat.device_ptr();
//
//    const auto pl_iter_range = A.mpi_rank_dat->get_particle_loop_iter_range();
//    const auto pl_stride = A.mpi_rank_dat->get_particle_loop_cell_stride();
//    const auto pl_npart_cell = A.mpi_rank_dat->get_particle_loop_npart_cell();
//    const REAL two_over_sqrt_pi = 1.1283791670955126;
//    const REAL reweight = extent[0] * extent[1] / ((REAL) N_total);
//
//    ErrorPropagate ep(sycl_target);
//    auto k_ep = ep.device_ptr();
//
//    nprint("pl_iter_range", pl_iter_range);
//    nprint("pl_stride", pl_stride);
//
//    sycl_target.queue
//        .submit([&](sycl::handler &cgh) {
//            //sycl::stream out(1024, 256, cgh);
//          cgh.parallel_for<>(
//              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
//
//                NESO_PARTICLES_KERNEL_START
//                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
//                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;
//
//                const INT pid = k_ID[cellx][0][layerx];
//                const REAL x = k_P[cellx][0][layerx];
//                const REAL y = k_P[cellx][1][layerx];
//                const REAL exp_eval = two_over_sqrt_pi * exp(-(4.0 * ((x - 0.5)*(x - 0.5) + (y - 0.5)*(y - 0.5))));
//                k_Q[cellx][0][layerx] = exp_eval * reweight + pid;
//                //k_Q[cellx][0][layerx] = pid;
//
//                //const INT pid = k_ID[cellx][0][layerx];
//                //const REAL x = k_P[cellx][0][layerx];
//                //const REAL y = k_P[cellx][1][layerx];
//                //k_Q[cellx][0][layerx] = pid + x + y;
//                
//                bool valid = ((cellx != 0) && (pl_npart_cell[cellx] == 0)) || ((cellx == 0) && (pl_npart_cell[cellx] == 200000)) ;
//                NESO_KERNEL_ASSERT(valid, k_ep);
//                
//                NESO_PARTICLES_KERNEL_END
//              });
//        })
//        .wait_and_throw();
//        ep.check_and_throw("Exp kernel failed.");
//        
//
//  nprint(7);
//  sycl_target.check_ptrs();
//
//
//  //const int Ntest = 20000000;
//  //int * d_a = sycl::malloc_device<int>(Ntest, sycl_target.queue);
//  //sycl_target.queue
//  //    .submit([&](sycl::handler &cgh) {
//  //      cgh.parallel_for<>(
//  //          sycl::range<1>(Ntest), [=](sycl::id<1> idx) {
//  //            d_a[idx] = (int)idx;
//  //          });
//  //    })
//  //    .wait_and_throw();
//
//
//
//  DisContField d(session,
//                graph,
//                "u");
//
//  //sycl::free(d_a, sycl_target.queue);
//
//  //nprint(7.5);
//  //H5Part h5part2("exp.h5part", A, 
//  //    Sym<REAL>("P"),
//  //    Sym<REAL>("Q"),
//  //    Sym<INT>("ID"),
//  //    Sym<INT>("NESO_MPI_RANK"),
//  //    Sym<REAL>("NESO_REFERENCE_POSITIONS"),
//  //    Sym<REAL>("FUNC_EVALS"),
//  //    Sym<REAL>("FUNC_EVALS_Q")
//  //);
//  //h5part2.write();
//  //h5part2.close();
//
//  nprint(8);
//
//  auto lambda_exp = [&] (const NekDouble x, const NekDouble y) {
//    const REAL two_over_sqrt_pi = 1.1283791670955126;
//    const REAL exp_eval = two_over_sqrt_pi * exp(-(4.0 * ((x - 0.5)*(x - 0.5) + (y - 0.5)*(y - 0.5))));
//    return exp_eval;
//  };
//
//  nprint(9);
//  /*
//  DisContField d(session,
//                graph,
//                "u");
//*/
//  nprint(10);
//  //StdPhysEvaluate 	( 	const Array< OneD, const NekDouble > &  	Lcoord,
//	//	const Array< OneD, const NekDouble > &  	physvals 
//	//) 	
//  
//
//  mesh.free();
//  return;
//
//
////  std::cout << "N quads local: " << graph->GetAllQuadGeoms().size() << std::endl;
////  std::cout << "N tris local: " << graph->GetAllTriGeoms().size() << std::endl;
////  std::cout << "N coeffs local: " << d.GetNcoeffs() << std::endl;
////
////
////  auto d_expansions = d.GetExp();
////  std::cout << "N expansions: " << d_expansions->size() << std::endl;
////
////  //for (int ex=0 ; ex<d_expansions->size() ; ex++){
////  for (int ex=0 ; ex<1 ; ex++){
////
////    auto de0 = (*d_expansions)[ex];
////    const Array< OneD, const LibUtilities::BasisSharedPtr > de0_basis = de0->GetBase();
////
////
////    std::cout << "basis array size: " << de0_basis.size() << std::endl;
////    for(int bx=0 ; bx<2 ; bx++){
////      auto de0_basis0 = de0_basis[bx];
////      std::cout << "\tnum modes: " << de0_basis0->GetNumModes() << std::endl;
////      auto basis_key = de0_basis0->GetBasisKey();
////      std::cout << "\tbasis key: " << basis_key << std::endl;
////    }
////  }
////
////  auto coeffs = d.GetCoeffs();
////  auto Ncoeffs = coeffs.size();
////  std::cout << "Ncoeffs: " << Ncoeffs << std::endl;
////
////  // offset in coeffs for i^th expansion
////  auto offset_4 = d.GetCoeff_Offset(4);
////  nprint("offset_4", offset_4);
////
////  
////  // get the number of quadrature points over all elements
////  const int tot_points = d.GetTotPoints();
////  nprint("GetTotPoints:", tot_points);
////
////  Array<OneD, NekDouble> x = Array<OneD, NekDouble>(tot_points);
////  Array<OneD, NekDouble> y = Array<OneD, NekDouble>(tot_points);
////  Array<OneD, NekDouble> f = Array<OneD, NekDouble>(tot_points);
////
////  auto lambda_f = [&] (const NekDouble x, const NekDouble y) {
////    return 10.0 * (x - 0.25) * (x - 0.75) * (y - 0.2) * (y - 0.8);
////  };
////
////  d.GetCoords(x, y);
////  for(int pointx=0 ; pointx<tot_points ; pointx++){
////    f[pointx] = lambda_f(x[pointx], y[pointx]);
////  }
////  
////  Array<OneD, NekDouble> phys_f(tot_points);
////  Array<OneD, NekDouble> coeffs_f((unsigned) d.GetNcoeffs());
////  
////
////  /*
////  //Project onto expansion
////  d.FwdTrans(f, coeffs_f);
////  //Backward transform solution to get projected values
////  d.BwdTrans(coeffs_f, phys_f);
////  d.SetPhys(phys_f);
////
////  FieldEvaluate field_evaluate(d, A, cell_id_translation);
////  field_evaluate.evaluate(Sym<REAL>("FUNC_EVALS"));
////
////  if (rank == 0) {
////    std::cout << "Checking Eval..." << std::endl;
////  }
////
////  auto lambda_check_evals = [&] {
////
////    Array<OneD, NekDouble> point(2);
////
////    double max_err = 0.0;
////
////    for (int cellx = 0; cellx < cell_count; cellx++) {
////
////      auto positions = A.position_dat->cell_dat.get_cell(cellx);
////      auto func_evals = A[Sym<REAL>("FUNC_EVALS")]->cell_dat.get_cell(cellx);
////
////      for (int rowx = 0; rowx < positions->nrow; rowx++) {
////
////        point[0] = (*positions)[0][rowx];
////        point[1] = (*positions)[1][rowx];
////        
////        const double eval_dat = (*func_evals)[0][rowx];
////        const double eval_correct = lambda_f(point[0], point[1]);
////        
////        const double err = ABS(eval_dat - eval_correct);
////        max_err = std::max(err, max_err);
////      }
////    }
////
////    nprint("max_err:", max_err);
////  };
////  
////  lambda_check_evals();
////  */
////  
////  // zero d
////  for(int pointx=0 ; pointx<tot_points ; pointx++){
////    f[pointx] = 0.0;
////  }
////  //Project onto expansion
////  d.FwdTrans(f, coeffs_f);
////  //Backward transform solution to get projected values
////  d.BwdTrans(coeffs_f, phys_f);
////  d.SetPhys(phys_f);
////
////  nprint(11);
////  FieldProject field_project(d, A, cell_id_translation);
////  field_project.project(Sym<REAL>("Q"));
////
////  write_vtk(d, "projected.vtu");
////
////  FieldEvaluate field_evaluate_q(d, A, cell_id_translation);
////  field_evaluate_q.evaluate(Sym<REAL>("FUNC_EVALS_Q"));
////
////
////  H5Part h5part("exp.h5part", A, 
////      Sym<REAL>("P"),
////      Sym<REAL>("Q"),
////      Sym<INT>("ID"),
////      Sym<INT>("NESO_MPI_RANK"),
////      Sym<REAL>("NESO_REFERENCE_POSITIONS"),
////      Sym<REAL>("FUNC_EVALS"),
////      Sym<REAL>("FUNC_EVALS_Q")
////  );
////  h5part.write();
////  h5part.close();
////
////
////
////  for(int pointx=0 ; pointx<tot_points ; pointx++){
////    f[pointx] = lambda_exp(x[pointx], y[pointx]);
////  }
////  //Project onto expansion
////  d.FwdTrans(f, coeffs_f);
////  //Backward transform solution to get projected values
////  d.BwdTrans(coeffs_f, phys_f);
////  d.SetPhys(phys_f);
////
////  write_vtk(d, "target_density.vtu");
////
////
////
//////WriteVtkHeader <- explist
//////PhysEvaluateBasis?
////
////
////
////// ExpList::SetupCoeffPhys
////// m_coeff_offset[i]
////
////  /*
////   *
////   *
//// * m_bdata[i + j*m_numpoints] =
//// * \f$ \phi^a_i(z_j) = \left \{
//// * \begin{array}{ll} \left ( \frac{1-z_j}{2}\right ) & i = 0 \\
//// * \\
//// * \left ( \frac{1+z_j}{2}\right ) & i = 1 \\
//// * \\
//// * \left ( \frac{1-z_j}{2}\right )\left ( \frac{1+z_j}{2}\right )
//// *  P^{1,1}_{i-2}(z_j) & 2\leq i < P\\
//// *  \end{array} \right . \f$
////   *
////           case eModified_A:
////  
////             // Note the following packing deviates from the
////             // definition in the Book by Karniadakis in that we
////             // put the vertex degrees of freedom at the lower
////             // index range to follow a more hierarchic structure.
////             
////
////            z: quadrature points
////            w: quadrature weights
////            m_bdata: basis definition array -> get with GetBdata()
////
////  
////             for (i = 0; i < numPoints; ++i)
////             {
////                 m_bdata[i]             = 0.5 * (1 - z[i]);
////                 m_bdata[numPoints + i] = 0.5 * (1 + z[i]);
////             }
////  
////             mode = m_bdata.data() + 2 * numPoints;
////  
////             for (p = 2; p < numModes; ++p, mode += numPoints)
////             {
////                 Polylib::jacobfd(numPoints, z.data(), mode, NULL, p - 2, 1.0,
////                                  1.0);
////  
////                 for (i = 0; i < numPoints; ++i)
////                 {
////                     mode[i] *= m_bdata[i] * m_bdata[numPoints + i];
////                 }
////             }
////  
////             // define derivative basis
////             Blas::Dgemm('n', 'n', numPoints, numModes, numPoints, 1.0, D,
////                         numPoints, m_bdata.data(), numPoints, 0.0,
////                         m_dbdata.data(), numPoints);
////
////
////  */
////  
////
////
////
////
////
////  mesh.free();
//
//}
