#ifndef _FUNCTION_PROJECTION
#define _FUNCTION_PROJECTION

#include <map>
#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <neso_particles.hpp>

using namespace Nektar::LibUtilities;
using namespace Nektar::SpatialDomains;
using namespace Nektar::MultiRegions;

template <typename T>
inline void multiply_by_inverse_mass_matrix(
  T & field,
  const Array< OneD, const NekDouble > &inarray,
  Array< OneD, NekDouble> &outarray
){
  field.MultiplyByInvMassMatrix(inarray, outarray);
}


template <>
inline void multiply_by_inverse_mass_matrix(
  DisContField &field,
  const Array< OneD, const NekDouble > &inarray,
  Array< OneD, NekDouble> &outarray
){
  field.MultiplyByElmtInvMass(inarray, outarray);

    /*
  auto expansions = field.GetExp();
  const int num_expansions = (*expansions).size();
  for(int ex=0 ; ex<num_expansions ; ex++){
    auto exp = (*expansions)[ex];
    const int exp_offset = field.GetCoeff_Offset(ex);
    
    exp->MultiplyByInvMassMatrix(
      inarray + exp_offset, 
      outarray + exp_offset
    );
  }
    */
}



/**
 * TODO
 */
template <typename T>
class FieldProject {

private:
  T &field;
  ParticleGroup &particle_group;
  SYCLTarget &sycl_target;
  CellIDTranslation &cell_id_translation;

  // map from Nektar++ geometry ids to Nektar++ expanions ids for the field
  std::map<int, int> geom_to_exp;

public:

  ~FieldProject(){};

  /**
   * TODO
   */
  FieldProject(
    T &field,
    ParticleGroup &particle_group,
    CellIDTranslation &cell_id_translation
  ) : 
    field(field),
    particle_group(particle_group),
    sycl_target(particle_group.sycl_target),
    cell_id_translation(cell_id_translation)
  {
    
    // build the map from geometry ids to expansion ids
    auto expansions = this->field.GetExp();
    const int num_expansions = (*expansions).size();
    for(int ex=0 ; ex<num_expansions ; ex++){
      auto exp = (*expansions)[ex];
      // The indexing in Nektar++ source suggests that ex is the important
      // index if these do not match in future.
      NESOASSERT(ex == exp->GetElmtId(), "expected expansion id to match element id?");
      int geom_gid = exp->GetGeom()->GetGlobalID();
      this->geom_to_exp[geom_gid] = ex;
    }

  };
  
  /**
   * TODO
   */
  template <typename U>
  inline void project(
    Sym<U> sym
  ){

    auto input_dat = this->particle_group[sym];
    auto ref_position_dat = this->particle_group[Sym<REAL>("NESO_REFERENCE_POSITIONS")];

    const int nrow_max = this->particle_group.mpi_rank_dat->cell_dat.get_nrow_max();
    const int ncol = input_dat->ncomp;
    const int particle_ndim = ref_position_dat->ncomp;
  
    Array<OneD, NekDouble> local_coord(particle_ndim);

    // Get the physvals from the Nektar++ field.
    auto global_physvals = this->field.GetPhys();

    const int ncoeffs = this->field.GetNcoeffs();
    Array<OneD, NekDouble> global_coeffs = Array<OneD, NekDouble>(ncoeffs);
    Array<OneD, NekDouble> global_phi = Array<OneD, NekDouble>(ncoeffs);

    CellDataT<U> input_tmp(this->sycl_target, nrow_max, ncol);
    CellDataT<REAL> ref_positions_tmp(this->sycl_target, nrow_max, particle_ndim);
    EventStack event_stack;

    const int neso_cell_count = this->particle_group.domain.mesh.get_cell_count();
    for(int neso_cellx=0 ; neso_cellx<neso_cell_count ; neso_cellx++){
      // Get the source values.
      input_dat->cell_dat.get_cell_async(neso_cellx, input_tmp, event_stack);
      // Get the reference positions from the particle in the cell
      ref_position_dat->cell_dat.get_cell_async(neso_cellx, ref_positions_tmp,
                                            event_stack);
      event_stack.wait();

      // Get the nektar++ geometry id that corresponds to this NESO cell id
      const int nektar_geom_id = this->cell_id_translation.map_to_nektar[neso_cellx];

      // Map from the geometry id to the expansion id for the field.
      NESOASSERT(this->geom_to_exp.count(nektar_geom_id),
          "Could not find expansion id for geom id");
      const int nektar_expansion_id = this->geom_to_exp[nektar_geom_id];
      const int coordim = this->field.GetCoordim(nektar_expansion_id);
      NESOASSERT(particle_ndim >= coordim,
          "mismatch in coordinate size");

      // Get the expansion object that corresponds to this expansion id
      auto nektar_expansion = this->field.GetExp(nektar_expansion_id);

      // compute the number of modes in this expansion
      int num_modes_t = 1;
      for(int dimx=0 ; dimx<coordim ; dimx++){
        num_modes_t *= nektar_expansion->GetBasisNumModes(dimx);
      }
      const int num_modes = num_modes_t;

      auto phi = global_phi + this->field.GetCoeff_Offset(nektar_expansion_id);

      // zero the output array
      for(int modex=0; modex<num_modes ; modex++){
        phi[modex] = 0.0;
      }

      const int nrow = input_dat->cell_dat.nrow[neso_cellx];
      // for each particle in the cell
      for(int rowx=0 ; rowx<nrow ; rowx++){

        const REAL quantity = input_tmp[0][rowx];
        
        // read the reference position from the particle
        for(int dimx=0 ; dimx<particle_ndim ; dimx++){
          local_coord[dimx] = ref_positions_tmp[dimx][rowx];
        }

        // for each mode in the expansion
        for(int modex=0; modex<num_modes ; modex++){
          const double phi_j = nektar_expansion->PhysEvaluateBasis(local_coord, modex);
          phi[modex] += phi_j * quantity;
        }

      }
    }
    
    // input array then output array -> FAILS?
    //this->field.MultiplyByInvMassMatrix(global_phi, global_coeffs);
    multiply_by_inverse_mass_matrix(this->field, global_phi, global_coeffs);
    
    // set the coefficients on the function
    this->field.SetCoeffsArray(global_coeffs);

    Array<OneD, NekDouble> global_phys(this->field.GetTotPoints());
    this->field.BwdTrans(global_coeffs, global_phys);
    this->field.SetPhys(global_phys);
  }


};

#endif
