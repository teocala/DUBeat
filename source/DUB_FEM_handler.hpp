/********************************************************************************
  Copyright (C) 2022 by the DUBeat authors.

  This file is part of DUBeat.

  DUBeat is free software; you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  DUBeat is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with DUBeat.  If not, see <http://www.gnu.org/licenses/>.
********************************************************************************/

/**
 * @file
 *
 * @author Federica Botta <federica.botta@mail.polimi.it>.
 * @author Matteo Calafà <matteo.calafa@mail.polimi.it>.
 */

#ifndef DUBFEMHandler_HPP_
#define DUBFEMHandler_HPP_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping_q1_eulerian.h>

#include <deal.II/lac/trilinos_vector.h>

#include <cmath>
#include <filesystem>
#include <vector>

#include "DUBValues.hpp"
#include "dof_handler_DG.hpp"
#include "source/geometry/mesh_handler.hpp"
#include "source/init.hpp"
#include "volume_handler_DG.hpp"

/**
 * @brief
 * Class used to discretize analytical solutions as linear combinations of DGFEM
 * or Dubiner basis. It also provides conversions for solution vectors with
 * respect to the DGFEM basis to the Dubiner basis and viceversa. This class is
 * necessary, for instance, to discretize initial analytical solutions for
 * time-dependent problems or to obtain solution vectors for contour plots at
 * the end of the system solving.
 */
template <class basis>
class DUBFEMHandler : public DUBValues<lifex::dim>
{
private:
  /// Dof handler object of the problem.
  const DoFHandlerDG<basis> &dof_handler;

  /// Number of quadrature points in the volume element.
  /// By default: @f$(degree+2)^{dim}@f$.
  const unsigned int n_quad_points;

public:
  /// Constructor.
  DUBFEMHandler<basis>(const unsigned int         degree,
                       const DoFHandlerDG<basis> &dof_hand)
    : DUBValues<lifex::dim>(degree)
    , dof_handler(dof_hand)
    , n_quad_points(static_cast<int>(std::pow(degree + 2, lifex::dim)))
  {}

  /// Default copy constructor.
  DUBFEMHandler<basis>(DUBFEMHandler<basis> &DUBFEMHandler) = default;

  /// Default const copy constructor.
  DUBFEMHandler<basis>(const DUBFEMHandler<basis> &DUBFEMHandler) = default;

  /// Default move constructor.
  DUBFEMHandler<basis>(DUBFEMHandler<basis> &&DUBFEMHandler) = default;

  /// Conversion of a discretized solution vector from Dubiner coefficients to
  /// FEM coefficients. The output FEM vector belongs to a space of order at
  /// most 2 due to the deal.II current availabilities. See dof_handler_DG.hpp
  /// description for more information.
  lifex::LinAlg::MPI::Vector
  dubiner_to_fem(const lifex::LinAlg::MPI::Vector &dub_solution) const;

  /// Same as dubiner_to_fem but allows to choose the grid refinement and
  /// polynomial order for the FE evaluations. This generalization overcomes the
  /// issue of a contour visualization that might be less refined than the
  /// numerical solution itself.
  lifex::LinAlg::MPI::Vector
  dubiner_to_fem(const lifex::LinAlg::MPI::Vector &dub_solution,
                 const unsigned int                n_ref_grid,
                 const std::string                &subsection,
                 const MPI_Comm                   &mpi_comm_,
                 const unsigned int                degree_fem = 1) const;

  /// Conversion of a discretized solution vector from FEM coefficients to
  /// Dubiner coefficients.
  lifex::LinAlg::MPI::Vector
  fem_to_dubiner(const lifex::LinAlg::MPI::Vector &fem_solution) const;

  /// Conversion of an analytical solution to a vector of Dubiner coefficients.
  lifex::LinAlg::MPI::Vector
  analytical_to_dubiner(
    lifex::LinAlg::MPI::Vector                           dub_solution,
    const std::shared_ptr<dealii::Function<lifex::dim>> &u_analytical) const;
};

template <class basis>
lifex::LinAlg::MPI::Vector
DUBFEMHandler<basis>::dubiner_to_fem(
  const lifex::LinAlg::MPI::Vector &dub_solution) const
{
  lifex::LinAlg::MPI::Vector fem_solution;
  fem_solution.reinit(dub_solution);


  // Due to the current deal.II availabilities, FE spaces can be of order at
  // most 2, see dof_handler_DG.hpp description. If the Dubiner space is of
  // higher order, this method generates a FEM vector of order 2, i.e. the
  // maximum possible. Hence, due to the possibility that DUB space and FEM
  // space have different orders, we need to specify two different degrees,
  // dof_handler, dofs_per_cell...
  const unsigned int degree_fem =
    (this->poly_degree < 3) ? this->poly_degree : 2;

  const dealii::FE_SimplexDGP<lifex::dim>      fe_dg(degree_fem);
  const std::vector<dealii::Point<lifex::dim>> support_points =
    fe_dg.get_unit_support_points();
  const unsigned int dofs_per_cell_fem = this->get_dofs_per_cell(degree_fem);

  // Generation of a new dof_handler for the FE evaluations.
  dealii::DoFHandler<lifex::dim> dof_handler_fem(
    dof_handler.get_triangulation());
  dof_handler_fem.distribute_dofs(fe_dg);

  std::vector<unsigned int> dof_indices(this->dofs_per_cell);
  std::vector<unsigned int> dof_indices_fem(dofs_per_cell_fem);


  // To perform the conversion to FEM, we just need to evaluate the linear
  // combination of Dubiner functions over the dof points.
  for (const auto &cell : dof_handler_fem.active_cell_iterators())
    {
      dof_indices = dof_handler.get_dof_indices(cell);
      cell->get_dof_indices(dof_indices_fem);

      for (unsigned int i = 0; i < dofs_per_cell_fem; ++i)
        {
          for (unsigned int j = 0; j < this->dofs_per_cell; ++j)
            {
              fem_solution[dof_indices_fem[i]] +=
                dub_solution[dof_indices[j]] *
                this->shape_value(j, support_points[i]);
            }
        }
    }

  return fem_solution;
}


template <class basis>
lifex::LinAlg::MPI::Vector
DUBFEMHandler<basis>::dubiner_to_fem(
  const lifex::LinAlg::MPI::Vector &dub_solution,
  const unsigned int                n_ref_grid,
  const std::string                &subsection,
  const MPI_Comm                   &mpi_comm_,
  const unsigned int                degree_fem) const
{
  // Creation of the new FEM evaluation mesh.
  lifex::utils::MeshHandler triangulation_fem(subsection, mpi_comm_);
  std::string mesh_path = "../meshes/" + std::to_string(lifex::dim) + "D_" +
                          std::to_string(n_ref_grid) + ".msh";
  AssertThrow(std::filesystem::exists(mesh_path),
              dealii::StandardExceptions::ExcMessage(
                "This mesh file/directory does not exist."));
  triangulation_fem.initialize_from_file(mesh_path, 1);
  triangulation_fem.set_element_type(
    lifex::utils::MeshHandler::ElementType::Tet);
  triangulation_fem.create_mesh();

  const dealii::FE_SimplexDGP<lifex::dim>      fe_dg(degree_fem);
  const std::vector<dealii::Point<lifex::dim>> support_points =
    fe_dg.get_unit_support_points();
  const unsigned int dofs_per_cell_fem = this->get_dofs_per_cell(degree_fem);

  // Generation of the mapping.
  const std::unique_ptr<dealii::MappingFE<lifex::dim>> mapping(
    std::make_unique<dealii::MappingFE<lifex::dim>>(fe_dg));

  // Generation of a new dof_handler for the FE evaluations.
  dealii::DoFHandler<lifex::dim> dof_handler_fem;
  dof_handler_fem.reinit(triangulation_fem.get());
  dof_handler_fem.distribute_dofs(fe_dg);

  // Initialization of the FEM evaluation vector.
  lifex::LinAlg::MPI::Vector fem_solution;
  dealii::IndexSet           owned_dofs = dof_handler_fem.locally_owned_dofs();
  fem_solution.reinit(owned_dofs, mpi_comm_);

  // Initialization of the dof_indices.
  std::vector<unsigned int> dof_indices(this->dofs_per_cell);
  std::vector<unsigned int> dof_indices_fem(dofs_per_cell_fem);

  // Tolerance.
  const double tol = 1e-10;

  // To perform the conversion to FEM, we just need to evaluate the linear
  // combination of Dubiner functions over the dof points.
  for (const auto &cell_fem : dof_handler_fem.active_cell_iterators())
    {
      cell_fem->get_dof_indices(dof_indices_fem);
      for (unsigned int i = 0; i < dofs_per_cell_fem; ++i)
        {
          dealii::Point<lifex::dim> real_support_point =
            mapping->transform_unit_to_real_cell(cell_fem, support_points[i]);

          for (const auto &cell : dof_handler.active_cell_iterators())
            {
              // This condition needs to guarantee that neighbor elements do not
              // both contribute to the FEM evaluation.
              if (fem_solution[dof_indices_fem[i]] < tol)
                {
                  dealii::Point<lifex::dim> unit_support_point_dub =
                    mapping->transform_real_to_unit_cell(cell,
                                                         real_support_point);
                  dof_indices = dof_handler.get_dof_indices(cell);
                  for (unsigned int j = 0; j < this->dofs_per_cell; ++j)
                    {
                      fem_solution[dof_indices_fem[i]] +=
                        (dub_solution[dof_indices[j]] *
                         this->shape_value(j, unit_support_point_dub));
                    }
                }
            }
        }
    }

  return fem_solution;
}

template <class basis>
lifex::LinAlg::MPI::Vector
DUBFEMHandler<basis>::fem_to_dubiner(
  const lifex::LinAlg::MPI::Vector &fem_solution) const
{
  const dealii::FE_SimplexDGP<lifex::dim> fe_dg(this->poly_degree);
  VolumeHandlerDG<lifex::dim>             vol_handler(this->poly_degree);

  lifex::LinAlg::MPI::Vector dub_solution;
  dub_solution.reinit(fem_solution);

  std::vector<unsigned int> dof_indices(this->dofs_per_cell);
  double                    eval_on_quad;


  // To perform the conversion to Dubiner, we just need to perform a (numerical)
  // L2 scalar product between the discretized solution and the Dubiner
  // functions. More precisely, thanks to the L2-orthonormality of the Dubiner
  // basis, the i-th coefficient w.r.t. the Dubiner basis is the L2 scalar
  // product between the solution and the i-th Dubiner function.
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      dof_indices = dof_handler.get_dof_indices(cell);
      vol_handler.reinit(cell);

      for (unsigned int i = 0; i < this->dofs_per_cell; ++i)
        {
          for (unsigned int q = 0; q < n_quad_points; ++q)
            {
              eval_on_quad = 0;

              for (unsigned int j = 0; j < this->dofs_per_cell; ++j)
                {
                  eval_on_quad +=
                    fem_solution[dof_indices[j]] *
                    fe_dg.shape_value(j, vol_handler.quadrature_ref(q));
                }

              dub_solution[dof_indices[i]] +=
                eval_on_quad *
                this->shape_value(i, vol_handler.quadrature_ref(q)) *
                vol_handler.quadrature_weight(q);
            }
        }
    }

  return dub_solution;
}

template <class basis>
lifex::LinAlg::MPI::Vector
DUBFEMHandler<basis>::analytical_to_dubiner(
  lifex::LinAlg::MPI::Vector                           dub_solution,
  const std::shared_ptr<dealii::Function<lifex::dim>> &u_analytical) const
{
  VolumeHandlerDG<lifex::dim> vol_handler(this->poly_degree);

  dealii::IndexSet owned_dofs = dof_handler.locally_owned_dofs();

  std::vector<unsigned int> dof_indices(this->dofs_per_cell);
  double                    eval_on_quad;

  // Here, we apply the same idea as in fem_to_dubiner. The only difference is
  // that here we can directly evaluate the solution on the quadrature point
  // since the solution is analytical.
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      dof_indices = dof_handler.get_dof_indices(cell);
      vol_handler.reinit(cell);

      for (unsigned int i = 0; i < this->dofs_per_cell; ++i)
        {
          dub_solution[dof_indices[i]] = 0;
          for (unsigned int q = 0; q < n_quad_points; ++q)
            {
              eval_on_quad =
                u_analytical->value(vol_handler.quadrature_real(q));

              dub_solution[dof_indices[i]] +=
                eval_on_quad *
                this->shape_value(i, vol_handler.quadrature_ref(q)) *
                vol_handler.quadrature_weight(q);
            }
        }
    }

  return dub_solution;
}

#endif /* DUBFEMHandler_HPP_*/
