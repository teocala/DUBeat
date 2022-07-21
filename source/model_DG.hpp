/********************************************************************************
  Copyright (C) 2019 - 2022 by the lifex authors.

  This file is part of lifex.

  lifex is free software; you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  lifex is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with lifex.  If not, see <http://www.gnu.org/licenses/>.
********************************************************************************/

/**
 * @file
 *
 * @author Federica Botta <federica.botta@mail.polimi.it>.
 * @author Matteo Calafà <matteo.calafa@mail.polimi.it>.
 */

#ifndef ModelDG_HPP_
#define ModelDG_HPP_


#include "lifex/core/core_model.hpp"
#include "lifex/core/init.hpp"

#include "lifex/utils/geometry/mesh_handler.hpp"

#include "lifex/utils/io/data_writer.hpp"

#include "lifex/utils/numerics/bc_handler.hpp"
#include "lifex/utils/numerics/linear_solver_handler.hpp"
#include "lifex/utils/numerics/preconditioner_handler.hpp"
#include "lifex/utils/numerics/tools.hpp"

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/fe/mapping_q1_eulerian.h>

#include <deal.II/lac/full_matrix.h>

#include <memory>
#include <string>
#include <vector>

#include "DG_Assemble.hpp"
#include "DG_Face_handler.hpp"
#include "DG_Volume_handler.hpp"
#include "DG_error_parser.hpp"
#include "DUB_FEM_handler.hpp"

/**
 * @brief Class representing the resolution of problems using discontinuous Galerkin methods.
 */
template <class basis>
class ModelDG : public lifex::CoreModel
{
public:
  /// Constructor.
  ModelDG(std::string model_name)
    : CoreModel(model_name)
    , model_name(model_name)
    , triangulation(prm_subsection_path, mpi_comm)
    , linear_solver(prm_subsection_path + " / Linear solver",
                    {"CG", "GMRES", "BiCGStab"},
                    "GMRES")
    , preconditioner(prm_subsection_path + " / Preconditioner", true)
  {}

  /// Default copy constructor.
  ModelDG<basis>(ModelDG<basis> &ModelDG) = default;

  /// Default const copy constructor.
  ModelDG<basis>(const ModelDG<basis> &ModelDG) = default;

  /// Default move constructor.
  ModelDG<basis>(ModelDG<basis> &&ModelDG) = default;

  /// Declare main parameters.
  virtual void
  declare_parameters(lifex::ParamHandler &params) const override;

  /// Parse parameters from .prm file.
  virtual void
  parse_parameters(lifex::ParamHandler &params) override;

  /// Run the simulation.
  virtual void
  run() override;

  /// Destructor.
  virtual ~ModelDG() = default;

protected:
  /// To convert the final solution in FEM basis (does nothing if problem is in
  /// DGFEM).
  virtual void
  conversion_to_fem(lifex::LinAlg::MPI::Vector &sol_owned);

  /// To convert the initial solution in Dubiner basis (only for problems using
  /// Dubiner basis).
  virtual void
  conversion_to_dub(lifex::LinAlg::MPI::Vector &sol_owned);

  /// Conversion of an analytical solution from FEM to basis coefficients.
  virtual void
  discretize_analytical_solution(const std::shared_ptr<dealii::Function<lifex::dim>> &u_analytical,
                                 lifex::LinAlg::MPI::Vector &sol_owned);

  /// Setup of the problem before the resolution.
  virtual void
  setup_system();

  /// Assembly of the linear system, pure virtual.
  virtual void
  assemble_system() = 0;

  /// To compute errors at the end of system solving.
  void
  compute_errors(const lifex::LinAlg::MPI::Vector &solution_owned,
                 const lifex::LinAlg::MPI::Vector &solution_ex_owned,
                 const std::shared_ptr<dealii::Function<lifex::dim>> &u_ex,
                 const std::shared_ptr<dealii::Function<lifex::dim>> &grad_u_ex,
                 const char *solution_name) const;

  /// Creation of mesh, default path.
  void
  create_mesh();

  /// Creation of mesh.
  void
  create_mesh(std::string mesh_path);

  /// System solving.
  void
  solve_system();

  /// Output of results.
  void
  output_results() const;

  /// Name of the class/problem.
  const std::string model_name;
  /// Polynomials degree.
  unsigned int prm_fe_degree;
  /// Mesh refinement level (>=1).
  unsigned int prm_n_refinements;
  /// DG Penalty coefficient.
  double prm_penalty_coeff;
  /// DG stabilty coefficient.
  double prm_stability_coeff;
  /// Triangulation (internal use for useful already implemented methods).
  lifex::utils::MeshHandler triangulation;
  /// FE space (internal use for useful already implemented methods).
  std::unique_ptr<dealii::FE_SimplexDGP<lifex::dim>> fe;
  /// DoFHandler (internal use for useful already implemented methods).
  dealii::DoFHandler<lifex::dim> dof_handler;
  /// Matrix assembler.
  std::unique_ptr<DGAssemble<basis>> assemble;
  /// Linear solver handler.
  lifex::utils::LinearSolverHandler<lifex::LinAlg::MPI::Vector> linear_solver;
  /// Preconditioner handler.
  lifex::utils::PreconditionerHandler preconditioner;
  /// Distributed matrix of the linear system.
  lifex::LinAlg::MPI::SparseMatrix matrix;
  /// Distributed right hand side vector of the linear system.
  lifex::LinAlg::MPI::Vector rhs;
  /// Distributed solution vector, without ghost entries.
  lifex::LinAlg::MPI::Vector solution_owned;
  /// Distributed solution vector, with ghost entries.
  lifex::LinAlg::MPI::Vector solution;
  /// Distributed exact solution vector, without ghost entries.
  lifex::LinAlg::MPI::Vector solution_ex_owned;
  /// Distributed exact solution vector, without ghost entries.
  lifex::LinAlg::MPI::Vector solution_ex;
  /// Pointer to exact solution function.
  std::shared_ptr<lifex::utils::FunctionDirichlet> u_ex;
  /// Pointer to exact gradient solution Function
  std::shared_ptr<dealii::Function<lifex::dim>> grad_u_ex;
  /// Known forcing term.
  std::shared_ptr<dealii::Function<lifex::dim>> f_ex;
  /// Neumann boundary conditions.
  std::shared_ptr<dealii::Function<lifex::dim>> g_n;
};


template <class basis>
void
ModelDG<basis>::declare_parameters(lifex::ParamHandler &params) const
{
  // Default parameters.
  linear_solver.declare_parameters(params);
  preconditioner.declare_parameters(params);

  // Extra parameters.
  params.enter_subsection("Mesh and space discretization");
  {
    params.declare_entry(
      "Number of refinements",
      "3",
      dealii::Patterns::Integer(0),
      "Number of global mesh refinement steps applied to initial grid.");
    params.declare_entry("FE space degree",
                         "1",
                         dealii::Patterns::Integer(1),
                         "Degree of the FE space.");
  }
  params.leave_subsection();

  params.enter_subsection("Discontinuous Galerkin");
  {
    params.declare_entry(
      "Penalty coefficient",
      "-1",
      dealii::Patterns::Double(-1, 1),
      "Penalty coefficient in the Discontinuous Galerkin formulation.");
    params.declare_entry(
      "Stability coefficient",
      "10",
      dealii::Patterns::Double(0),
      "Stabilization term in the Discontinuous Galerkin formulation.");
  }
  params.leave_subsection();
}


template <class basis>
void
ModelDG<basis>::parse_parameters(lifex::ParamHandler &params)
{
  // Parse input file.
  params.parse();
  // Read input parameters.
  linear_solver.parse_parameters(params);
  preconditioner.parse_parameters(params);

  // Extra parameters.
  params.enter_subsection("Mesh and space discretization");
  prm_n_refinements = params.get_integer("Number of refinements");

  prm_fe_degree = params.get_integer("FE space degree");
  params.leave_subsection();

  params.enter_subsection("Discontinuous Galerkin");
  prm_penalty_coeff = params.get_double("Penalty coefficient");
  AssertThrow(prm_penalty_coeff == 1. || prm_penalty_coeff == 0. ||
                prm_penalty_coeff == -1.,
              dealii::StandardExceptions::ExcMessage(
                "Penalty coefficient must be 1 (SIP method) or 0 (IIP method) "
                "or -1 (NIP method)."));

  prm_stability_coeff = params.get_double("Stability coefficient");
  params.leave_subsection();
}


template <class basis>
void
ModelDG<basis>::run()
{
  create_mesh();
  setup_system();

  dealii::VectorTools::interpolate(dof_handler, *u_ex, solution_ex_owned);
  solution_ex = solution_ex_owned;

  // Initial guess.
  solution = solution_owned = 0;

  assemble_system();
  solve_system();

  conversion_to_fem(solution_owned);
  solution = solution_owned;

  compute_errors(solution_owned, solution_ex_owned, u_ex, grad_u_ex, "u");

  output_results();
}


template <class basis>
void
ModelDG<basis>::compute_errors(
  const lifex::LinAlg::MPI::Vector &                   solution_owned,
  const lifex::LinAlg::MPI::Vector &                   solution_ex_owned,
  const std::shared_ptr<dealii::Function<lifex::dim>> &u_ex,
  const std::shared_ptr<dealii::Function<lifex::dim>> &grad_u_ex,
  const char *                                         solution_name) const
{
  AssertThrow(u_ex != nullptr,
              dealii::StandardExceptions::ExcMessage(
                "Not valid pointer to the exact solution."));

  AssertThrow(grad_u_ex != nullptr,
              dealii::StandardExceptions::ExcMessage(
                "Not valid pointer to the gradient of the exact solution."));

  AssertThrow(solution_owned.size() == solution_ex_owned.size(),
              dealii::StandardExceptions::ExcMessage(
                "The exact solution vector and the approximate solution vector "
                "must have the same length."));

  std::cout << solution_name << " ERRORS: " << std::endl;
  std::vector<double> errors = {0, 0, 0, 0};


  // error L+inf
  lifex::LinAlg::MPI::Vector difference = solution_owned;
  difference -= solution_ex_owned;
  errors[0] = difference.linfty_norm();
  std::cout << "L-inf error norm: " << errors[0] << std::endl;

  // error L2
  double error_L2 = 0;

  DGVolumeHandler<lifex::dim> vol_handler(fe->degree);
  const unsigned int          n_q_points =
    static_cast<int>(std::pow(fe->degree + 2, lifex::dim));
  const unsigned int                          dofs_per_cell = fe->dofs_per_cell;
  std::vector<lifex::types::global_dof_index> dof_indices(dofs_per_cell);

  double local_approx;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      vol_handler.reinit(cell);

      if (cell->is_locally_owned())
        {
          const dealii::Tensor<2, lifex::dim> BJinv =
            vol_handler.get_jacobian_inverse();
          const double det = 1 / determinant(BJinv);

          cell->get_dof_indices(dof_indices);

          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              local_approx = 0;

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  local_approx +=
                    fe->shape_value(i, vol_handler.quadrature_ref(q)) *
                    solution_owned[dof_indices[i]];
                }

              error_L2 +=
                pow(local_approx - u_ex->value(vol_handler.quadrature_real(q)),
                    2) *
                det * vol_handler.quadrature_weight(q);
            }
        }
    }

  errors[1] = sqrt(error_L2);
  std::cout << "L-2 error norm: " << errors[1] << std::endl;

  // error H1
  double                        error_semi_H1 = 0;
  dealii::Tensor<1, lifex::dim> local_approx_gradient;
  dealii::Tensor<1, lifex::dim> local_grad_exact;
  dealii::Tensor<1, lifex::dim> pointwise_diff;
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      vol_handler.reinit(cell);

      if (cell->is_locally_owned())
        {
          const dealii::Tensor<2, lifex::dim> BJinv =
            vol_handler.get_jacobian_inverse();
          const double det = 1 / determinant(BJinv);

          cell->get_dof_indices(dof_indices);

          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              local_grad_exact      = 0;
              local_approx_gradient = 0;
              pointwise_diff        = 0;

              for (unsigned int j = 0; j < lifex::dim; ++j)
                {
                  local_grad_exact[j] =
                    grad_u_ex->value(vol_handler.quadrature_real(q), j);

                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      local_approx_gradient[j] +=
                        fe->shape_grad(i, vol_handler.quadrature_ref(q))[j] *
                        solution_owned[dof_indices[i]];
                    }
                }

              pointwise_diff =
                local_grad_exact - (local_approx_gradient * BJinv);

              error_semi_H1 += pointwise_diff * pointwise_diff * det *
                               vol_handler.quadrature_weight(q);
            }
        }
    }

  errors[2] = sqrt(error_L2 + error_semi_H1);
  std::cout << "H-1 error norm: " << errors[2] << std::endl;

  // Error DG
  double error_DG = 0;

  const unsigned int n_q_points_face =
    static_cast<int>(std::pow(fe->degree + 2, lifex::dim - 1));
  DGFaceHandler<lifex::dim> face_handler(fe->degree);
  DGFaceHandler<lifex::dim> face_handler_neigh(fe->degree);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      vol_handler.reinit(cell);

      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(dof_indices);

          for (const auto &edge : cell->face_indices())
            {
              face_handler.reinit(cell, edge);

              const double face_measure      = face_handler.get_measure();
              const double unit_face_measure = (4.0 - lifex::dim) / 2;
              const double measure_ratio     = face_measure / unit_face_measure;
              const double h_local           = (cell->measure()) / face_measure;

              const double local_stability_coeff =
                (prm_stability_coeff * pow(fe->degree, 2)) / h_local;

              std::vector<lifex::types::global_dof_index> dof_indices_neigh(
                dofs_per_cell);

              for (unsigned int q = 0; q < n_q_points_face; ++q)
                {
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                          error_DG +=
                            difference[dof_indices[i]] *
                            difference[dof_indices[j]] * local_stability_coeff *
                            fe->shape_value(i, face_handler.quadrature_ref(q)) *
                            fe->shape_value(j, face_handler.quadrature_ref(q)) *
                            face_handler.quadrature_weight(q) * measure_ratio;

                          if (!cell->at_boundary(edge))
                            {
                              const auto neighcell = cell->neighbor(edge);
                              const auto neighedge =
                                cell->neighbor_face_no(edge);
                              neighcell->get_dof_indices(dof_indices_neigh);
                              face_handler_neigh.reinit(neighcell, neighedge);

                              const unsigned int nq =
                                face_handler.corresponding_neigh_index(
                                  q, face_handler_neigh);

                              error_DG -=
                                difference[dof_indices[i]] *
                                difference[dof_indices_neigh[j]] *
                                local_stability_coeff *
                                fe->shape_value(
                                  i, face_handler.quadrature_ref(q)) *
                                fe->shape_value(
                                  j, face_handler_neigh.quadrature_ref(nq)) *
                                face_handler.quadrature_weight(q) *
                                measure_ratio;
                            }
                        }
                    }
                }
            }
        }
    }

  errors[3] = sqrt(error_semi_H1 + error_DG);
  std::cout << "DG error norm: " << errors[3] << std::endl << std::endl;

  error_parser::update_datafile(
    lifex::dim, prm_n_refinements, model_name, errors, solution_name);
}


template <class basis>
void
ModelDG<basis>::create_mesh()
{
  std::string mesh_path =
    "../meshes/" +
    std::to_string(lifex::dim) + "D_" + std::to_string(prm_n_refinements) +
    ".msh";
  AssertThrow(std::filesystem::exists(mesh_path),
              dealii::StandardExceptions::ExcMessage(
                "This mesh file/directory does not exist."));

  // Tetrahedral meshes can currently be imported only from file
  triangulation.initialize_from_file(mesh_path, 1);
  triangulation.set_element_type(lifex::utils::MeshHandler::ElementType::Tet);
  triangulation.create_mesh();
}


template <class basis>
void
ModelDG<basis>::create_mesh(std::string mesh_path)
{
  AssertThrow(std::filesystem::exists(mesh_path),
              dealii::StandardExceptions::ExcMessage(
                "This mesh file/directory does not exist."));

  // Tetrahedral meshes can currently be imported only from file
  triangulation.initialize_from_file(mesh_path, 1);
  triangulation.set_element_type(lifex::utils::MeshHandler::ElementType::Tet);
  triangulation.create_mesh();
}


template <class basis>
void
ModelDG<basis>::setup_system()
{
  fe       = std::make_unique<dealii::FE_SimplexDGP<lifex::dim>>(prm_fe_degree);
  assemble = std::make_unique<DGAssemble<basis>>(prm_fe_degree);

  dof_handler.reinit(triangulation.get());
  dof_handler.distribute_dofs(*fe);

  triangulation.get_info().print(prm_subsection_path,
                                 dof_handler.n_dofs(),
                                 true);

  dealii::IndexSet owned_dofs = dof_handler.locally_owned_dofs();

  dealii::IndexSet relevant_dofs;
  dealii::IndexSet active_dofs;
  lifex::DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
  lifex::DoFTools::extract_locally_active_dofs(dof_handler, active_dofs);

  dealii::DynamicSparsityPattern dsp(relevant_dofs);

  // Add (dof, dof_neigh) to dsp, so to the matrix
  const unsigned int                          dofs_per_cell = fe->dofs_per_cell;
  std::vector<lifex::types::global_dof_index> dof_indices(dofs_per_cell);
  std::vector<lifex::types::global_dof_index> dof_indices_neigh(dofs_per_cell);
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(dof_indices);

          for (const auto &edge : cell->face_indices())
            {
              if (!cell->at_boundary(edge))
                {
                  const auto neighcell = cell->neighbor(edge);
                  neighcell->get_dof_indices(dof_indices_neigh);

                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                          dsp.add(dof_indices[i], dof_indices_neigh[j]);
                        }
                    }
                }
            }
        }
    }

  lifex::DoFTools::make_sparsity_pattern(dof_handler, dsp);

  lifex::SparsityTools::distribute_sparsity_pattern(dsp,
                                                    owned_dofs,
                                                    mpi_comm,
                                                    relevant_dofs);

  lifex::utils::initialize_matrix(
    matrix, owned_dofs, dsp, preconditioner.uses_bddc(), active_dofs);

  rhs.reinit(owned_dofs, mpi_comm);

  solution_owned.reinit(owned_dofs, mpi_comm);
  solution.reinit(owned_dofs, relevant_dofs, mpi_comm);

  solution_ex_owned.reinit(owned_dofs, mpi_comm);
  solution_ex.reinit(owned_dofs, relevant_dofs, mpi_comm);
}


template <class basis>
void
ModelDG<basis>::output_results() const
{
  lifex::DataOut<lifex::dim> data_out;

  // Solutions.
  data_out.add_data_vector(dof_handler, solution, "u");
  data_out.build_patches();
  data_out.add_data_vector(dof_handler, solution_ex, "u_ex");
  data_out.build_patches();
  lifex::utils::dataout_write_hdf5(data_out, "solution", false);

  data_out.clear();
}


template <class basis>
void
ModelDG<basis>::solve_system()
{
  preconditioner.initialize(matrix);
  linear_solver.solve(matrix, solution_owned, rhs, preconditioner);
  solution = solution_owned;
}


/// Conversion of a discretized solution from Dubiner coefficients to FEM
/// coefficients. Useless if we are not using Dubiner basis functions.
template <class basis>
void
ModelDG<basis>::conversion_to_fem(lifex::LinAlg::MPI::Vector &sol_owned)
{
  return;
}


/// Conversion of a discretized solution from Dubiner coefficients to FEM
/// coefficients.
template <>
void
ModelDG<DUBValues<lifex::dim>>::conversion_to_fem(
  lifex::LinAlg::MPI::Vector &sol_owned)
{
  const DUBFEMHandler<lifex::dim> dub_fem_values(fe->degree, dof_handler);
  sol_owned = dub_fem_values.dubiner_to_fem(sol_owned);
}


/// Conversion of a discretized solution from FEM coefficients to Dubiner
/// coefficients. Useless if we are not using Dubiner basis functions.
template <class basis>
void
ModelDG<basis>::conversion_to_dub(lifex::LinAlg::MPI::Vector &sol_owned)
{
  return;
}


/// Conversion of a discretized solution from FEM coefficients to Dubiner
/// coefficients.
template <>
void
ModelDG<DUBValues<lifex::dim>>::conversion_to_dub(
  lifex::LinAlg::MPI::Vector &sol_owned)
{
  DUBFEMHandler<lifex::dim> dub_fem_values(fe->degree, dof_handler);
  sol_owned = dub_fem_values.fem_to_dubiner(sol_owned);
}


/// Conversion of an analytical solution from FEM to basis coefficients.
/// Specialization for FEM basis.
template <>
void
ModelDG<dealii::FE_SimplexDGP<lifex::dim>>::discretize_analytical_solution(
  const std::shared_ptr<dealii::Function<lifex::dim>> &u_analytical, lifex::LinAlg::MPI::Vector &sol_owned)
{
  dealii::VectorTools::interpolate(dof_handler, *u_analytical, sol_owned);
}


/// Conversion of an analytical solution from FEM to basis coefficients.
/// Specialization for Dubiner basis.
template <>
void
ModelDG<DUBValues<lifex::dim>>::discretize_analytical_solution(
  const std::shared_ptr<dealii::Function<lifex::dim>> &u_analytical, lifex::LinAlg::MPI::Vector &sol_owned)
{
  DUBFEMHandler<lifex::dim> dub_fem_values(fe->degree, dof_handler);
  sol_owned = dub_fem_values.analytical_to_dubiner(sol_owned, u_analytical);
}


#endif /* ModelDG_HPP_*/
