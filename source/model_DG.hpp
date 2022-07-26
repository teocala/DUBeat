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

#ifndef ModelDG_HPP_
#define ModelDG_HPP_

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
#include "compute_errors_DG.hpp"
#include "source/core_model.hpp"
#include "source/geometry/mesh_handler.hpp"
#include "source/init.hpp"
#include "source/io/data_writer.hpp"
#include "source/numerics/bc_handler.hpp"
#include "source/numerics/linear_solver_handler.hpp"
#include "source/numerics/preconditioner_handler.hpp"
#include "source/numerics/tools.hpp"

/**
 * @brief Class representing the resolution of problems using discontinuous
 * Galerkin methods.
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

  /// Return the number of degrees of freedom per element.
  unsigned int get_dof_per_cell() const;

  /// Run the simulation.
  virtual void
  run() override;

  /// Destructor.
  virtual ~ModelDG() = default;

protected:
  /// Setup of the problem before the resolution.
  virtual void
  setup_system();

  /// To inizialize the solutions using the deal.II reinit.
  void
  initialize_solution(lifex::LinAlg::MPI::Vector &solution_owned,
                      lifex::LinAlg::MPI::Vector &solution);

  /// Assembly of the linear system, pure virtual.
  virtual void
  assemble_system() = 0;

  /// Creation of mesh, default path.
  void
  create_mesh();

  /// Creation of mesh.
  void
  create_mesh(std::string mesh_path);

  /// System solving.
  void
  solve_system();

  /// To compute errors at the end of system solving.
  void
  compute_errors(const lifex::LinAlg::MPI::Vector &solution_owned,
                 const lifex::LinAlg::MPI::Vector &solution_ex_owned,
                 const std::shared_ptr<dealii::Function<lifex::dim>> &u_ex,
                 const std::shared_ptr<dealii::Function<lifex::dim>> &grad_u_ex,
                 const char *solution_name = (char *)"u") const;

  /// Output of results.
  void
  output_results() const;

  /// To convert a discretized solution in FEM basis (does nothing if problem is
  /// in DGFEM). In-place version.
  void
  conversion_to_fem(lifex::LinAlg::MPI::Vector &sol_owned);

  /// To convert a discretized solution in FEM basis (does nothing if problem is
  /// in DGFEM). Const version.
  lifex::LinAlg::MPI::Vector
  conversion_to_fem(const lifex::LinAlg::MPI::Vector &sol_owned) const;

  /// To convert a discretized solution in Dubiner basis (only for problems
  /// using Dubiner basis). In-place version.
  void
  conversion_to_dub(lifex::LinAlg::MPI::Vector &sol_owned);

  /// To convert a discretized solution in Dubiner basis (only for problems
  /// using Dubiner basis). Const version.
  lifex::LinAlg::MPI::Vector
  conversion_to_dub(const lifex::LinAlg::MPI::Vector &sol_owned) const;

  /// Conversion of an analytical solution from FEM to basis coefficients.
  void
  discretize_analytical_solution(
    const std::shared_ptr<dealii::Function<lifex::dim>> &u_analytical,
    lifex::LinAlg::MPI::Vector &                         sol_owned);

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
  /// Number of degrees of freedom per cell.
  unsigned int dofs_per_cell;
  /// DoFHandler (internal use for useful already implemented methods).
  dealii::DoFHandler<lifex::dim> dof_handler;
  /// Member used for conversions between analytical, nodal and modal
  /// representations of the solutions.
  std::shared_ptr<DUBFEMHandler<lifex::dim>> dub_fem_values;
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
unsigned int
ModelDG<basis>::get_dof_per_cell() const
{
  // The analytical formula is:
  // n_dof_per_cell = (p+1)*(p+2)*...(p+d) / d!,
  // where p is the space order and d the space dimension..

  unsigned int denominator = 1;
  unsigned int nominator = 1;

  for (unsigned int i = 1; i <= lifex::dim; i++)
  {
    denominator *= i;
    nominator *= prm_fe_degree+i;
  }

  return (int)(nominator/denominator);
}

template <class basis>
void
ModelDG<basis>::run()
{
  // Initialization
  create_mesh();
  setup_system();
  initialize_solution(solution_owned, solution);
  initialize_solution(solution_ex_owned, solution_ex);
  discretize_analytical_solution(u_ex, solution_ex_owned);
  solution_ex = solution_ex_owned;
  conversion_to_fem(solution_ex);

  // Initial guess.
  solution = solution_owned = 0;

  // Computation of the numerical solution.
  assemble_system();
  solve_system();

  // Provide FEM solution for the graphical output.
  solution = solution_owned;
  conversion_to_fem(solution);

  // Computation and output of the errors.
  compute_errors(solution_owned, solution_ex_owned, u_ex, grad_u_ex, "u");

  // Generation of the graphical output.
  output_results();
}

template <class basis>
void
ModelDG<basis>::setup_system()
{
  std::unique_ptr<dealii::FE_SimplexDGP<lifex::dim>> fe =
    std::make_unique<dealii::FE_SimplexDGP<lifex::dim>>(prm_fe_degree);
  assemble = std::make_unique<DGAssemble<basis>>(prm_fe_degree);

  dof_handler.reinit(triangulation.get());
  dof_handler.distribute_dofs(*fe);
  dub_fem_values =
    std::make_shared<DUBFEMHandler<lifex::dim>>(prm_fe_degree, dof_handler);

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
  dofs_per_cell = get_dof_per_cell();
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

  lifex::utils::initialize_matrix(matrix, owned_dofs, dsp);

  rhs.reinit(owned_dofs, mpi_comm);
}

template <class basis>
void
ModelDG<basis>::initialize_solution(lifex::LinAlg::MPI::Vector &solution_owned,
                                    lifex::LinAlg::MPI::Vector &solution)
{
  dealii::IndexSet owned_dofs = dof_handler.locally_owned_dofs();
  dealii::IndexSet relevant_dofs;
  dealii::IndexSet active_dofs;

  lifex::DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
  lifex::DoFTools::extract_locally_active_dofs(dof_handler, active_dofs);

  solution_owned.reinit(owned_dofs, mpi_comm);
  solution.reinit(owned_dofs, relevant_dofs, mpi_comm);
}

template <class basis>
void
ModelDG<basis>::create_mesh()
{
  std::string mesh_path = "../meshes/" + std::to_string(lifex::dim) + "D_" +
                          std::to_string(prm_n_refinements) + ".msh";
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
ModelDG<basis>::solve_system()
{
  preconditioner.initialize(matrix);
  linear_solver.solve(matrix, solution_owned, rhs, preconditioner);
  solution = solution_owned;
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
  Compute_Errors_DG<basis> error_calculator(prm_fe_degree,
                                            prm_stability_coeff,
                                            dofs_per_cell,
                                            dof_handler);
  error_calculator.reinit(
    solution_owned, solution_ex_owned, u_ex, grad_u_ex, solution_name);
  error_calculator.compute_errors();
  std::vector<double> errors = error_calculator.output_errors();

  pcout << std::endl
        << solution_name << ":" << std::endl
        << "Error L^inf: " << std::setw(6) << std::fixed << std::setprecision(6)
        << errors[0] << std::endl
        << "Error L^2:   " << std::setw(6) << std::fixed << std::setprecision(6)
        << errors[1] << std::endl
        << "Error H^1:   " << std::setw(6) << std::fixed << std::setprecision(6)
        << errors[2] << std::endl
        << "Error DG:    " << std::setw(6) << std::fixed << std::setprecision(6)
        << errors[3] << std::endl;

  error_parser::update_datafile(
    lifex::dim, prm_n_refinements, model_name, errors, solution_name);
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
  sol_owned = dub_fem_values->dubiner_to_fem(sol_owned);
}

/// Conversion to FEM coefficients, const version.
template <class basis>
lifex::LinAlg::MPI::Vector
ModelDG<basis>::conversion_to_fem(
  const lifex::LinAlg::MPI::Vector &sol_owned) const
{
  return sol_owned;
}

/// Conversion to FEM coefficients, const version.
template <>
lifex::LinAlg::MPI::Vector
ModelDG<DUBValues<lifex::dim>>::conversion_to_fem(
  const lifex::LinAlg::MPI::Vector &sol_owned) const
{
  lifex::LinAlg::MPI::Vector sol_fem =
    dub_fem_values->dubiner_to_fem(sol_owned);
  return sol_fem;
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
  sol_owned = dub_fem_values->fem_to_dubiner(sol_owned);
}

/// Conversion to DUB coefficients, const version.
template <class basis>
lifex::LinAlg::MPI::Vector
ModelDG<basis>::conversion_to_dub(
  const lifex::LinAlg::MPI::Vector &sol_owned) const
{
  return sol_owned;
}

/// Conversion to DUB coefficients, const version.
template <>
lifex::LinAlg::MPI::Vector
ModelDG<DUBValues<lifex::dim>>::conversion_to_dub(
  const lifex::LinAlg::MPI::Vector &sol_owned) const
{
  lifex::LinAlg::MPI::Vector sol_dub =
    dub_fem_values->fem_to_dubiner(sol_owned);
  return sol_dub;
}

/// Conversion of an analytical solution from FEM to basis coefficients.
/// Specialization for FEM basis.
template <>
void
ModelDG<dealii::FE_SimplexDGP<lifex::dim>>::discretize_analytical_solution(
  const std::shared_ptr<dealii::Function<lifex::dim>> &u_analytical,
  lifex::LinAlg::MPI::Vector &                         sol_owned)
{
  dealii::VectorTools::interpolate(dof_handler, *u_analytical, sol_owned);
}

/// Conversion of an analytical solution from FEM to basis coefficients.
/// Specialization for Dubiner basis.
template <>
void
ModelDG<DUBValues<lifex::dim>>::discretize_analytical_solution(
  const std::shared_ptr<dealii::Function<lifex::dim>> &u_analytical,
  lifex::LinAlg::MPI::Vector &                         sol_owned)
{
  sol_owned = dub_fem_values->analytical_to_dubiner(sol_owned, u_analytical);
}

#endif /* ModelDG_HPP_*/
