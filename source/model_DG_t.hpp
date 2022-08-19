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

#ifndef MODEL_DG_T_HPP_
#define MODEL_DG_T_HPP_

#include <deal.II/base/parameter_handler.h>

#include <deal.II/fe/mapping_q1_eulerian.h>

#include <deal.II/lac/full_matrix.h>

#include <memory>
#include <string>
#include <vector>

#include "DUBValues.hpp"
#include "DUB_FEM_handler.hpp"
#include "assemble_DG.hpp"
#include "face_handler_DG.hpp"
#include "model_DG.hpp"
#include "source/core_model.hpp"
#include "source/geometry/mesh_handler.hpp"
#include "source/init.hpp"
#include "source/io/data_writer.hpp"
#include "source/numerics/bc_handler.hpp"
#include "source/numerics/linear_solver_handler.hpp"
#include "source/numerics/preconditioner_handler.hpp"
#include "source/numerics/time_handler.hpp"
#include "source/numerics/tools.hpp"
#include "volume_handler_DG.hpp"

/**
 * @brief Class representing the resolution of time-dependent problems using
 * discontinuous Galerkin methods.
 */
template <class basis>
class ModelDG_t : public ModelDG<basis>
{
public:
  /// Constructor.
  ModelDG_t(std::string model_name)
    : ModelDG<basis>(model_name)
    , time(prm_time_init)
    , timestep_number(0)
  {}

  /// Default copy constructor.
  ModelDG_t<basis>(ModelDG_t<basis> &ModelDG_t) = default;

  /// Default const copy constructor.
  ModelDG_t<basis>(const ModelDG_t<basis> &ModelDG_t) = default;

  /// Default move constructor.
  ModelDG_t<basis>(ModelDG_t<basis> &&ModelDG_t) = default;

  /// Destructor.
  virtual ~ModelDG_t() = default;

protected:
  /// Override for declaration of additional parameters.
  virtual void
  declare_parameters(lifex::ParamHandler &params) const override;

  /// Override to parse additional parameters.
  virtual void
  parse_parameters(lifex::ParamHandler &params) override;

  /// Setup for the time-dependent problems at time-step 0.
  virtual void
  time_initialization();

  /// To perform the time increment.
  virtual void
  update_time();

  /// Computation of the errors at an intermediate time-step.
  virtual void
  intermediate_error_print(
    const lifex::LinAlg::MPI::Vector                    &solution_owned,
    const lifex::LinAlg::MPI::Vector                    &solution_ex_owned,
    const std::shared_ptr<dealii::Function<lifex::dim>> &u_ex,
    const char *solution_name = (char *)"u");

  /// Override for the simulation run.
  void
  run() override;

  /// Initial time.
  double prm_time_init;
  /// Final time.
  double prm_time_final;
  /// Time-step amplitude.
  double prm_time_step;
  /// BDF order.
  unsigned int prm_bdf_order;
  /// Current time.
  double time;
  /// Current time-step number.
  unsigned int timestep_number;
  /// BDF time advancing handler.
  lifex::utils::BDFHandler<lifex::LinAlg::MPI::Vector> bdf_handler;
  /// BDF solution, with ghost entries.
  lifex::LinAlg::MPI::Vector solution_bdf;
  /// BDF extrapolated solution, with ghost entries.
  lifex::LinAlg::MPI::Vector solution_ext;
};

template <class basis>
void
ModelDG_t<basis>::declare_parameters(lifex::ParamHandler &params) const
{
  // Default parameters.
  this->linear_solver.declare_parameters(params);
  this->preconditioner.declare_parameters(params);

  // Extra parameters.
  params.enter_subsection("Mesh and space discretization");
  {
    params.declare_entry(
      "Number of refinements",
      "2",
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

  params.enter_subsection("Time solver");
  {
    params.declare_entry("Initial time",
                         "0",
                         dealii::Patterns::Double(0),
                         "Initial time.");
    params.declare_entry("Final time",
                         "0.001",
                         dealii::Patterns::Double(0),
                         "Final time.");
    params.declare_entry("Time step",
                         "0.0001",
                         dealii::Patterns::Double(0),
                         "Time step.");
    params.declare_entry("BDF order",
                         "1",
                         dealii::Patterns::Integer(1, 3),
                         "BDF order: 1, 2, 3.");
  }
  params.leave_subsection();
}

template <class basis>
void
ModelDG_t<basis>::parse_parameters(lifex::ParamHandler &params)
{
  // Parse input file.
  params.parse();
  // Read input parameters.
  this->linear_solver.parse_parameters(params);
  this->preconditioner.parse_parameters(params);

  // Extra parameters.
  params.enter_subsection("Mesh and space discretization");
  this->prm_n_refinements = params.get_integer("Number of refinements");
  this->prm_fe_degree     = params.get_integer("FE space degree");
  params.leave_subsection();

  params.enter_subsection("Discontinuous Galerkin");
  this->prm_penalty_coeff = params.get_double("Penalty coefficient");
  AssertThrow(this->prm_penalty_coeff == 1. || this->prm_penalty_coeff == 0. ||
                this->prm_penalty_coeff == -1.,
              dealii::StandardExceptions::ExcMessage(
                "Penalty coefficient must be 1 (SIP method) or 0 (IIP method) "
                "or -1 (NIP method)."));

  this->prm_stability_coeff = params.get_double("Stability coefficient");
  params.leave_subsection();

  params.enter_subsection("Time solver");
  this->prm_time_init  = params.get_double("Initial time");
  this->prm_time_final = params.get_double("Final time");
  AssertThrow(this->prm_time_final > this->prm_time_init,
              dealii::StandardExceptions::ExcMessage(
                "Final time must be greater than initial time."));

  this->prm_time_step = params.get_double("Time step");
  this->prm_bdf_order = params.get_integer("BDF order");
  params.leave_subsection();
}

template <class basis>
void
ModelDG_t<basis>::time_initialization()
{
  // Set initial time to the exact analytical solution.
  this->u_ex->set_time(prm_time_init);

  // Solution_owned and solution_ex_owned at the initial time are the
  // discretization of the analytical u_ex.
  this->discretize_analytical_solution(this->u_ex, this->solution_owned);
  this->solution_ex_owned = this->solution_owned;

  // Initialization of the initial solution.
  const std::vector<lifex::LinAlg::MPI::Vector> sol_init(this->prm_bdf_order,
                                                         this->solution_owned);

  // Initialization of the BDFHandler
  bdf_handler.initialize(this->prm_bdf_order, sol_init);
}

template <class basis>
void
ModelDG_t<basis>::intermediate_error_print(
  const lifex::LinAlg::MPI::Vector                    &solution_owned,
  const lifex::LinAlg::MPI::Vector                    &solution_ex_owned,
  const std::shared_ptr<dealii::Function<lifex::dim>> &u_ex,
  const char                                          *solution_name)
{
  AssertThrow(u_ex != nullptr,
              dealii::StandardExceptions::ExcMessage(
                "Not valid pointer to the exact solution."));

  AssertThrow(solution_owned.size() == solution_ex_owned.size(),
              dealii::StandardExceptions::ExcMessage(
                "The exact solution vector and the approximate solution vector "
                "must have the same length."));

  lifex::LinAlg::MPI::Vector error_owned =
    this->conversion_to_fem(solution_owned);
  error_owned -= this->conversion_to_fem(solution_ex_owned);

  pcerr << solution_name << ":"
        << "\tL-inf error norm: " << error_owned.linfty_norm() << std::endl;
}

template <class basis>
void
ModelDG_t<basis>::update_time()
{
  // Update time for all the known analytical functions.
  this->u_ex->set_time(this->time);
  this->f_ex->set_time(this->time);
  this->g_n->set_time(this->time);
  this->grad_u_ex->set_time(this->time);

  bdf_handler.time_advance(this->solution_owned, true);
  this->solution_bdf = bdf_handler.get_sol_bdf();

  // Update solution_ex_owned from the updated u_ex.
  this->discretize_analytical_solution(this->u_ex, this->solution_ex_owned);
}

template <class basis>
void
ModelDG_t<basis>::run()
{
  this->create_mesh();
  this->setup_system();
  this->initialize_solution(this->solution_owned, this->solution);
  this->initialize_solution(this->solution_ex_owned, this->solution_ex);
  this->time_initialization();

  while (this->time < this->prm_time_final)
    {
      time += prm_time_step;
      ++timestep_number;

      pcout << "Time step " << std::setw(6) << timestep_number
            << " at t = " << std::setw(8) << std::fixed << std::setprecision(6)
            << time << std::endl;

      this->update_time();
      this->solution_ext = bdf_handler.get_sol_extrapolation();

      this->assemble_system();

      // Initial guess.
      this->solution_owned = this->solution_ext;
      this->solve_system();

      this->intermediate_error_print(this->solution_owned,
                                     this->solution_ex_owned,
                                     this->u_ex);
    }

  this->compute_errors(this->solution_owned,
                       this->solution_ex_owned,
                       this->u_ex,
                       this->grad_u_ex);


  // Generation of the graphical output.
  if (this->prm_fe_degree < 3) // due to the current deal.II availabilities.
    {
      this->solution_ex = this->solution_ex_owned;
      this->conversion_to_fem(this->solution_ex);
      this->solution = this->solution_owned;
      this->conversion_to_fem(this->solution);
      this->output_results();
    }
}

#endif /* MODEL_DG_T_HPP_*/
