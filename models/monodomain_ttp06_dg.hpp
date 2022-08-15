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

#ifndef MONODOMAIN_TTP06_DG_HPP_
#define MONODOMAIN_TTP06_DG_HPP_

#include <math.h>

#include <memory>
#include <vector>

#include "../source/DG_Assemble.hpp"
#include "../source/DG_Face_handler.hpp"
#include "../source/DG_Volume_handler.hpp"
#include "../source/DUBValues.hpp"
#include "../source/DUB_FEM_handler.hpp"
#include "../source/QGaussLegendreSimplex.hpp"
#include "../source/model_DG.hpp"
#include "../source/model_DG_t.hpp"
#include "source/core_model.hpp"
#include "source/fiber_generation.hpp"
#include "source/geometry/mesh_handler.hpp"
#include "source/helpers/applied_current.hpp"
#include "source/helpers/ischemic_region.hpp"
#include "source/init.hpp"
#include "source/io/data_writer.hpp"
#include "source/ionic/ttp06.hpp"
#include "source/numerics/bc_handler.hpp"
#include "source/numerics/linear_solver_handler.hpp"
#include "source/numerics/preconditioner_handler.hpp"
#include "source/numerics/tools.hpp"

namespace lifex::examples
{
  namespace monodomain_ttp06_DG
  {
    /**
     * @brief Exact solution of the trans-membrane potential.
     */
    class ExactSolution : public utils::FunctionDirichlet
    {
    public:
      /// Constructor.
      ExactSolution()
        : utils::FunctionDirichlet()
      {}

      /// Evaluate the exact solution in a point.
      virtual double
      value(const Point<dim> &p,
            const unsigned int /*component*/ = 0) const override
      {
        if (dim == 2)
          return 0 * p[0] * p[1] * this->get_time();
        else
          return 0 * (p[0]) * (p[1]) * (p[2]) * (this->get_time());
      }
    };

    /**
     * @brief Neumann boundary condition of the trans-membrane potential.
     */
    class BCNeumann : public Function<dim>
    {
    public:
      /// Constrcutor.
      BCNeumann()
        : Function<dim>()
      {}

      /// Evaluate the Neumann boundary condition function in a point.
      virtual double
      value(const Point<dim> &p,
            const unsigned int /*component*/ = 0) const override
      {
        if (dim == 2)
          return 0 * p[0] * p[1] * this->get_time();
        else
          return 0 * p[0] * p[1] * p[2] * this->get_time();
      }
    };

    /**
     * @brief Gradient of the trans-membrane potential.
     */
    class GradExactSolution : public Function<dim>
    {
    public:
      /// Constructor.
      GradExactSolution()
        : Function<dim>()
      {}

      /// Evaluate the gradient of the exact solution in a point.
      virtual double
      value(const Point<dim>  &p,
            const unsigned int component = 0) const override
      {
        if (dim == 2)
          {
            if (component == 0) // x
              return 0 * p[0] * p[1] * this->get_time();
            else // y
              return 0 * p[0] * p[1] * p[2] * this->get_time();
          }
        else // dim=3
          {
            if (component == 0) // x.
              return 0 * p[0] * p[1] * p[2] * this->get_time();
            if (component == 1) // y.
              return 0 * p[0] * p[1] * p[2] * this->get_time();
            else // z.
              return 0 * p[0] * p[1] * p[2] * this->get_time();
          }
      }
    };
  } // namespace monodomain_ttp06_DG

  /**
   * @brief  Class to solve the monodomain equation with Fitzhugh-Nagumo ionic model for the cardiac electrophysiology
   * using the discontinuous Galerkin method.
   *
   * @f[
   * \begin{aligned}
   * \chi_m C_m \frac{\partial V_m}{\partial t} -\nabla \cdot (\Sigma \nabla
   * V_m)+\chi_m I_{ion}(V_m, w) &= I_{ext} & \quad & \text{in } \Omega = (-1,
   * 1)^d \times (0, T], \\
   * I_{ion}(V_m, w) &= k V_m (V_m - a)(V_m - 1) + w  & \quad & \text{in }
   * \Omega \times (0, T], \\
   * \frac{\partial w}{\partial t} &= \epsilon (V_m - \gamma w)  & \quad &
   * \text{in } \Omega \times (0, T], \\
   * \Sigma \frac{\partial V_m}{\partial n} &= g & \quad & \text{on }
   * \partial\Omega \times (0, T], \\ V_m &= V_{m_\mathrm{ex}} &
   * \quad & \text{in } \Omega \times  \{ t = 0 \}, \\
   * w &= w_\mathrm{ex} & \quad
   * & \text{in } \Omega \times  \{ t = 0 \} \end{aligned}
   * @f]
   * where @f$ I_{ion}(V_m, w)@f$ is defined through the FitzHugh-Nagumo model.
   *
   * In particular, it can be solved using the DGFEM basis
   * (basis=dealii::FE_SimplexDGP<lifex::dim>) or the Dubiner basis
   * (basis=DUBValues<lifex::dim>).
   *
   * The problem is time-discretized using the implicit finite difference scheme
   * @f$\mathrm{BDF}\sigma@f$ (where @f$\sigma = 1,2,...@f$
   * is the order of the BDF formula) as follows:
   * @f[
   * \begin{aligned}
   * \chi_m C_m ( \frac{\alpha_{\mathrm{BDF}\sigma} V_m^{n+1} -
   * V_{m_{\mathrm{BDF}\sigma}}^n}{\Delta t}) -\nabla \cdot (\Sigma \nabla
   * V_m^{n+1})+\chi_m I_{ion}(V_m^{n+1}, w^n) &= I_{ext}^{n+1} & \quad &
   * \text{in } \Omega = (-1, 1)^d \times \{n=0, \dots, N \}, \\
   * I_{ion}(V_m^{n+1}, w^n) &= k V_m^{n+1} (V_m^n - a)(V_m^n - 1) + w^n  &
   * \quad & \text{in } \Omega \times \{n=0, \dots, N \}, \\
   * \frac{\alpha_{\mathrm{BDF}\sigma} w^{n+1} -
   * w_{\mathrm{BDF}\sigma}^n}{\Delta t} &= \epsilon (V_m^{n+1} - \gamma w^n) &
   * \quad & \text{in } \Omega \times \{n=0, \dots, N \}, \\
   * \Sigma \frac{\partial V_m^{n+1}}{\partial n} &= g & \quad & \text{on }
   * \partial\Omega \times \{n=0, \dots, N \}, \\ V_m^0 &= V_{m_\mathrm{ex}} &
   * \quad & \text{in } \Omega \times  \{ n = 0 \}, \\
   * w^0 &= w_\mathrm{ex} &
   * \quad & \text{in } \Omega \times  \{ n = 0 \} \end{aligned}
   * @f]
   * where @f$\Delta t = t^{n+1}-t^{n}@f$ is the time step.
   *
   * Boundary conditions, initial conditions and source terms are provided
   * assuming that the exact solution is:
   *
   * @f[
   * \begin{alignat*}{5}
   * d=2: \: &V_{m_\mathrm{ex}}(x,y) &&= \sin(2\pi x)\sin(2\pi y)e^{-5t},
   * \hspace{6mm} &&(x,y) &&\in \Omega=(1,1)^2&&, t \in [0,T], \\
   *  &w_{\mathrm{ex}}(x,y) &&= \frac{\epsilon}{\epsilon\cdot\gamma -5}\sin(2\pi
   * x)\sin(2\pi y)e^{-5t}, \hspace{6mm} &&(x,y) &&\in \Omega=(1,1)^2&&, t \in
   * [0,T], \\
   * d=3: \: &V_{m_\mathrm{ex}}(x,y,z) &&= \sin\left(2\pi x +
   * \frac{\pi}{4}\right)\sin\left(2\pi y + \frac{\pi}{4}\right)\sin\left(2\pi z
   * + \frac{\pi}{4}\right) e^{-5t}, \hspace{6mm} &&(x,y,z) &&\in
   * \Omega=(1,1)^3&&, t \in [0,T], \\ &w_{\mathrm{ex}}(x,y,z) &&=
   * \frac{\epsilon}{\epsilon\cdot\gamma -5} \sin(2\pi x)\sin(2\pi y)\sin(2\pi
   * z) e^{-5t}, \hspace{6mm} &&(x,y,z) &&\in \Omega=(1,1)^3&&, t \in [0,T].
   * \end{alignat*}
   * @f]
   * Finally, @f$d@f$ is specified in the lifex configuration and @f$T@f$ as
   * well as the monodomain scalar parameters in the .prm parameter file.
   */

  template <class basis>
  class Monodomain_ttp06_DG : public ModelDG_t<basis>
  {
  public:
    /// Constructor.
    Monodomain_ttp06_DG<basis>()
      : ModelDG_t<basis>("Monodomain TTP06")
      , ChiM(1e5)
      , Sigma(0.12)
      , Cm(1e-2)
      , ionic_model(
          std::make_shared<lifex::TTP06>("Monodomain TTP06 / Ionic model",
                                         false))
      , fiber_generation("Fiber generation", false)
      , ischemic_region_generation("Monodomain TTP06 / Ischemic region")
      , I_app(std::make_shared<lifex::AppliedCurrent>(
          "Monodomain TTP06 / Applied current"))
    {
      this->u_ex = std::make_shared<monodomain_ttp06_DG::ExactSolution>();
      this->grad_u_ex =
        std::make_shared<monodomain_ttp06_DG::GradExactSolution>();
      this->g_n = std::make_shared<monodomain_ttp06_DG::BCNeumann>();
    }

  private:
    /// Monodomain equation parameter.
    double ChiM;
    /// Diffusion scalar parameter.
    double Sigma;
    /// Membrane capacity.
    double Cm;
    /// Ionic model
    std::shared_ptr<lifex::TTP06> ionic_model;
    /// FIber generation
    lifex::FiberGeneration fiber_generation;
    /// Transmural vector (0 in the endocardium, 1 in the epicardium).
    LinAlg::MPI::Vector endo_epi_vec;
    /// Ischemic region for both idealized and patient-specific cases.
    lifex::IschemicRegion ischemic_region_generation;
    /// Quadrature evaluation of ischemic region.
    std::unique_ptr<lifex::QuadratureIschemicRegion> ischemic_region_fun;
    /// Applied current
    std::shared_ptr<lifex::AppliedCurrent> I_app;

    /// Override for the simulation run.
    void
    run() override;

    /// Assembly of the Monodomain system.
    void
    assemble_system() override;

    /// Override for declaration of additional parameters.
    void
    declare_parameters(lifex::ParamHandler &params) const override;

    /// Override to parse additional parameters.
    void
    parse_parameters(lifex::ParamHandler &params) override;

    /// Setup for the ionic model
    void
    setup_ionic_model();

    /// To perform the time increment.
    void
    update_time() override;

    /// Setup for the time-dependent problems at time-step 0.
    void
    time_initialization() override;
  };

  template <class basis>
  void
  Monodomain_ttp06_DG<basis>::update_time()
  {
    // Update time for all the known analytical functions.
    this->u_ex->set_time(this->time);
    I_app->set_time(this->time);
    this->g_n->set_time(this->time);
    this->grad_u_ex->set_time(this->time);

    this->bdf_handler.time_advance(this->solution_owned, true);
    this->solution_bdf = this->bdf_handler.get_sol_bdf();

    // Update solution_ex_owned from the updated u_ex.
    this->discretize_analytical_solution(this->u_ex, this->solution_ex_owned);
  }

  template <class basis>
  void
  Monodomain_ttp06_DG<basis>::declare_parameters(
    lifex::ParamHandler &params) const
  {
    // Default parameters.
    this->linear_solver.declare_parameters(params);
    this->preconditioner.declare_parameters(params);
    ionic_model->declare_parameters(params);
    fiber_generation.declare_parameters(params);
    ischemic_region_generation.declare_parameters(params);
    I_app->declare_parameters(params);

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
  Monodomain_ttp06_DG<basis>::parse_parameters(lifex::ParamHandler &params)
  {
    // Parse input file.
    params.parse();

    // Read input parameters.
    this->linear_solver.parse_parameters(params);
    this->preconditioner.parse_parameters(params);
    ionic_model->parse_parameters(params);
    fiber_generation.parse_parameters(params);
    ischemic_region_generation.parse_parameters(params);
    I_app->parse_parameters(params);

    // Extra parameters.
    params.enter_subsection("Mesh and space discretization");
    this->prm_n_refinements = params.get_integer("Number of refinements");
    this->prm_fe_degree     = params.get_integer("FE space degree");
    params.leave_subsection();

    params.enter_subsection("Discontinuous Galerkin");
    this->prm_penalty_coeff = params.get_double("Penalty coefficient");
    AssertThrow(
      this->prm_penalty_coeff == 1. || this->prm_penalty_coeff == 0. ||
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
  Monodomain_ttp06_DG<basis>::time_initialization()
  {
    // Set initial time to the exact analytical solution.
    this->u_ex->set_time(this->prm_time_init);

    // Solution_owned and solution_ex_owned at the initial time are the
    // discretization of the analytical u_ex.
    this->discretize_analytical_solution(this->u_ex, this->solution_ex_owned);

    this->solution_owned = this->solution =
      ionic_model->setup_initial_transmembrane_potential();

    // Initialization of the initial solution.
    const std::vector<lifex::LinAlg::MPI::Vector> sol_init(
      this->prm_bdf_order, this->solution_owned);

    // Initialization of the BDFHandler
    this->bdf_handler.initialize(this->prm_bdf_order, sol_init);
  }

  template <class basis>
  void
  Monodomain_ttp06_DG<basis>::setup_ionic_model()
  {
    std::shared_ptr<QGaussLegendreSimplex<lifex::dim>> quadrature_formula(
      std::make_shared<QGaussLegendreSimplex<lifex::dim>>(this->prm_fe_degree +
                                                          2));

    fiber_generation.set_mesh_and_fe_space(this->triangulation,
                                           this->prm_fe_degree);
    fiber_generation.run();

    dealii::IndexSet owned_dofs    = this->dof_handler.locally_owned_dofs();
    dealii::IndexSet relevant_dofs = owned_dofs;
    endo_epi_vec.reinit(owned_dofs, relevant_dofs, this->mpi_comm);
    endo_epi_vec = fiber_generation.get_endo_epi();

    ischemic_region_generation.generate(this->prm_fe_degree,
                                        this->triangulation);
    ischemic_region_generation.postprocess_volumetric(
      fiber_generation.get_endo_epi_owned());
    ischemic_region_fun = std::make_unique<lifex::QuadratureIschemicRegion>(
      ischemic_region_generation, *quadrature_formula);
    ischemic_region_fun->init();

    const lifex::LinAlg::MPI::Vector &ischemic_region_vec =
      ischemic_region_generation.get_ischemic_region();

    std::map<dealii::types::material_id, std::string> map_id_volume;
    map_id_volume[0] = "Epicardium";

    ionic_model->initialize_3d(this->triangulation,
                               this->prm_fe_degree,
                               quadrature_formula,
                               I_app,
                               this->prm_bdf_order,
                               map_id_volume,
                               "Epicardium");
    ionic_model->setup_system(true);

    ionic_model->set_ischemic_region(
      &ischemic_region_vec,
      ischemic_region_generation.get_prm_scar_tolerance());
    ionic_model->set_endo_epi(&endo_epi_vec);
  }

  template <class basis>
  void
  Monodomain_ttp06_DG<basis>::run()
  {
    this->create_mesh();
    this->setup_system();

    setup_ionic_model();

    this->initialize_solution(this->solution_owned, this->solution);
    this->initialize_solution(this->solution_ex_owned, this->solution_ex);

    this->time_initialization();

    while (this->time < this->prm_time_final)
      {
        this->time += this->prm_time_step;
        ++this->timestep_number;

        pcout << "Time step " << std::setw(6) << this->timestep_number
              << " at t = " << std::setw(8) << std::fixed
              << std::setprecision(6) << this->time << std::endl;

        this->update_time();
        this->solution_ext = this->bdf_handler.get_sol_extrapolation();

        this->assemble_system();

        // Initial guess.
        this->solution_owned = this->solution_ext;
        this->solve_system();

        this->intermediate_error_print(this->solution_owned,
                                       this->solution_ex_owned,
                                       this->u_ex,
                                       "u");
      }


    this->compute_errors(this->solution_owned,
                         this->solution_ex_owned,
                         this->u_ex,
                         this->grad_u_ex,
                         "u");

    // Generation of the graphical output.
    if (this->prm_fe_degree < 3) // due to the current deal.II availabilities.
      {
        this->solution_ex = this->solution_ex_owned;
        this->conversion_to_fem(this->solution_ex);
        this->solution = this->solution_owned;
        this->conversion_to_fem(this->solution);
        this->output_results();
      }
    this->output_results();
  }

  template <class basis>
  void
  Monodomain_ttp06_DG<basis>::assemble_system()
  {
    ionic_model->solve_time_step(this->solution,
                                 this->prm_time_step,
                                 this->time);
    ionic_model->assemble_Iion_ICI(this->solution, this->solution_ext);
    ischemic_region_fun->init();
    ionic_model->init();

    const double &alpha_bdf = this->bdf_handler.get_alpha();

    this->matrix = 0;
    this->rhs    = 0;

    // The method is needed to define how the system matrix and rhs term are
    // defined for the monodomain problem with Fithugh-Nagumo ionic model. The
    // full matrix is composed by different sub-matrices that are called with
    // simple capital letters. We refer here to the DG_Assemble methods for
    // their definition.

    // See DG_Assemble::local_V().
    FullMatrix<double> V(this->dofs_per_cell, this->dofs_per_cell);
    // See DG_Assemble::local_M().
    FullMatrix<double> M(this->dofs_per_cell, this->dofs_per_cell);
    // See DG_Assemble::local_S().
    FullMatrix<double> S(this->dofs_per_cell, this->dofs_per_cell);
    // See DG_Assemble::local_I().
    FullMatrix<double> I(this->dofs_per_cell, this->dofs_per_cell);
    // Transpose of I.
    FullMatrix<double> I_t(this->dofs_per_cell, this->dofs_per_cell);
    // See DG_Assemble::local_IN().
    FullMatrix<double> IN(this->dofs_per_cell, this->dofs_per_cell);
    // Transpose of IN.
    FullMatrix<double> IN_t(this->dofs_per_cell, this->dofs_per_cell);
    // See DG_Assemble::local_SN().
    FullMatrix<double> SN(this->dofs_per_cell, this->dofs_per_cell);

    Vector<double>                       cell_rhs(this->dofs_per_cell);
    Vector<double>                       cell_rhs_ttp06(this->dofs_per_cell);
    Vector<double>                       cell_rhs_edge(this->dofs_per_cell);
    Vector<double>                       u0_rhs(this->dofs_per_cell);
    Vector<double>                       w0_rhs(this->dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices(this->dofs_per_cell);

    dealii::IndexSet owned_dofs = this->dof_handler.locally_owned_dofs();
    ;

    for (const auto &cell : this->dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            this->assemble->reinit(cell);
            dof_indices = this->dof_handler.get_dof_indices(cell);
            ischemic_region_fun->reinit(cell);
            ionic_model->reinit(cell);

            V = this->assemble->local_V();
            V *= Sigma;
            M = this->assemble->local_M();
            M /= this->prm_time_step;
            M *= alpha_bdf;
            M *= ChiM;
            M *= Cm;

            cell_rhs = this->assemble->local_rhs(I_app);

            cell_rhs_ttp06 = this->assemble->local_ttp06(ionic_model);
            cell_rhs_ttp06 *= ChiM;
            cell_rhs_ttp06 *= (-1);

            u0_rhs =
              this->assemble->local_u0_M_rhs(this->solution_bdf, dof_indices);
            u0_rhs /= this->prm_time_step;
            u0_rhs *= ChiM;
            u0_rhs *= Cm;

            this->matrix.add(dof_indices, V);
            this->matrix.add(dof_indices, M);
            this->rhs.add(dof_indices, cell_rhs_ttp06);
            this->rhs.add(dof_indices, cell_rhs);
            this->rhs.add(dof_indices, u0_rhs);

            for (const auto &edge : cell->face_indices())
              {
                this->assemble->reinit(cell, edge);
                std::vector<types::global_dof_index> dof_indices_neigh(
                  this->dofs_per_cell);

                if (!cell->at_boundary(edge))
                  {
                    S = this->assemble->local_S(this->prm_stability_coeff);
                    S *= Sigma;
                    std::tie(I, I_t) =
                      this->assemble->local_I(this->prm_penalty_coeff);
                    I *= Sigma;
                    I_t *= Sigma;
                    this->matrix.add(dof_indices, S);
                    this->matrix.add(dof_indices, I);
                    this->matrix.add(dof_indices, I_t);

                    const auto neighcell = cell->neighbor(edge);
                    dof_indices_neigh =
                      this->dof_handler.get_dof_indices(neighcell);

                    std::tie(IN, IN_t) =
                      this->assemble->local_IN(this->prm_penalty_coeff);
                    IN *= Sigma;
                    IN_t *= Sigma;
                    SN = this->assemble->local_SN(this->prm_stability_coeff);
                    SN *= Sigma;

                    this->matrix.add(dof_indices, dof_indices_neigh, IN);
                    this->matrix.add(dof_indices_neigh, dof_indices, IN_t);
                    this->matrix.add(dof_indices, dof_indices_neigh, SN);
                  }
                else
                  {
                    cell_rhs_edge =
                      this->assemble->local_rhs_edge_neumann(this->g_n);
                    this->rhs.add(dof_indices, cell_rhs_edge);
                  }
              }
          }
      }

    this->matrix.compress(VectorOperation::add);
    this->rhs.compress(VectorOperation::add);
  }
} // namespace lifex::examples

#endif /* MONODOMAIN_TTP06_DG_HPP_*/
