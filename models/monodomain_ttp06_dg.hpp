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

#include "../3rdparty/lifex-dev_0d/ttp06.hpp"
#include "../source/DUBValues.hpp"
#include "../source/DUB_FEM_handler.hpp"
#include "../source/QGaussLegendreSimplex.hpp"
#include "../source/face_handler_DG.hpp"
#include "../source/model_DG.hpp"
#include "../source/model_DG_t.hpp"
#include "../source/volume_handler_DG.hpp"
#include "source/core_model.hpp"
#include "source/geometry/mesh_handler.hpp"
#include "source/helpers/applied_current.hpp"
#include "source/helpers/ischemic_region.hpp"
#include "source/init.hpp"
#include "source/io/data_writer.hpp"
#include "source/numerics/bc_handler.hpp"
#include "source/numerics/linear_solver_handler.hpp"
#include "source/numerics/preconditioner_handler.hpp"
#include "source/numerics/tools.hpp"

namespace DUBeat::models
{
  namespace monodomain_ttp06_DG
  {
    /**
     * @brief Exact solution of the trans-membrane potential.
     */
    class ExactSolution : public lifex::utils::FunctionDirichlet
    {
    public:
      /// Constructor.
      ExactSolution()
        : lifex::utils::FunctionDirichlet()
      {}

      /// Evaluate the exact solution in a point.
      virtual double
      value(const dealii::Point<lifex::dim> &p,
            const unsigned int /*component*/ = 0) const override
      {
        if (lifex::dim == 2)
          return -84e-3 + 0 * p[0] * p[1] * this->get_time();
        else
          return -84e-3 + 0 * (p[0]) * (p[1]) * (p[2]) * (this->get_time());
      }
    };

    /**
     * @brief Neumann boundary condition of the trans-membrane potential.
     */
    class BCNeumann : public lifex::Function<lifex::dim>
    {
    public:
      /// Constrcutor.
      BCNeumann()
        : lifex::Function<lifex::dim>()
      {}

      /// Evaluate the Neumann boundary condition function in a point.
      virtual double
      value(const dealii::Point<lifex::dim> &p,
            const unsigned int /*component*/ = 0) const override
      {
        if (lifex::dim == 2)
          return 0 * p[0] * p[1] * this->get_time();
        else
          return 0 * p[0] * p[1] * p[2] * this->get_time();
      }
    };

    /**
     * @brief Gradient of the trans-membrane potential.
     */
    class GradExactSolution : public lifex::Function<lifex::dim>
    {
    public:
      /// Constructor.
      GradExactSolution()
        : lifex::Function<lifex::dim>()
      {}

      /// Evaluate the gradient of the exact solution in a point.
      virtual double
      value(const dealii::Point<lifex::dim> &p,
            const unsigned int               component = 0) const override
      {
        if (lifex::dim == 2)
          {
            if (component == 0) // x
              return 0 * p[0] * p[1] * this->get_time();
            else // y
              return 0 * p[0] * p[1] * this->get_time();
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
   * @brief  Class to solve the monodomain equation with Fitzhugh-Nagumo ionic
   * model for the cardiac electrophysiology using the discontinuous Galerkin
   * method.
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
  class MonodomainTTP06DG : public ModelDG_t<basis>
  {
  public:
    /// Constructor.
    MonodomainTTP06DG<basis>()
      : ModelDG_t<basis>("Monodomain TTP06")
      , Sigma()
      , ionic_model(
          std::make_shared<lifex0d::TTP06>("Monodomain TTP06 / Ionic model"))
      , I_app(std::make_shared<lifex::AppliedCurrent>(
          "Monodomain TTP06 / Applied current"))
    {
      this->u_ex = std::make_shared<monodomain_ttp06_DG::ExactSolution>();
      this->grad_u_ex =
        std::make_shared<monodomain_ttp06_DG::GradExactSolution>();
      this->g_n = std::make_shared<monodomain_ttp06_DG::BCNeumann>();
    }

  private:
    /// Surface-to-volume ratio of cells parameter.
    double ChiM;
    /// Diffusion tensor parameter.
    dealii::Tensor<2, lifex::dim, double> Sigma;
    /// Membrane capacity.
    double Cm;
    /// Transversal conductivity @f$\left[\mathrm{m^2 s^{-1}}\right]@f$.
    double prm_sigma_m_t;
    /// Longitudinal conductivity @f$\left[\mathrm{m^2 s^{-1}}\right]@f$.
    double prm_sigma_m_l;
    /// Normal conductivity @f$\left[\mathrm{m^2 s^{-1}}\right]@f$.
    double prm_sigma_m_n;
    /// Bool to specify whether activation time has to be computed.
    bool prm_activation_time_compute;
    /// Ionic model
    std::shared_ptr<lifex0d::TTP06> ionic_model;
    /// Applied current
    std::shared_ptr<lifex::AppliedCurrent> I_app;
    /// Additional BDF handler for the gating variables.
    lifex::utils::BDFHandler<std::vector<std::vector<double>>> bdf_handler_w;
    /// Mesh file path.
    std::string mesh_path;
    /// Distributed right hand side vector associated with the ionic current.
    lifex::LinAlg::MPI::Vector rhs_ionic;
    /// ischemic region vector
    lifex::LinAlg::MPI::Vector ischemic_region;
    /// Maximum time derivative @f$\frac{\partial u}{\partial t}@f$ at each dof,
    /// used to compute the activation time, without ghost entries.
    lifex::LinAlg::MPI::Vector derivative_max_owned;
    /// Activation time vector.
    lifex::LinAlg::MPI::Vector activation_time;
    /// Activation time vector, without ghost entries.
    lifex::LinAlg::MPI::Vector activation_time_owned;

    /// Override for the simulation run.
    void
    run() override;

    /// Assembly of the Monodomain system.
    void
    assemble_system() override;

    /// Assembly of the ionic current term.
    void
    assemble_ionic();

    /// Override for declaration of additional parameters.
    void
    declare_parameters(lifex::ParamHandler &params) const override;

    /// Override to parse additional parameters.
    void
    parse_parameters(lifex::ParamHandler &params) override;

    /// To perform the time increment.
    void
    update_time() override;

    /// To print value max, min and menium of the potential.
    void
    intermediate_error_print() const;

    /// Compute the activation time, from lifex library.
    void compute_activation_time();

    /// Compute activation time in nine specific points: the eight vertices and
    /// the point at center of the volumetric domain.
    void activation_time_specific() const;
  };


  template <class basis>
  void
  MonodomainTTP06DG<basis>::update_time()
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
  MonodomainTTP06DG<basis>::declare_parameters(
    lifex::ParamHandler &params) const
  {
    // Default parameters.
    this->linear_solver.declare_parameters(params);
    this->preconditioner.declare_parameters(params);
    ionic_model->declare_parameters(params);
    I_app->declare_parameters(params);

    // Extra parameters.
    params.enter_subsection("File");
    {
      params.declare_entry("Filename",
                           "../meshes/slab_test_mesh.msh",
                           dealii::Patterns::Anything(),
                           "File msh for the mesh");
      params.declare_entry("Scaling factor",
                           "1",
                           dealii::Patterns::Double(0),
                           "Scaling factor");
    }
    params.leave_subsection();

    params.enter_subsection("Mesh and space discretization");
    {
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
        "1",
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
                           "0.01",
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

    params.enter_subsection("Parameters of the model");
    {
      params.declare_entry("ChiM",
                           "1",
                           dealii::Patterns::Double(0),
                           "Surface-to-volume ratio of cells parameter.");
      params.declare_entry("Cm",
                           "1",
                           dealii::Patterns::Double(0),
                           "Membrane capacity.");
      params.declare_entry("sigma_m_t",
                           "0.3494e-4",
                           dealii::Patterns::Double(),
                           "Transversal conductivity.");
      params.declare_entry("sigma_m_l",
                           "0.7643e-4",
                           dealii::Patterns::Double(),
                           "Longitudinal conductivity.");
      params.declare_entry("sigma_m_n",
                           "0.1125e-4",
                           dealii::Patterns::Double(),
                           "Normal conductivity.");
    }
    params.leave_subsection();

    params.enter_subsection("Activation time");
    {
      params.declare_entry("Compute",
                           "true",
                           dealii::Patterns::Bool(),
                           "To compute or not the activation time.");
    }
    params.leave_subsection();
  }


  template <class basis>
  void
  MonodomainTTP06DG<basis>::parse_parameters(lifex::ParamHandler &params)
  {
    // Parse input file.
    params.parse();

    // Read input parameters.
    this->linear_solver.parse_parameters(params);
    this->preconditioner.parse_parameters(params);
    ionic_model->parse_parameters(params);
    I_app->parse_parameters(params);

    // Extra parameters.
    params.enter_subsection("File");
    mesh_path            = params.get("Filename");
    this->scaling_factor = params.get_double("Scaling factor");
    params.leave_subsection();

    // Extra parameters.
    params.enter_subsection("Mesh and space discretization");
    this->prm_fe_degree = params.get_integer("FE space degree");
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

    params.enter_subsection("Parameters of the model");
    ChiM          = params.get_double("ChiM");
    Cm            = params.get_double("Cm");
    prm_sigma_m_t = params.get_double("sigma_m_t");
    prm_sigma_m_l = params.get_double("sigma_m_l");
    prm_sigma_m_n = params.get_double("sigma_m_n");

    params.leave_subsection();

    params.enter_subsection("Activation time");
    {
      prm_activation_time_compute = params.get_bool("Compute");
    }
    params.leave_subsection();
  }


  template <class basis>
  void
  MonodomainTTP06DG<basis>::run()
  {
    this->create_mesh(mesh_path);
    this->setup_system();

    this->initialize_solution(this->solution_owned, this->solution);
    this->initialize_solution(this->solution_ex_owned, this->solution_ex);

    this->time_initialization();
    std::vector<std::vector<double>> w;
    const int n_quad_points = std::pow(this->prm_fe_degree + 2, lifex::dim);

    for (const auto &cell : this->dof_handler.active_cell_iterators())
      {
        for (unsigned int i = 0; i < n_quad_points; ++i)
          {
            std::vector<double> w_quad(17);
            w_quad = ionic_model->setup_initial_conditions();
            w.push_back(w_quad);
          }
      }
    const std::vector<std::vector<std::vector<double>>> w_init(
      this->prm_bdf_order, w);
    bdf_handler_w.initialize(this->prm_bdf_order, w_init);

    dealii::IndexSet owned_dofs    = this->dof_handler.locally_owned_dofs();
    dealii::IndexSet relevant_dofs = owned_dofs;

    rhs_ionic.reinit(owned_dofs, this->mpi_comm);

    if (prm_activation_time_compute)
      {
        activation_time_owned.reinit(owned_dofs, this->mpi_comm);
        activation_time.reinit(owned_dofs, relevant_dofs, this->mpi_comm);
        derivative_max_owned.reinit(owned_dofs, this->mpi_comm);
      }

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
        this->assemble_ionic();

        this->rhs += rhs_ionic;

        // Initial guess.
        this->solution_owned = this->solution_ext;
        this->solve_system();

        compute_activation_time();

        intermediate_error_print();

        rhs_ionic.reinit(owned_dofs, this->mpi_comm);
      }

    this->compute_errors(this->solution_owned,
                         this->solution_ex_owned,
                         this->u_ex,
                         this->grad_u_ex,
                         "u");

    this->solution_ex = this->solution_ex_owned;
    this->conversion_to_fem(this->solution_ex,
                            mesh_path,
                            this->prm_fe_degree,
                            this->scaling_factor);
    this->solution = this->solution_owned;
    this->conversion_to_fem(this->solution,
                            mesh_path,
                            this->prm_fe_degree,
                            this->scaling_factor);
    this->output_results();

    activation_time_specific();
  }

  template <class basis>
  void
  MonodomainTTP06DG<basis>::assemble_system()
  {
    const double &alpha_bdf = this->bdf_handler.get_alpha();

    this->matrix = 0;
    this->rhs    = 0;

    Sigma[0][0] = prm_sigma_m_l;
    Sigma[1][1] = prm_sigma_m_t;

    if (lifex::dim == 3)
      {
        Sigma[2][2] = prm_sigma_m_n;
      }

    // The method is needed to define how the system matrix and rhs term are
    // defined for the monodomain problem with Fithugh-Nagumo ionic model. The
    // full matrix is composed by different sub-matrices that are called with
    // simple capital letters. We refer here to the DG_Assemble methods for
    // their definition.

    // See DG_Assemble::local_V().
    dealii::FullMatrix<double> V(this->dofs_per_cell, this->dofs_per_cell);
    // See DG_Assemble::local_M().
    dealii::FullMatrix<double> M(this->dofs_per_cell, this->dofs_per_cell);
    // See DG_Assemble::local_S().
    dealii::FullMatrix<double> S(this->dofs_per_cell, this->dofs_per_cell);
    // See DG_Assemble::local_I().
    dealii::FullMatrix<double> I(this->dofs_per_cell, this->dofs_per_cell);
    // Transpose of I.
    dealii::FullMatrix<double> I_t(this->dofs_per_cell, this->dofs_per_cell);
    // See DG_Assemble::local_IN().
    dealii::FullMatrix<double> IN(this->dofs_per_cell, this->dofs_per_cell);
    // Transpose of IN.
    dealii::FullMatrix<double> IN_t(this->dofs_per_cell, this->dofs_per_cell);
    // See DG_Assemble::local_SN().
    dealii::FullMatrix<double> SN(this->dofs_per_cell, this->dofs_per_cell);

    dealii::Vector<double> cell_rhs(this->dofs_per_cell);
    dealii::Vector<double> cell_rhs_edge(this->dofs_per_cell);
    dealii::Vector<double> u0_rhs(this->dofs_per_cell);
    dealii::Vector<double> w0_rhs(this->dofs_per_cell);
    std::vector<lifex::types::global_dof_index> dof_indices(
      this->dofs_per_cell);

    for (const auto &cell : this->dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            this->assemble->reinit(cell);
            dof_indices = this->dof_handler.get_dof_indices(cell);

            V = this->assemble->local_V(Sigma);
            M = this->assemble->local_M();
            M /= this->prm_time_step;
            M *= alpha_bdf;
            M *= ChiM;
            M *= Cm;

            cell_rhs = this->assemble->local_rhs(I_app);

            u0_rhs =
              this->assemble->local_u0_M_rhs(this->solution_bdf, dof_indices);
            u0_rhs /= this->prm_time_step;
            u0_rhs *= ChiM;
            u0_rhs *= Cm;

            this->matrix.add(dof_indices, V);
            this->matrix.add(dof_indices, M);
            this->rhs.add(dof_indices, cell_rhs);
            this->rhs.add(dof_indices, u0_rhs);

            for (const auto &edge : cell->face_indices())
              {
                this->assemble->reinit(cell, edge);
                std::vector<lifex::types::global_dof_index> dof_indices_neigh(
                  this->dofs_per_cell);

                if (!cell->at_boundary(edge))
                  {
                    S = this->assemble->local_SC(this->prm_stability_coeff,
                                                 Sigma);
                    std::tie(I, I_t) =
                      this->assemble->local_IC(this->prm_penalty_coeff, Sigma);
                    this->matrix.add(dof_indices, S);
                    this->matrix.add(dof_indices, I);
                    this->matrix.add(dof_indices, I_t);

                    const auto neighcell = cell->neighbor(edge);
                    dof_indices_neigh =
                      this->dof_handler.get_dof_indices(neighcell);

                    std::tie(IN, IN_t) =
                      this->assemble->local_IN(this->prm_penalty_coeff, Sigma);
                    SN = this->assemble->local_SN(this->prm_stability_coeff,
                                                  Sigma);

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

    this->matrix.compress(lifex::VectorOperation::add);
    this->rhs.compress(lifex::VectorOperation::add);
  }

  template <class basis>
  void
  MonodomainTTP06DG<basis>::assemble_ionic()
  {
    double det = 0;
    rhs_ionic  = 0;

    std::vector<unsigned int>   dof_indices(this->dofs_per_cell);
    dealii::Vector<double>      cell_rhs_ttp06(this->dofs_per_cell);
    VolumeHandlerDG<lifex::dim> vol_handler(this->prm_fe_degree);
    const int    n_quad_points = std::pow(this->prm_fe_degree + 2, lifex::dim);
    unsigned int n_cell        = 0;
    std::vector<std::vector<double>> w_vec(bdf_handler_w.get_sol_bdf());

    for (const auto &cell : this->dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            vol_handler.reinit(cell);
            dof_indices    = this->dof_handler.get_dof_indices(cell);
            cell_rhs_ttp06 = 0;

            for (unsigned int q = 0; q < n_quad_points; ++q)
              {
                double u_nodal = this->dub_fem_values->eval_on_point(
                  this->solution_owned,
                  dof_indices,
                  vol_handler.quadrature_ref(q));
                unsigned int i = n_cell * n_quad_points +
                                 q; // To represent the n_cells x n_quadrature
                                    // matrix as a vector.

                auto w = ionic_model->solve_time_step_0d(
                  u_nodal,
                  bdf_handler_w.get_alpha(),
                  bdf_handler_w.get_sol_bdf()[i],
                  bdf_handler_w.get_sol_extrapolation()[i],
                  0,
                  I_app->value(vol_handler.quadrature_real(q)));
                double Iion =
                  ionic_model->Iion(u_nodal, u_nodal, w.first, 0).first;

                det = 1 / determinant(vol_handler.get_jacobian_inverse());
                dof_indices   = this->dof_handler.get_dof_indices(cell);
                w_vec[i] = w.first;

                for (unsigned int ii = 0; ii < this->dofs_per_cell; ++ii)
                  {
                    cell_rhs_ttp06(ii) += Iion *
                                         this->dub_fem_values->shape_value(
                                           ii, vol_handler.quadrature_ref(q)) *
                                         vol_handler.quadrature_weight(q) * det;
                  }
              }
            cell_rhs_ttp06 *= ChiM;
            cell_rhs_ttp06 *= (-1);
            rhs_ionic.add(dof_indices, cell_rhs_ttp06);
          }
        n_cell++;
      }
    rhs_ionic.compress(lifex::VectorOperation::add);
    bdf_handler_w.time_advance(w_vec);
  }

  template <class basis>
  void
  MonodomainTTP06DG<basis>::intermediate_error_print() const
  {
    std::cout<<"Max value "<<this->solution.max()<<std::endl;
    std::cout<<"Min value "<<this->solution.min()<<std::endl;
    std::cout<<"Mean value "<<this->solution.mean_value()<<std::endl;
  }

  template <class basis>
  void MonodomainTTP06DG<basis>::compute_activation_time() {
    LinAlg::MPI::Vector solution_fem(this->solution);
    this->conversion_to_fem(solution_fem,
                            mesh_path,
                            this->prm_fe_degree,
                            this->scaling_factor);

  LinAlg::MPI::Vector solution_bdf_fem(this->solution_bdf);
  this->conversion_to_fem(solution_bdf_fem,
                          mesh_path,
                          this->prm_fe_degree,
                          this->scaling_factor);

    // Compute activation time.
    if (prm_activation_time_compute) {
      const double &alpha_bdf = this->bdf_handler.get_alpha();
      double derivative = 0;
      for (unsigned int idx = 0; idx < solution_fem.size(); ++idx) {
        derivative = (alpha_bdf * solution_fem[idx] - solution_bdf_fem[idx]) *
                     ionic_model->dt_fact / this->prm_time_step;

        if (((idx < solution_fem.size()) &&
             (derivative > derivative_max_owned[idx])) || // u
            ((idx >= solution_fem.size()) &&
             (derivative < derivative_max_owned[idx]))) // u_e
        {
          derivative_max_owned[idx] = derivative;

          activation_time_owned[idx] = this->time - this->prm_time_step;
        }
      }

      activation_time_owned.compress(lifex::VectorOperation::insert);
      activation_time = activation_time_owned;
    }
  }

  template <class basis>
  void MonodomainTTP06DG<basis>::activation_time_specific() const {
    if (!prm_activation_time_compute)
      return;

    std::vector<dealii::Point<lifex::dim>> points;
    points.push_back(dealii::Point<lifex::dim>(0.0, 0.0, 0.0));
    points.push_back(dealii::Point<lifex::dim>(0.0, 0.007, 0.0));
    points.push_back(dealii::Point<lifex::dim>(0.02, 0.0, 0.0));
    points.push_back(dealii::Point<lifex::dim>(0.02, 0.007, 0.0));
    points.push_back(dealii::Point<lifex::dim>(0.0, 0.0, 0.003));
    points.push_back(dealii::Point<lifex::dim>(0.0, 0.007, 0.003));
    points.push_back(dealii::Point<lifex::dim>(0.02, 0.0, 0.003));
    points.push_back(dealii::Point<lifex::dim>(0.02, 0.007, 0.003));
    points.push_back(dealii::Point<lifex::dim>(0.01, 0.0035, 0.0015));

    const dealii::FE_SimplexDGP<lifex::dim> fe_dg(this->prm_fe_degree);
    const std::unique_ptr<dealii::MappingFE<lifex::dim>> mapping(
        std::make_unique<dealii::MappingFE<lifex::dim>>(fe_dg));

    std::vector<unsigned int> dof_indices(this->dofs_per_cell);

    for (const auto &P : points) {
      double activation_time_point = 0;

      for (const auto &cell : this->dof_handler.active_cell_iterators()) {
        if (cell->is_locally_owned()) {
          if (cell->point_inside(P)) {
            dof_indices = this->dof_handler.get_dof_indices(cell);
            dealii::Point<lifex::dim> cell_unit =
                mapping->transform_real_to_unit_cell(cell, P);

            for (unsigned int i = 0; i < this->dofs_per_cell; ++i) {
              activation_time_point += activation_time[dof_indices[i]] *
                                       fe_dg.shape_value(i, cell_unit);
            }
          }
        }
      }

      std::cout << "For the point of coordinates " << P[0] << " " << P[1] << " "
                << P[2] << " the activation time is " << activation_time_point
                << " seconds" << std::endl;
    }
  }


} // namespace DUBeat::models

#endif /* MONODOMAIN_TTP06_DG_HPP_*/
