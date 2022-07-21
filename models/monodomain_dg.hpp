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

#ifndef MONODOMAIN_DG_HPP_
#define MONODOMAIN_DG_HPP_

#include "lifex/core/core_model.hpp"
#include "lifex/core/init.hpp"

#include "lifex/utils/geometry/mesh_handler.hpp"

#include "lifex/utils/io/data_writer.hpp"

#include "lifex/utils/numerics/bc_handler.hpp"
#include "lifex/utils/numerics/linear_solver_handler.hpp"
#include "lifex/utils/numerics/preconditioner_handler.hpp"
#include "lifex/utils/numerics/tools.hpp"

#include <math.h>

#include <memory>
#include <vector>

#include "../source/DG_Assemble.hpp"
#include "../source/DG_Face_handler.hpp"
#include "../source/DG_Volume_handler.hpp"
#include "../source/DUBValues.hpp"
#include "../source/DUB_FEM_handler.hpp"
#include "../source/model_DG.hpp"
#include "../source/model_DG_t.hpp"

namespace lifex::examples
{
  namespace monodomain_DG
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
          return std::sin(2 * M_PI * p[0]) * std::sin(2 * M_PI * p[1]) *
                 std::exp(-5 * this->get_time());
        else
          return std::sin(2 * M_PI * p[0] + M_PI / 4) *
                 std::sin(2 * M_PI * p[1] + M_PI / 4) *
                 std::sin(2 * M_PI * p[2] + M_PI / 4) *
                 std::exp(-5 * this->get_time());
      }
    };

    /**
     * @brief Source term: applied current.
     */
    class RightHandSide : public Function<dim>
    {
    private:
      /// Parameter monodomain equation
      double ChiM;

      /// Diffusion scalar parameter
      double Sigma;

      /// Membrane capacity
      double Cm;

      /// Factor for the nonlinear reaction in Fitzhugh Nagumo model
      double kappa;

      /// Parameter ODE
      double epsilon;

      /// Parameter ODE
      double gamma;

      /// Parameter ODE
      double a;

    public:
      /// Constructor.
      RightHandSide(double ChiM,
                    double Sigma,
                    double Cm,
                    double kappa,
                    double epsilon,
                    double gamma,
                    double a)
        : Function<dim>()
        , ChiM(ChiM)
        , Sigma(Sigma)
        , Cm(Cm)
        , kappa(kappa)
        , epsilon(epsilon)
        , gamma(gamma)
        , a(a)
      {}

      /// Evaluate the source term in a point.
      virtual double
      value(const Point<dim> &p,
            const unsigned int /*component*/ = 0) const override
      {
        if (dim == 2)
          return std::sin(2 * M_PI * p[0]) * std::sin(2 * M_PI * p[1]) *
                 std::exp(-5 * this->get_time()) *
                 (-ChiM * Cm * 5 + Sigma * 8 * pow(M_PI, 2) +
                  ChiM * kappa *
                    (std::sin(2 * M_PI * p[0]) * std::sin(2 * M_PI * p[1]) *
                       std::exp(-5 * this->get_time()) -
                     a) *
                    (std::sin(2 * M_PI * p[0]) * std::sin(2 * M_PI * p[1]) *
                       std::exp(-5 * this->get_time()) -
                     1) +
                  ChiM * (epsilon / (epsilon * gamma - 5)));
        else
          return std::sin(2 * M_PI * p[0] + M_PI / 4) *
                 std::sin(2 * M_PI * p[1] + M_PI / 4) *
                 std::sin(2 * M_PI * p[2] + M_PI / 4) *
                 std::exp(-5 * this->get_time()) *
                 (-ChiM * Cm * 5 + Sigma * 12 * pow(M_PI, 2) +
                  ChiM * kappa *
                    (std::sin(2 * M_PI * p[0] + M_PI / 4) *
                       std::sin(2 * M_PI * p[1] + M_PI / 4) *
                       std::sin(2 * M_PI * p[2] + M_PI / 4) *
                       std::exp(-5 * this->get_time()) -
                     a) *
                    (std::sin(2 * M_PI * p[0] + M_PI / 4) *
                       std::sin(2 * M_PI * p[1] + M_PI / 4) *
                       std::sin(2 * M_PI * p[2] + M_PI / 4) *
                       std::exp(-5 * this->get_time()) -
                     1) +
                  ChiM * (epsilon / (epsilon * gamma - 5)));
      }
    };

    /**
     * @brief Neumann boundary condition of the trans-membrane potential.
     */
    class BCNeumann : public Function<dim>
    {
    private:
      /// Diffusion scalar parameter
      double Sigma;

    public:
      /// Constrcutor.
      BCNeumann(double Sigma)
        : Function<dim>()
        , Sigma(Sigma)
      {}

      /// Evaluate the Neumann boundary condition function in a point.
      virtual double
      value(const Point<dim> &p,
            const unsigned int /*component*/ = 0) const override
      {
        if (dim == 2)
          return Sigma *
                 (-2 * M_PI * std::sin(2 * M_PI * p[0]) *
                    std::cos(2 * M_PI * p[1]) *
                    std::exp(-5 * this->get_time()) * (std::abs(p[1]) < 1e-10) +
                  2 * M_PI * std::cos(2 * M_PI * p[0]) *
                    std::sin(2 * M_PI * p[1]) *
                    std::exp(-5 * this->get_time()) *
                    (std::abs(p[0] - 1) < 1e-10) +
                  2 * M_PI * std::sin(2 * M_PI * p[0]) *
                    std::cos(2 * M_PI * p[1]) *
                    std::exp(-5 * this->get_time()) *
                    (std::abs(p[1] - 1) < 1e-10) -
                  2 * M_PI * std::cos(2 * M_PI * p[0]) *
                    std::sin(2 * M_PI * p[1]) *
                    std::exp(-5 * this->get_time()) * (std::abs(p[0]) < 1e-10));
        else
          return Sigma *
                 (2 * M_PI * std::cos(2 * M_PI * p[0] + M_PI / 4) *
                    std::sin(2 * M_PI * p[1] + M_PI / 4) *
                    std::sin(2 * M_PI * p[2] + M_PI / 4) *
                    std::exp(-5 * this->get_time()) *
                    (std::abs(p[0] - 1) < 1e-10) -
                  2 * M_PI * std::cos(2 * M_PI * p[0] + M_PI / 4) *
                    std::sin(2 * M_PI * p[1] + M_PI / 4) *
                    std::sin(2 * M_PI * p[2] + M_PI / 4) *
                    std::exp(-5 * this->get_time()) * (std::abs(p[0]) < 1e-10) +
                  2 * M_PI * std::sin(2 * M_PI * p[0] + M_PI / 4) *
                    std::cos(2 * M_PI * p[1] + M_PI / 4) *
                    std::sin(2 * M_PI * p[2] + M_PI / 4) *
                    std::exp(-5 * this->get_time()) *
                    (std::abs(p[1] - 1) < 1e-10) -
                  2 * M_PI * std::sin(2 * M_PI * p[0] + M_PI / 4) *
                    std::cos(2 * M_PI * p[1] + M_PI / 4) *
                    std::sin(2 * M_PI * p[2] + M_PI / 4) *
                    std::exp(-5 * this->get_time()) * (std::abs(p[1]) < 1e-10) +
                  2 * M_PI * std::sin(2 * M_PI * p[0] + M_PI / 4) *
                    std::sin(2 * M_PI * p[1] + M_PI / 4) *
                    std::cos(2 * M_PI * p[2] + M_PI / 4) *
                    std::exp(-5 * this->get_time()) *
                    (std::abs(p[2] - 1) < 1e-10) -
                  2 * M_PI * std::sin(2 * M_PI * p[0] + M_PI / 4) *
                    std::sin(2 * M_PI * p[1] + M_PI / 4) *
                    std::cos(2 * M_PI * p[2] + M_PI / 4) *
                    std::exp(-5 * this->get_time()) * (std::abs(p[2]) < 1e-10));
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
      value(const Point<dim> & p,
            const unsigned int component = 0) const override
      {
        if (dim == 2)
          {
            if (component == 0) // x
              return 2 * M_PI * std::cos(2 * M_PI * p[0]) *
                     std::sin(2 * M_PI * p[1]) *
                     std::exp(-5 * this->get_time());
            else // y
              return 2 * M_PI * std::sin(2 * M_PI * p[0]) *
                     std::cos(2 * M_PI * p[1]) *
                     std::exp(-5 * this->get_time());
          }
        else // dim=3
          {
            if (component == 0) // x
              return 2 * M_PI * std::cos(2 * M_PI * p[0] + M_PI / 4) *
                     std::sin(2 * M_PI * p[1] + M_PI / 4) *
                     std::sin(2 * M_PI * p[2] + M_PI / 4) *
                     std::exp(-5 * this->get_time());
            if (component == 1) // y
              return 2 * M_PI * std::sin(2 * M_PI * p[0] + M_PI / 4) *
                     std::cos(2 * M_PI * p[1] + M_PI / 4) *
                     std::sin(2 * M_PI * p[2] + M_PI / 4) *
                     std::exp(-5 * this->get_time());
            else // z
              return 2 * M_PI * std::sin(2 * M_PI * p[0] + M_PI / 4) *
                     std::sin(2 * M_PI * p[1] + M_PI / 4) *
                     std::cos(2 * M_PI * p[2] + M_PI / 4) *
                     std::exp(-5 * this->get_time());
          }
      }
    };

    /**
     * @brief Exact solution of the gating variable.
     */
    class ExactSolution_w : public utils::FunctionDirichlet
    {
    private:
      /// Parameter ODE
      double epsilon;

      /// Parameter ODE
      double gamma;

    public:
      /// Constructor.
      ExactSolution_w(double epsilon, double gamma)
        : utils::FunctionDirichlet()
        , epsilon(epsilon)
        , gamma(gamma)
      {}

      /// Evaluate the exact solution in a point.
      virtual double
      value(const Point<dim> &p,
            const unsigned int /*component*/ = 0) const override
      {
        if (dim == 2)
          return epsilon / (epsilon * gamma - 5) * std::sin(2 * M_PI * p[0]) *
                 std::sin(2 * M_PI * p[1]) * std::exp(-5 * this->get_time());
        else
          return epsilon / (epsilon * gamma - 5) * std::sin(2 * M_PI * p[0]) *
                 std::sin(2 * M_PI * p[1]) * std::sin(2 * M_PI * p[2]) *
                 std::exp(-5 * this->get_time());
      }
    };

    /**
     * @brief Gradient of the gating variable.
     */
    class GradExactSolution_w : public Function<dim>
    {
    private:
      /// Parameter ODE
      double epsilon;

      /// Parameter ODE
      double gamma;

    public:
      /// Constructor.
      GradExactSolution_w(double epsilon, double gamma)
        : Function<dim>()
        , epsilon(epsilon)
        , gamma(gamma)
      {}

      /// Evaluate the gradient of the exact solution in a point.
      virtual double
      value(const Point<dim> & p,
            const unsigned int component = 0) const override
      {
        if (dim == 2)
          {
            if (component == 0) // x
              return 2 * M_PI * epsilon / (epsilon * gamma - 5) *
                     std::cos(2 * M_PI * p[0]) * std::sin(2 * M_PI * p[1]) *
                     std::exp(-5 * this->get_time());
            else // y
              return 2 * M_PI * epsilon / (epsilon * gamma - 5) *
                     std::sin(2 * M_PI * p[0]) * std::cos(2 * M_PI * p[1]) *
                     std::exp(-5 * this->get_time());
          }
        else // dim=3
          {
            if (component == 0) // x
              return 2 * M_PI * epsilon / (epsilon * gamma - 5) *
                     std::cos(2 * M_PI * p[0]) * std::sin(2 * M_PI * p[1]) *
                     std::sin(2 * M_PI * p[2]) *
                     std::exp(-5 * this->get_time());
            if (component == 1) // y
              return 2 * M_PI * epsilon / (epsilon * gamma - 5) *
                     std::sin(2 * M_PI * p[0]) * std::cos(2 * M_PI * p[1]) *
                     std::sin(2 * M_PI * p[2]) *
                     std::exp(-5 * this->get_time());
            else // z
              return 2 * M_PI * epsilon / (epsilon * gamma - 5) *
                     std::sin(2 * M_PI * p[0]) * std::sin(2 * M_PI * p[1]) *
                     std::cos(2 * M_PI * p[2]) *
                     std::exp(-5 * this->get_time());
          }
      }
    };
  } // namespace monodomain_DG

  /**
   * @brief  Class to solve the monodomain equation for the electrophysiology
   * problem using the Discontinuous Galerkin method.
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
   * Note that now the problem is linear.
   */

  template <class basis>
  class Monodomain_DG : public ModelDG_t<basis>
  {
  public:
    /// Constructor.
    Monodomain_DG<basis>()
      : ModelDG_t<basis>("Monodomain")
      , ChiM(1e5)
      , Sigma(0.12)
      , Cm(1e-2)
      , kappa(19.5)
      , epsilon(1.2)
      , gamma(0.1)
      , a(13e-3)
    {
      this->u_ex      = std::make_shared<monodomain_DG::ExactSolution>();
      this->grad_u_ex = std::make_shared<monodomain_DG::GradExactSolution>();
      this->f_ex      = std::make_shared<monodomain_DG::RightHandSide>(
        ChiM, Sigma, Cm, kappa, epsilon, gamma, a);
      this->g_n = std::make_shared<monodomain_DG::BCNeumann>(Sigma);
      w_ex = std::make_shared<monodomain_DG::ExactSolution_w>(epsilon, gamma);
      grad_w_ex =
        std::make_shared<monodomain_DG::GradExactSolution_w>(epsilon, gamma);
    }

  private:
    /// Monodomain equation parameter.
    double ChiM;
    /// Diffusion scalar parameter.
    double Sigma;
    /// Membrane capacity.
    double Cm;
    /// Factor for the nonlinear reaction in Fitzhugh Nagumo model.
    double kappa;
    /// ODE parameter.
    double epsilon;
    /// ODE parameter.
    double gamma;
    /// ODe parameter.
    double a;
    /// Solution gating variable, without ghost entries.
    lifex::LinAlg::MPI::Vector solution_owned_w;
    /// Solution gating variable, with ghost entries.
    lifex::LinAlg::MPI::Vector solution_w;
    /// Solution exact gating variable, without ghost entries.
    lifex::LinAlg::MPI::Vector solution_ex_owned_w;
    /// Solution exact gating variable, without ghost entries.
    lifex::LinAlg::MPI::Vector solution_ex_w;
    /// Pointer to exact solution function gating variable.
    std::shared_ptr<lifex::utils::FunctionDirichlet> w_ex;
    /// Pointer to exact gradient solution Function gating variable
    std::shared_ptr<dealii::Function<lifex::dim>> grad_w_ex;
    /// BDF time advancing handler.
    lifex::utils::BDFHandler<lifex::LinAlg::MPI::Vector> bdf_handler_w;
    /// BDF solution, with ghost entries.
    lifex::LinAlg::MPI::Vector solution_bdf_w;
    /// BDF extrapolated solution, with ghost entries.
    lifex::LinAlg::MPI::Vector solution_ext_w;

    /// Override for the simulation run.
    void
    run() override;

    /// Override to update time for both u and w.
    void
    update_time() override;

    /// Override to initialize both u and w.
    void
    time_initializaton() override;

    /// Assembly of the Monodomain system.
    void
    assemble_system() override;
  };

  template <class basis>
  void
  Monodomain_DG<basis>::run()
  {
    this->create_mesh();
    this->setup_system();
    time_initializaton();

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
        this->intermediate_error_print(this->solution_owned_w,
                                       this->solution_ex_owned_w,
                                       this->w_ex,
                                       "w");
      }

    this->conversion_to_fem(this->solution_owned);
    this->conversion_to_fem(this->solution_owned_w);
    this->solution   = this->solution_owned;
    this->solution_w = this->solution_owned_w;

    this->compute_errors(this->solution_owned,
                         this->solution_ex_owned,
                         this->u_ex,
                         this->grad_u_ex,
                         "u");
    this->compute_errors(this->solution_owned_w,
                         this->solution_ex_owned_w,
                         this->w_ex,
                         this->grad_w_ex,
                         "w");

    this->output_results();
  }

  template <class basis>
  void
  Monodomain_DG<basis>::update_time()
  {
    this->u_ex->set_time(this->time);
    this->f_ex->set_time(this->time);
    this->g_n->set_time(this->time);
    this->grad_u_ex->set_time(this->time);

    w_ex->set_time(this->time);
    grad_w_ex->set_time(this->time);

    this->bdf_handler.time_advance(this->solution_owned, true);
    this->solution_bdf = this->bdf_handler.get_sol_bdf();
    bdf_handler_w.time_advance(solution_owned_w, true);
    solution_bdf_w = this->bdf_handler_w.get_sol_bdf();

    dealii::VectorTools::interpolate(this->dof_handler,
                                     *(this->u_ex),
                                     this->solution_ex_owned);
    this->solution_ex = this->solution_ex_owned;
    dealii::VectorTools::interpolate(this->dof_handler,
                                     *(this->w_ex),
                                     this->solution_ex_owned_w);
    this->solution_ex_w = this->solution_ex_owned_w;
  }

  template <class basis>
  void
  Monodomain_DG<basis>::time_initializaton()
  {
    // Initialize BDF handler.
    dealii::IndexSet owned_dofs = this->dof_handler.locally_owned_dofs();
    dealii::IndexSet relevant_dofs;
    dealii::IndexSet active_dofs;
    lifex::DoFTools::extract_locally_relevant_dofs(this->dof_handler,
                                                   relevant_dofs);
    lifex::DoFTools::extract_locally_active_dofs(this->dof_handler,
                                                 active_dofs);

    solution_owned_w.reinit(owned_dofs, this->mpi_comm);
    solution_w.reinit(owned_dofs, relevant_dofs, this->mpi_comm);

    solution_ex_owned_w.reinit(owned_dofs, this->mpi_comm);
    solution_ex_w.reinit(owned_dofs, relevant_dofs, this->mpi_comm);

    w_ex->set_time(this->prm_time_init);
    lifex::VectorTools::interpolate(this->dof_handler,
                                    *w_ex,
                                    solution_ex_owned_w);
    solution_ex_w = solution_ex_owned_w;
    solution_w = solution_owned_w = solution_ex_owned_w;

    solution_w = solution_owned_w;

    this->u_ex->set_time(this->prm_time_init);
    lifex::VectorTools::interpolate(this->dof_handler,
                                    *this->u_ex,
                                    this->solution_ex_owned);
    this->solution_ex = this->solution_ex_owned;
    this->solution = this->solution_owned = this->solution_ex_owned;

    this->conversion_to_dub(this->solution_owned);
    this->conversion_to_dub(this->solution_owned_w);

    const std::vector<lifex::LinAlg::MPI::Vector> sol_init(
      this->prm_bdf_order, this->solution_owned);

    this->bdf_handler.initialize(this->prm_bdf_order, sol_init);

    const std::vector<lifex::LinAlg::MPI::Vector> sol_init_w(
      this->prm_bdf_order, solution_owned_w);

    bdf_handler_w.initialize(this->prm_bdf_order, sol_init_w);
  }

  template <class basis>
  void
  Monodomain_DG<basis>::assemble_system()
  {
    const double &alpha_bdf = this->bdf_handler.get_alpha();

    solution_owned_w *= -gamma;
    solution_owned_w.add(1, this->solution_owned);
    solution_owned_w *= epsilon;
    solution_owned_w.add(1 / this->prm_time_step, this->solution_bdf_w);
    solution_owned_w *= this->prm_time_step / alpha_bdf;

    solution_w = solution_owned_w;

    this->matrix = 0;
    this->rhs    = 0;

    const unsigned int dofs_per_cell = this->fe->dofs_per_cell;

    FullMatrix<double> V(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> M(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> S(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> I(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> I_t(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> IN(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> IN_t(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> SN(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> C(dofs_per_cell, dofs_per_cell);

    Vector<double>                       cell_rhs(dofs_per_cell);
    Vector<double>                       cell_rhs_edge(dofs_per_cell);
    Vector<double>                       u0_rhs(dofs_per_cell);
    Vector<double>                       w0_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    for (const auto &cell : this->dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            this->assemble->reinit(cell);
            cell->get_dof_indices(dof_indices);

            V = this->assemble->local_V();
            V *= Sigma;
            M = this->assemble->local_M();
            M /= this->prm_time_step;
            M *= alpha_bdf;
            M *= ChiM;
            M *= Cm;
            C = this->assemble->local_non_linear(this->solution_owned, a);
            C *= kappa;
            C *= ChiM;

            cell_rhs = this->assemble->local_rhs(this->f_ex);

            u0_rhs = this->assemble->local_u0_M_rhs(this->solution_bdf);
            u0_rhs /= this->prm_time_step;
            u0_rhs *= ChiM;
            u0_rhs *= Cm;

            w0_rhs = this->assemble->local_w0_M_rhs(solution_owned_w);
            w0_rhs *= ChiM;
            w0_rhs *= (-1);

            this->matrix.add(dof_indices, V);
            this->matrix.add(dof_indices, M);
            this->matrix.add(dof_indices, C);
            this->rhs.add(dof_indices, cell_rhs);
            this->rhs.add(dof_indices, u0_rhs);
            this->rhs.add(dof_indices, w0_rhs);

            for (const auto &edge : cell->face_indices())
              {
                this->assemble->reinit(cell, edge);
                std::vector<types::global_dof_index> dof_indices_neigh(
                  dofs_per_cell);

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
                    neighcell->get_dof_indices(dof_indices_neigh);

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

#endif /* MONODOMAIN_DG_HPP_*/
