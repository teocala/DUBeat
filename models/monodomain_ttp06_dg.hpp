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

#ifndef MONODOMAIN_ttp06_DG_HPP_
#define MONODOMAIN_ttp06_DG_HPP_

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
#include "source/core_model.hpp"
#include "source/geometry/mesh_handler.hpp"
#include "source/init.hpp"
#include "source/io/data_writer.hpp"
#include "source/numerics/bc_handler.hpp"
#include "source/numerics/linear_solver_handler.hpp"
#include "source/numerics/preconditioner_handler.hpp"
#include "source/numerics/tools.hpp"
#include "source/ionic/ttp06.hpp"

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
      /// Parameter monodomain equation.
      double ChiM;

      /// Diffusion scalar parameter.
      double Sigma;

      /// Membrane capacity.
      double Cm;

      /// Factor for the nonlinear reaction in Fitzhugh Nagumo model.
      double kappa;

      /// Parameter ODE.
      double epsilon;

      /// Parameter ODE.
      double gamma;

      /// Parameter ODE.
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
      /// Diffusion scalar parameter.
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
            if (component == 0) // x.
              return 2 * M_PI * std::cos(2 * M_PI * p[0] + M_PI / 4) *
                     std::sin(2 * M_PI * p[1] + M_PI / 4) *
                     std::sin(2 * M_PI * p[2] + M_PI / 4) *
                     std::exp(-5 * this->get_time());
            if (component == 1) // y.
              return 2 * M_PI * std::sin(2 * M_PI * p[0] + M_PI / 4) *
                     std::cos(2 * M_PI * p[1] + M_PI / 4) *
                     std::sin(2 * M_PI * p[2] + M_PI / 4) *
                     std::exp(-5 * this->get_time());
            else // z.
              return 2 * M_PI * std::sin(2 * M_PI * p[0] + M_PI / 4) *
                     std::sin(2 * M_PI * p[1] + M_PI / 4) *
                     std::cos(2 * M_PI * p[2] + M_PI / 4) *
                     std::exp(-5 * this->get_time());
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
      : ModelDG_t<basis>("Monodomain ttp06")
      , ChiM(1e5)
      , Sigma(0.12)
      , Cm(1e-2)
      , kappa(19.5)
      , epsilon(1.2)
      , gamma(0.1)
      , a(13e-3)
    {
      this->u_ex = std::make_shared<monodomain_ttp06_DG::ExactSolution>();
      this->grad_u_ex =
        std::make_shared<monodomain_ttp06_DG::GradExactSolution>();
      this->f_ex = std::make_shared<monodomain_ttp06_DG::RightHandSide>(
        ChiM, Sigma, Cm, kappa, epsilon, gamma, a);
      this->g_n = std::make_shared<monodomain_ttp06_DG::BCNeumann>(Sigma);

      ionic_model = std::make_shared<lifex::TTP06> ("Monodomain ttp06", true);
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
    // /// Solution gating variable, without ghost entries.
    std::map<unsigned int, lifex::LinAlg::MPI::Vector> solution_owned_w;
    // /// Solution gating variable, with ghost entries.
    std::map<unsigned int, lifex::LinAlg::MPI::Vector> solution_w;
    /// BDF time advancing handler.
    std::map<unsigned int, lifex::utils::BDFHandler<lifex::LinAlg::MPI::Vector>> bdf_handler_w;
    /// BDF solution, with ghost entries.
    std::map<unsigned int, lifex::LinAlg::MPI::Vector>  solution_bdf_w;
    /// BDF extrapolated solution, with ghost entries.
    std::map<unsigned int, lifex::LinAlg::MPI::Vector>  solution_ext_w;
    /// Ionic model
    std::shared_ptr<lifex::TTP06> ionic_model;
    /// Applied current for fixed time without ghost entries
    lifex::LinAlg::MPI::Vector I_app_owned;
    /// Applied current for fixed time with ghost entries
    lifex::LinAlg::MPI::Vector I_app;


    /// Override for the simulation run.
    void
    run() override;

    /// Override to update time for both u and w.
    void
    update_time() override;

    /// Override to initialize both u and w.
    void
    time_initialization() override;

    /// Assembly of the Monodomain system.
    void
    assemble_system() override;
  };

  template <class basis>
  void
  Monodomain_ttp06_DG<basis>::run()
  {
    this->create_mesh();
    this->setup_system();

    ionic_model->setup_system_0d();

    this->initialize_solution(this->solution_owned, this->solution);
    this->initialize_solution(this->solution_ex_owned, this->solution_ex);
    this->initialize_solution(I_app_owned, I_app);

    for(unsigned int i = 0; i < 18; ++i)
    {
        this->initialize_solution(this->solution_owned_w[i], this->solution_w[i]);
    }

    time_initialization();

    while (this->time < this->prm_time_final)
      {
        this->time += this->prm_time_step;
        ++this->timestep_number;

        pcout << "Time step " << std::setw(6) << this->timestep_number
              << " at t = " << std::setw(8) << std::fixed
              << std::setprecision(6) << this->time << std::endl;

        this->update_time();

        this->solution_ext = this->bdf_handler.get_sol_extrapolation();

        for(unsigned int i = 0; i < 18; ++i)
        {
          this->solution_ext_w[i] = bdf_handler_w.at(i).get_sol_extrapolation();
        }

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
  Monodomain_ttp06_DG<basis>::update_time()
  {
    this->u_ex->set_time(this->time);
    this->f_ex->set_time(this->time);
    this->g_n->set_time(this->time);
    this->grad_u_ex->set_time(this->time);

    this->bdf_handler.time_advance(this->solution_owned, true);
    this->solution_bdf = this->bdf_handler.get_sol_bdf();

    for(unsigned int i = 0; i < 18; ++i)
    {
      bdf_handler_w.at(i).time_advance(solution_owned_w.at(i), true);
      solution_bdf_w[i] = this->bdf_handler_w.at(i).get_sol_bdf();
    }

    // Update solution_ex_owned from the updated u_ex.
    this->discretize_analytical_solution(this->u_ex, this->solution_ex_owned);
    this->discretize_analytical_solution(this->f_ex, I_app);
  }

  template <class basis>
  void
  Monodomain_ttp06_DG<basis>::time_initialization()
  {
    this->u_ex->set_time(this->prm_time_init);
    this->discretize_analytical_solution(this->u_ex, this->solution_ex_owned);
    this->solution_ex = this->solution_ex_owned;
    this->solution = this->solution_owned = this->solution_ex_owned;

    const std::vector<lifex::LinAlg::MPI::Vector> sol_init(
      this->prm_bdf_order, this->solution_owned);

    this->bdf_handler.initialize(this->prm_bdf_order, sol_init);

    for(unsigned int i = 0; i < 18; ++i)
    {
      solution_w[i] = solution_owned_w[i] =  ionic_model->setup_initial_conditions()[i];

      const std::vector<lifex::LinAlg::MPI::Vector> sol_init_w(
        this->prm_bdf_order, solution_owned_w.at(i));

      bdf_handler_w[i].initialize(this->prm_bdf_order, sol_init_w);
    }
  }

  template <class basis>
  void
  Monodomain_ttp06_DG<basis>::assemble_system()
  {
    const double &alpha_bdf = this->bdf_handler.get_alpha();

    for(unsigned int p = 0; p < this->dof_handler.n_dofs(); ++p)
    {
      std::vector<double> tmp_w_bdf;
      std::vector<double> tmp_w_ext;
      std::vector<double> w;

      for(unsigned int k = 0; k < 18; ++k)
      {
        tmp_w_bdf.push_back(solution_bdf_w.at(k)[p]);
        tmp_w_ext.push_back(solution_ext_w.at(k)[p]);
      }

      w = ionic_model->solve_time_step_0d(this->solution[p],
      this->prm_bdf_order,
      tmp_w_bdf,
      tmp_w_ext,
      0,
      1,
      I_app[p]).first;

      for(unsigned int k = 0; k < 18; ++k)
      {
        solution_w.at(k)[p] = w[k];
      }
    }

    solution_w = solution_owned_w;

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

    for (const auto &cell : this->dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            this->assemble->reinit(cell);
            dof_indices = this->dof_handler.get_dof_indices(cell);

            V = this->assemble->local_V();
            V *= Sigma;
            M = this->assemble->local_M();
            M /= this->prm_time_step;
            M *= alpha_bdf;
            M *= ChiM;
            M *= Cm;

            cell_rhs = this->assemble->local_rhs(this->f_ex);

            cell_rhs_ttp06 = this->assemble->local_ttp06(ionic_model, this->solution, this->solution_ext,
            solution_w, dof_indices);
            cell_rhs_ttp06 *= ChiM;
            cell_rhs_ttp06 *= (-1);

            u0_rhs =
              this->assemble->local_u0_M_rhs(this->solution_bdf, dof_indices);
            u0_rhs /= this->prm_time_step;
            u0_rhs *= ChiM;
            u0_rhs *= Cm;

            this->matrix.add(dof_indices, V);
            this->matrix.add(dof_indices, M);
            this->rhs.add(dof_indices, cell_rhs);
            this->rhs.add(dof_indices, cell_rhs);
            this->rhs.add(dof_indices, u0_rhs);
            this->rhs.add(dof_indices, w0_rhs);

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

#endif /* MONODOMAIN_ttp06_DG_HPP_*/