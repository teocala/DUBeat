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

#ifndef HEAT_DG_HPP_
#define HEAT_DG_HPP_

#include <math.h>

#include <memory>
#include <vector>

#include "../source/DG_Assemble.hpp"
#include "../source/DG_Face_handler.hpp"
#include "../source/DG_Volume_handler.hpp"
#include "../source/DUBValues.hpp"
#include "../source/DUB_FEM_handler.hpp"
#include "../source/model_DG_t.hpp"
#include "source/core_model.hpp"
#include "source/geometry/mesh_handler.hpp"
#include "source/init.hpp"
#include "source/io/data_writer.hpp"
#include "source/numerics/bc_handler.hpp"
#include "source/numerics/linear_solver_handler.hpp"
#include "source/numerics/preconditioner_handler.hpp"
#include "source/numerics/tools.hpp"

namespace DUBeat::models
{
  namespace heat_DG
  {
    /**
     * @brief Exact solution of the problem.
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
     * @brief Source term.
     */
    class RightHandSide : public lifex::Function<lifex::dim>
    {
    public:
      /// Constructor.
      RightHandSide()
        : lifex::Function<lifex::dim>()
      {}

      /// Evaluate the source term in a point.
      virtual double
      value(const dealii::Point<lifex::dim> &p,
            const unsigned int /*component*/ = 0) const override
      {
        if (lifex::dim == 2)
          return std::sin(2 * M_PI * p[0]) * std::sin(2 * M_PI * p[1]) *
                 std::exp(-5 * this->get_time()) * (8 * M_PI * M_PI - 5);
        else
          return std::sin(2 * M_PI * p[0] + M_PI / 4) *
                 std::sin(2 * M_PI * p[1] + M_PI / 4) *
                 std::sin(2 * M_PI * p[2] + M_PI / 4) *
                 std::exp(-5 * this->get_time()) * (12 * M_PI * M_PI - 5);
      }
    };

    /**
     * @brief Neumann boundary condition.
     */
    class BCNeumann : public lifex::Function<lifex::dim>
    {
    public:
      /// Constructor.
      BCNeumann()
        : lifex::Function<lifex::dim>()
      {}

      /// Evaluate the Neumann boundary condition function in a point.
      virtual double
      value(const dealii::Point<lifex::dim> &p,
            const unsigned int /*component*/ = 0) const override
      {
        if (lifex::dim == 2)
          return -2 * M_PI * std::sin(2 * M_PI * p[0]) *
                   std::cos(2 * M_PI * p[1]) * std::exp(-5 * this->get_time()) *
                   (std::abs(p[1]) < 1e-10) +
                 2 * M_PI * std::cos(2 * M_PI * p[0]) *
                   std::sin(2 * M_PI * p[1]) * std::exp(-5 * this->get_time()) *
                   (std::abs(p[0] - 1) < 1e-10) +
                 2 * M_PI * std::sin(2 * M_PI * p[0]) *
                   std::cos(2 * M_PI * p[1]) * std::exp(-5 * this->get_time()) *
                   (std::abs(p[1] - 1) < 1e-10) -
                 2 * M_PI * std::cos(2 * M_PI * p[0]) *
                   std::sin(2 * M_PI * p[1]) * std::exp(-5 * this->get_time()) *
                   (std::abs(p[0]) < 1e-10);
        else
          return 2 * M_PI * std::cos(2 * M_PI * p[0] + M_PI / 4) *
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
                   std::exp(-5 * this->get_time()) * (std::abs(p[2]) < 1e-10);
      }
    };

    /**
     * @brief Gradient of the exact solution.
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
  } // namespace heat_DG

  /**
   * @brief  Class to solve the heat equation using the Discontinuous Galerkin
   * method.
   *
   * @f[
   * \begin{aligned}
   * \frac{\partial u}{\partial t} -\Delta u &= f, & \quad & \text{in } \Omega =
   * (-1, 1)^d \times (0, T], \\
   * \frac{\partial u}{\partial n} &= g, & \quad & \text{on } \partial\Omega
   * \times (0, T], \\ u &= u_\mathrm{ex}, & \quad  & \text{in }
   * \Omega \times  \{ t = 0 \}. \end{aligned}
   * @f]
   *
   * In particular, it can be solved using the basis functions the FEM basis
   * (basis=dealii::FE_SimplexDGP<lifex::dim>)
   * or the Dubiner basis (basis=DUBValues<lifex::dim>).
   *
   * The problem is time-discretized using the implicit finite difference scheme
   * @f$\mathrm{BDF}\sigma@f$ (where @f$\sigma = 1,2,...@f$
   * is the order of the BDF formula) as follows:
   * @f[
   * \frac{\alpha_{\mathrm{BDF}\sigma} u^{n+1} -
   * u_{\mathrm{BDF}\sigma}^n}{\Delta t} - \Delta u^{n+1} + u^{n+1} = f^{n+1},
   * @f]
   * where @f$\Delta t = t^{n+1}-t^{n}@f$ is the time step.
   *
   * Boundary conditions, initial condition and source terms are provided
   * assuming that the exact solution is:
   * @f[
   * \begin{alignat*}{5}
   * &d=2: \, u_\mathrm{ex}(x,y) &&= \sin(2\pi x)\sin(2\pi y)e^{-5t},
   * \hspace{6mm} &&(x,y)  &&\in \Omega=(1,1)^2&&, t \in [0,T], \\ &d=3: \,
   * u_\mathrm{ex}(x,y,z) &&= \sin\left(2\pi x +
   * \frac{\pi}{4}\right)\sin\left(2\pi y + \frac{\pi}{4}\right)\sin\left(2\pi z
   * + \frac{\pi}{4}\right) e^{-5t}, \hspace{6mm} &&(x,y,z) &&\in
   * \Omega=(1,1)^3&&, t \in [0,T]. \end{alignat*}
   * @f]
   * Finally, @f$d@f$ is specified in the lifex configuration and @f$T@f$ in the
   * .prm parameter file.
   */

  template <class basis>
  class Heat_DG : public ModelDG_t<basis>
  {
  public:
    /// Constructor.
    Heat_DG<basis>()
      : ModelDG_t<basis>("Heat")
    {
      this->u_ex      = std::make_shared<heat_DG::ExactSolution>();
      this->grad_u_ex = std::make_shared<heat_DG::GradExactSolution>();
      this->f_ex      = std::make_shared<heat_DG::RightHandSide>();
      this->g_n       = std::make_shared<heat_DG::BCNeumann>();
    }

  private:
    void
    assemble_system() override;
  };

  /// Assembly of the linear system.
  template <class basis>
  void
  Heat_DG<basis>::assemble_system()
  {
    this->matrix = 0;
    this->rhs    = 0;

    // The method is needed to define how the system matrix and rhs term are
    // defined for the heat problem with Neumann boundary conditions. The full
    // matrix is composed by different sub-matrices that are called with simple
    // capital letters. We refer here to the DG_Assemble methods for their
    // definition.

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

    // See DG_Assemble::local_rhs().
    dealii::Vector<double> cell_rhs(this->dofs_per_cell);
    // See DG_Assemble::local_rhs_edge_neumann().
    dealii::Vector<double> cell_rhs_edge(this->dofs_per_cell);
    // See DG_Assemble::local_u0_M_rhs().
    dealii::Vector<double> u0_rhs(this->dofs_per_cell);

    std::vector<lifex::types::global_dof_index> dof_indices(
      this->dofs_per_cell);
    dealii::IndexSet owned_dofs = this->dof_handler.locally_owned_dofs();
    const double    &alpha_bdf  = this->bdf_handler.get_alpha();

    for (const auto &cell : this->dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            this->assemble->reinit(cell);
            dof_indices = this->dof_handler.get_dof_indices(cell);

            V = this->assemble->local_V();
            M = this->assemble->local_M();
            M /= this->prm_time_step;
            M *= alpha_bdf;

            cell_rhs = this->assemble->local_rhs(this->f_ex);
            u0_rhs =
              this->assemble->local_u0_M_rhs(this->solution_bdf, dof_indices);
            u0_rhs /= this->prm_time_step;

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
                    S = this->assemble->local_S(this->prm_stability_coeff);
                    std::tie(I, I_t) =
                      this->assemble->local_I(this->prm_penalty_coeff);
                    this->matrix.add(dof_indices, S);
                    this->matrix.add(dof_indices, I);
                    this->matrix.add(dof_indices, I_t);

                    const auto neighcell = cell->neighbor(edge);
                    dof_indices_neigh =
                      this->dof_handler.get_dof_indices(neighcell);

                    std::tie(IN, IN_t) =
                      this->assemble->local_IN(this->prm_penalty_coeff);
                    SN = this->assemble->local_SN(this->prm_stability_coeff);

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

    this->matrix.compress(dealii::VectorOperation::add);
    this->rhs.compress(dealii::VectorOperation::add);
  }
} // namespace DUBeat::models

#endif /* HEAT_DG_HPP_*/
