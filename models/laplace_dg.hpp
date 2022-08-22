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

#ifndef LAPLACE_DG_HPP_
#define LAPLACE_DG_HPP_

#include <memory>
#include <vector>

#include "../source/DUBValues.hpp"
#include "../source/DUB_FEM_handler.hpp"
#include "../source/assemble_DG.hpp"
#include "../source/face_handler_DG.hpp"
#include "../source/model_DG.hpp"
#include "../source/volume_handler_DG.hpp"
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
  namespace laplace_DG
  {
    /**
     * @brief Exact solution.
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
          return std::exp(p[0] + p[1]);
        else
          return std::exp(p[0] + p[1] + p[2]);
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
        : Function<lifex::dim>()
      {}

      /// Evaluate the source term in a point.
      virtual double
      value(const dealii::Point<lifex::dim> &p,
            const unsigned int /*component*/ = 0) const override
      {
        if (lifex::dim == 2)
          return -2 * std::exp(p[0] + p[1]);
        else
          return -3 * std::exp(p[0] + p[1] + p[2]);
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
        : Function<lifex::dim>()
      {}

      /// Evaluate the gradient of the exact solution in a point.
      virtual double
      value(const dealii::Point<lifex::dim> &p,
            const unsigned int               component = 0) const override
      {
        if (lifex::dim == 2)
          {
            if (component == 0) // x
              return std::exp(p[0] + p[1]);
            else // y
              return std::exp(p[0] + p[1]);
          }
        else // dim=3
          {
            if (component == 0) // x
              return std::exp(p[0] + p[1] + p[2]);
            if (component == 1) // y
              return std::exp(p[0] + p[1] + p[2]);
            else // z
              return std::exp(p[0] + p[1] + p[2]);
          }
      }
    };
  } // namespace laplace_DG

  /**
   * @brief  Class to solve the Laplace equation using the Discontinuous Galerkin
   * method.
   *
   * @f[
   * \begin{aligned}
   * -\Delta u &= f & \quad & \text{in } \Omega = (-1, 1)^d, \\
   * u &= u_\mathrm{ex} & \quad & \text{on } \partial\Omega
   * \end{aligned}
   * @f]
   *
   * In particular, it can be solved using the FEM basis
   * (basis=dealii::FE_SimplexDGP<lifex::dim>) or the Dubiner basis
   * (basis=DUBValues<lifex::dim>).
   *
   * Boundary conditions and source terms are provided assuming that the exact
   * solution is:
   * @f[
   * \begin{aligned}
   * &d=2: \, u_\mathrm{ex}(x,y) = e^{x+y}, \hspace{6mm} (x,y) \in
   * \Omega=(1,1)^2, \\ &d=3: \, u_\mathrm{ex}(x,y,z) = e^{x+y+z}, \hspace{6mm}
   * (x,y,z) \in \Omega=(1,1)^3. \end{aligned}
   * @f]
   * Finally, @f$d@f$ is specified in the lifex configuration.
   */

  template <class basis>
  class LaplaceDG : public ModelDG<basis>
  {
  public:
    /// Constructor.
    LaplaceDG<basis>()
      : ModelDG<basis>("Laplace")
    {
      this->u_ex      = std::make_shared<laplace_DG::ExactSolution>();
      this->grad_u_ex = std::make_shared<laplace_DG::GradExactSolution>();
      this->f_ex      = std::make_shared<laplace_DG::RightHandSide>();
    }

  private:
    void
    assemble_system() override;
  };

  /// Assembly of the linear system.
  template <class basis>
  void
  LaplaceDG<basis>::assemble_system()
  {
    this->matrix = 0;
    this->rhs    = 0;

    // The method is needed to define how the system matrix and rhs term are
    // defined for the Laplace problem with Dirichlet boundary conditions. The
    // full matrix is composed by different sub-matrices that are called with
    // simple capital letters. We refer here to the DG_Assemble methods for
    // their definition.

    // See DG_Assemble::local_V().
    dealii::FullMatrix<double> V(this->dofs_per_cell, this->dofs_per_cell);
    // See DG_Assemble::local_SC().
    dealii::FullMatrix<double> SC(this->dofs_per_cell, this->dofs_per_cell);
    // See DG_Assemble::local_IC().
    dealii::FullMatrix<double> IC(this->dofs_per_cell, this->dofs_per_cell);
    // Transpose of IC.
    dealii::FullMatrix<double> IC_t(this->dofs_per_cell, this->dofs_per_cell);
    // See DG_Assemble::local_IB().
    dealii::FullMatrix<double> IB(this->dofs_per_cell, this->dofs_per_cell);
    // Transpose of IB.
    dealii::FullMatrix<double> IB_t(this->dofs_per_cell, this->dofs_per_cell);
    // See DG_Assemble::local_IN().
    dealii::FullMatrix<double> IN(this->dofs_per_cell, this->dofs_per_cell);
    // Transpose of IN.
    dealii::FullMatrix<double> IN_t(this->dofs_per_cell, this->dofs_per_cell);
    // See DG_Assemble::local_SN().
    dealii::FullMatrix<double> SN(this->dofs_per_cell, this->dofs_per_cell);

    // See DG_Assemble::local_rhs().
    dealii::Vector<double> cell_rhs(this->dofs_per_cell);
    // See DG_Assemble::local_rhs_edge_dirichlet().
    dealii::Vector<double> cell_rhs_edge(this->dofs_per_cell);

    std::vector<lifex::types::global_dof_index> dof_indices(
      this->dofs_per_cell);

    for (const auto &cell : this->dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            this->assemble->reinit(cell);
            dof_indices = this->dof_handler.get_dof_indices(cell);

            V        = this->assemble->local_V();
            cell_rhs = this->assemble->local_rhs(this->f_ex);

            this->matrix.add(dof_indices, V);
            this->rhs.add(dof_indices, cell_rhs);

            for (const auto &edge : cell->face_indices())
              {
                this->assemble->reinit(cell, edge);
                std::vector<lifex::types::global_dof_index> dof_indices_neigh(
                  this->dofs_per_cell);

                SC = this->assemble->local_SC(this->prm_stability_coeff);
                this->matrix.add(dof_indices, SC);

                if (!cell->at_boundary(edge))
                  {
                    std::tie(IC, IC_t) =
                      this->assemble->local_IC(this->prm_penalty_coeff);
                    this->matrix.add(dof_indices, IC);
                    this->matrix.add(dof_indices, IC_t);

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
                    std::tie(IB, IB_t) =
                      this->assemble->local_IB(this->prm_penalty_coeff);
                    cell_rhs_edge = this->assemble->local_rhs_edge_dirichlet(
                      this->prm_stability_coeff,
                      this->prm_penalty_coeff,
                      this->u_ex);
                    this->matrix.add(dof_indices, IB);
                    this->matrix.add(dof_indices, IB_t);
                    this->rhs.add(dof_indices, cell_rhs_edge);
                  }
              }
          }
      }

    this->matrix.compress(dealii::VectorOperation::add);
    this->rhs.compress(dealii::VectorOperation::add);
  }
} // namespace DUBeat::models

#endif /* LAPLACE_DG_HPP_*/
