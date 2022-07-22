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

#ifndef LAPLACE_DG_HPP_
#define LAPLACE_DG_HPP_

#include "lifex/core/core_model.hpp"
#include "lifex/core/init.hpp"

#include "lifex/utils/geometry/mesh_handler.hpp"

#include "lifex/utils/io/data_writer.hpp"

#include "lifex/utils/numerics/bc_handler.hpp"
#include "lifex/utils/numerics/linear_solver_handler.hpp"
#include "lifex/utils/numerics/preconditioner_handler.hpp"
#include "lifex/utils/numerics/tools.hpp"

#include <memory>
#include <vector>

#include "../source/DG_Assemble.hpp"
#include "../source/DG_Face_handler.hpp"
#include "../source/DG_Volume_handler.hpp"
#include "../source/DUBValues.hpp"
#include "../source/DUB_FEM_handler.hpp"
#include "../source/model_DG.hpp"

namespace lifex::examples
{
  namespace laplace_DG
  {
    /**
     * @brief Exact solution.
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
          return std::exp(p[0] + p[1]);
        else
          return std::exp(p[0] + p[1] + p[2]);
      }
    };

    /**
     * @brief Source term.
     */
    class RightHandSide : public Function<dim>
    {
    public:
      /// Constructor.
      RightHandSide()
        : Function<dim>()
      {}

      /// Evaluate the source term in a point.
      virtual double
      value(const Point<dim> &p,
            const unsigned int /*component*/ = 0) const override
      {
        if (dim == 2)
          return -2 * std::exp(p[0] + p[1]);
        else
          return -3 * std::exp(p[0] + p[1] + p[2]);
      }
    };

    /**
     * @brief Gradient of the exact solution.
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
   */

  template <class basis>
  class Laplace_DG : public ModelDG<basis>
  {
  public:
    /// Constructor.
    Laplace_DG<basis>()
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
  Laplace_DG<basis>::assemble_system()
  {
    this->matrix = 0;
    this->rhs    = 0;

    FullMatrix<double> V(this->dofs_per_cell, this->dofs_per_cell);
    FullMatrix<double> S(this->dofs_per_cell, this->dofs_per_cell);
    FullMatrix<double> I(this->dofs_per_cell, this->dofs_per_cell);
    FullMatrix<double> I_t(this->dofs_per_cell, this->dofs_per_cell);
    FullMatrix<double> IB(this->dofs_per_cell, this->dofs_per_cell);
    FullMatrix<double> IB_t(this->dofs_per_cell, this->dofs_per_cell);
    FullMatrix<double> IN(this->dofs_per_cell, this->dofs_per_cell);
    FullMatrix<double> IN_t(this->dofs_per_cell, this->dofs_per_cell);
    FullMatrix<double> SN(this->dofs_per_cell, this->dofs_per_cell);

    Vector<double>                       cell_rhs(this->dofs_per_cell);
    Vector<double>                       cell_rhs_edge(this->dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices(this->dofs_per_cell);

    for (const auto &cell : this->dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            this->assemble->reinit(cell);
            cell->get_dof_indices(dof_indices);

            V        = this->assemble->local_V();
            cell_rhs = this->assemble->local_rhs(this->f_ex);

            this->matrix.add(dof_indices, V);
            this->rhs.add(dof_indices, cell_rhs);

            for (const auto &edge : cell->face_indices())
              {
                this->assemble->reinit(cell, edge);
                std::vector<types::global_dof_index> dof_indices_neigh(
                  this->dofs_per_cell);

                S = this->assemble->local_S(this->prm_stability_coeff);
                this->matrix.add(dof_indices, S);

                if (!cell->at_boundary(edge))
                  {
                    std::tie(I, I_t) =
                      this->assemble->local_I(this->prm_penalty_coeff);
                    this->matrix.add(dof_indices, I);
                    this->matrix.add(dof_indices, I_t);

                    const auto neighcell = cell->neighbor(edge);
                    neighcell->get_dof_indices(dof_indices_neigh);

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
                      this->prm_stability_coeff, this->u_ex);
                    this->matrix.add(dof_indices, IB);
                    this->matrix.add(dof_indices, IB_t);
                    this->rhs.add(dof_indices, cell_rhs_edge);
                  }
              }
          }
      }

    this->matrix.compress(VectorOperation::add);
    this->rhs.compress(VectorOperation::add);
  }
} // namespace lifex::examples

#endif /* LAPLACE_DG_HPP_*/
