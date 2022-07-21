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

#ifndef DUBFEMHandler_HPP_
#define DUBFEMHandler_HPP_

#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping_q1_eulerian.h>

#include <deal.II/lac/trilinos_vector.h>

#include <cmath>
#include <vector>

#include "DUBValues.hpp"


/**
 * @brief Class to apply the conversions of a discretized solution between Dubiner basis and FEM basis representations.
 */
template <unsigned int dim>
class DUBFEMHandler : public DUBValues<dim>
{
private:
  /// Dof handler object of the problem.
  const dealii::DoFHandler<dim> &dof_handler;

  /// Number of quadrature points in the volume element.
  /// By default: @f$(degree+2)^{dim}@f$.
  const unsigned int n_quad_points;

public:
  /// Constructor.
  DUBFEMHandler<dim>(const unsigned int             degree,
                     const dealii::DoFHandler<dim> &dof_hand)
    : DUBValues<dim>(degree)
    , dof_handler(dof_hand)
    , n_quad_points(static_cast<int>(std::pow(degree + 2, dim)))
  {}

  /// Default copy constructor.
  DUBFEMHandler<dim>(DUBFEMHandler<dim> &DUBFEMHandler) = default;

  /// Default const copy constructor.
  DUBFEMHandler<dim>(const DUBFEMHandler<dim> &DUBFEMHandler) = default;

  /// Default move constructor.
  DUBFEMHandler<dim>(DUBFEMHandler<dim> &&DUBFEMHandler) = default;

  /// Conversion of a discretized solution from Dubiner coefficients to FEM
  /// coefficients.
  lifex::LinAlg::MPI::Vector
  dubiner_to_fem(const lifex::LinAlg::MPI::Vector &dub_solution) const;

  /// Conversion of a discretized solution from FEM coefficients to Dubiner
  /// coefficients.
  lifex::LinAlg::MPI::Vector
  fem_to_dubiner(const lifex::LinAlg::MPI::Vector &fem_solution);

  /// Conversion of an analytical solution to a vector of Dubiner coefficients.
  lifex::LinAlg::MPI::Vector
  analytical_to_dubiner(lifex::LinAlg::MPI::Vector dub_solution,
    const std::shared_ptr<dealii::Function<lifex::dim>> &u_analytical);
};


template <unsigned int dim>
lifex::LinAlg::MPI::Vector
DUBFEMHandler<dim>::dubiner_to_fem(
  const lifex::LinAlg::MPI::Vector &dub_solution) const
{
  lifex::LinAlg::MPI::Vector fem_solution;
  fem_solution.reinit(dub_solution);

  std::vector<unsigned int> dof_indices(this->n_functions);

  const dealii::FE_SimplexDGP<dim>      fe_dg(this->poly_degree);
  const std::vector<dealii::Point<dim>> support_points =
    fe_dg.get_unit_support_points();

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell->get_dof_indices(dof_indices);

      for (unsigned int i = 0; i < this->n_functions; ++i)
        {
          for (unsigned int j = 0; j < this->n_functions; ++j)
            {
              fem_solution[dof_indices[i]] +=
                dub_solution[dof_indices[j]] *
                this->shape_value(j, support_points[i]);
            }
        }
    }

  return fem_solution;
}


template <unsigned int dim>
lifex::LinAlg::MPI::Vector
DUBFEMHandler<dim>::fem_to_dubiner(
  const lifex::LinAlg::MPI::Vector &fem_solution)
{
  const dealii::FE_SimplexDGP<dim> fe_dg(this->poly_degree);
  DGVolumeHandler<dim>             vol_handler(this->poly_degree);

  lifex::LinAlg::MPI::Vector dub_solution;
  dub_solution.reinit(fem_solution);

  std::vector<unsigned int> dof_indices(this->n_functions);
  double                    eval_on_quad;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell->get_dof_indices(dof_indices);
      vol_handler.reinit(cell);

      for (unsigned int i = 0; i < this->n_functions; ++i)
        {
          for (unsigned int q = 0; q < n_quad_points; ++q)
            {
              eval_on_quad = 0;

              for (unsigned int j = 0; j < this->n_functions; ++j)
                {
                  eval_on_quad +=
                    fem_solution[dof_indices[j]] *
                    fe_dg.shape_value(j, vol_handler.quadrature_ref(q));
                }

              dub_solution[dof_indices[i]] +=
                eval_on_quad *
                this->shape_value(i, vol_handler.quadrature_ref(q)) *
                vol_handler.quadrature_weight(q);
            }
        }
    }

  return dub_solution;
}


template <unsigned int dim>
lifex::LinAlg::MPI::Vector
DUBFEMHandler<dim>::analytical_to_dubiner(
  lifex::LinAlg::MPI::Vector dub_solution,
  const std::shared_ptr<dealii::Function<lifex::dim>> &u_analytical)
{
  const dealii::FE_SimplexDGP<dim> fe_dg(this->poly_degree);
  DGVolumeHandler<dim>             vol_handler(this->poly_degree);

  std::vector<unsigned int> dof_indices(this->n_functions);
  double                    eval_on_quad;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell->get_dof_indices(dof_indices);
      vol_handler.reinit(cell);

      for (unsigned int i = 0; i < this->n_functions; ++i)
        {
          dub_solution[dof_indices[i]] = 0;
          for (unsigned int q = 0; q < n_quad_points; ++q)
            {
              eval_on_quad = u_analytical->value(vol_handler.quadrature_ref(q));

              dub_solution[dof_indices[i]] +=
                eval_on_quad *
                this->shape_value(i, vol_handler.quadrature_ref(q)) *
                vol_handler.quadrature_weight(q);
            }
        }
    }

  return dub_solution;
}


#endif /* DUBFEMHandler_HPP_*/
