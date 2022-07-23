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

#ifndef DGVolumeHandler_HPP_
#define DGVolumeHandler_HPP_

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_q.h>

#include <cmath>
#include <memory>
#include <utility>

#include "QGaussLegendreSimplex.hpp"

/**
 * @brief Class for the main operations on a discontinuous Galerkin element.
 */
template <unsigned int dim>
class DGVolumeHandler
{
protected:
  /// Number of quadrature points on one dimension.
  /// Default is polynomial degree + 2
  const unsigned int n_quad_points_1D;

  /// Actual DG cell.
  typename dealii::DoFHandler<dim>::active_cell_iterator cell;

  /// Internal DGFEM basis class. This internal member permits to exploit
  /// useful already implemented operations.
  const std::unique_ptr<dealii::FE_SimplexDGP<dim>> fe_dg;

  /// Mapping of the discretized space.
  const std::unique_ptr<dealii::MappingFE<dim>> mapping;

  /// Quadrature formula for volume elements.
  const QGaussLegendreSimplex<dim> QGLpoints;

  /// Internal FEM basis class. This internal member permits to exploit useful
  /// already implemented operations.
  std::unique_ptr<dealii::FEValues<dim>> fe_values;

  /// A bool to inform if the class is initialized or not.
  bool initialized = false;

public:
  /// Constructor.
  DGVolumeHandler<dim>(const unsigned int degree)
    : n_quad_points_1D(degree + 2)
    , fe_dg(std::make_unique<dealii::FE_SimplexDGP<dim>>(degree))
    , mapping(std::make_unique<dealii::MappingFE<dim>>(*fe_dg))
    , QGLpoints(n_quad_points_1D)
    , fe_values(std::make_unique<dealii::FEValues<dim>>(
        *mapping,
        *fe_dg,
        QGLpoints,
        dealii::update_inverse_jacobians | dealii::update_quadrature_points |
          dealii::update_values))
  {}

  /// Default copy constructor.
  DGVolumeHandler<dim>(DGVolumeHandler<dim> &DGVolumeHandler) = default;

  /// Default const copy constructor.
  DGVolumeHandler<dim>(const DGVolumeHandler<dim> &DGVolumeHandler) = default;

  /// Default move constructor.
  DGVolumeHandler<dim>(DGVolumeHandler<dim> &&DGVolumeHandler) = default;

  /// Reinit objects on the current new_cell.
  void
  reinit(
    const typename dealii::DoFHandler<dim>::active_cell_iterator &new_cell);

  /// Spatial quadrature point position (actual element).
  virtual dealii::Point<dim>
  quadrature_real(const unsigned int q) const;

  /// Spatial quadrature point position (reference element).
  virtual dealii::Point<dim>
  quadrature_ref(const unsigned int q) const;

  /// Quadrature_weight associated to quadrature node.
  virtual double
  quadrature_weight(const unsigned int quadrature_point_no) const;

  /// Inverse of the Jacobian of the reference to actual transformation.
  dealii::Tensor<2, dim>
  get_jacobian_inverse() const;

  /// Get the number of quadrature points on the current element.
  unsigned int
  get_n_quad_points() const;

  /// Destructor.
  virtual ~DGVolumeHandler() = default;
};

template <unsigned int dim>
void
DGVolumeHandler<dim>::reinit(
  const typename dealii::DoFHandler<dim>::active_cell_iterator &new_cell)
{
  cell = new_cell;
  fe_values->reinit(new_cell);

  initialized = true;
}

template <unsigned int dim>
dealii::Point<dim>
DGVolumeHandler<dim>::quadrature_real(const unsigned int q) const
{
  AssertThrow(initialized, dealii::StandardExceptions::ExcNotInitialized());

  AssertThrow(q < pow(n_quad_points_1D, dim),
              dealii::StandardExceptions::ExcMessage(
                "Index of quadrature point outside the limit."));

  return fe_values->quadrature_point(q);
}

template <unsigned int dim>
dealii::Point<dim>
DGVolumeHandler<dim>::quadrature_ref(const unsigned int q) const
{
  AssertThrow(initialized, dealii::StandardExceptions::ExcNotInitialized());

  AssertThrow(q < pow(n_quad_points_1D, dim),
              dealii::StandardExceptions::ExcMessage(
                "Index of quadrature point outside the limit."));

  return mapping->transform_real_to_unit_cell(cell, quadrature_real(q));
}

template <unsigned int dim>
double
DGVolumeHandler<dim>::quadrature_weight(
  const unsigned int quadrature_point_no) const
{
  AssertThrow(initialized, dealii::StandardExceptions::ExcNotInitialized());

  AssertThrow(quadrature_point_no < pow(n_quad_points_1D, dim),
              dealii::StandardExceptions::ExcMessage(
                "Index of quadrature point outside the limit."));

  return QGLpoints.weight(quadrature_point_no);
}

template <unsigned int dim>
dealii::Tensor<2, dim>
DGVolumeHandler<dim>::get_jacobian_inverse() const
{
  AssertThrow(initialized, dealii::StandardExceptions::ExcNotInitialized());

  dealii::Tensor<2, dim> BJinv;

  const dealii::DerivativeForm<1, dim, dim> jac_inv =
    fe_values->inverse_jacobian(0);

  for (unsigned int i = 0; i < dim; ++i)
    {
      for (unsigned int j = 0; j < dim; ++j)
        {
          BJinv[i][j] = jac_inv[i][j];
        }
    }

  return BJinv;
}

template <unsigned int dim>
unsigned int
DGVolumeHandler<dim>::get_n_quad_points() const
{
  AssertThrow(initialized, dealii::StandardExceptions::ExcNotInitialized());

  return fe_values.get_n_quad_points.size();
}

#endif /* DGVolumeHandler_HPP_*/
