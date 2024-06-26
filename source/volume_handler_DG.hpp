/********************************************************************************
  Copyright (C) 2024 by the DUBeat authors.

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

#ifndef VOLUME_HANDLER_DG_HPP_
#define VOLUME_HANDLER_DG_HPP_

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <cmath>
#include <memory>
#include <utility>

#include "QGaussLegendreSimplex.hpp"

/**
 * @brief Class for the main operations on a discontinuous Galerkin volume element.
 */
template <unsigned int dim>
class VolumeHandlerDG
{
protected:
  /// Number of quadrature points in one dimensional elements.
  /// Default is polynomial degree + 2.
  const unsigned int n_quad_points_1D;

  /// Actual DG cell.
  typename dealii::DoFHandler<dim>::active_cell_iterator cell;

  /// Internal Lagrangian basis class. This internal member permits to exploit
  /// useful already implemented operations. Polynomial order is always set to 1
  /// because the class executes only geometric operations not related to the
  /// degrees of freedom.
  const std::unique_ptr<dealii::FE_SimplexDGP<dim>> fe_dg;

  /// Mapping of the discretized space, needed for geometrical
  /// reference-to-actual operations.
  const std::unique_ptr<dealii::MappingFE<dim>> mapping;

  /// Quadrature formula for volume elements.
  const QGaussLegendreSimplex<dim> QGLpoints;

  /// Internal FEM basis class. This internal member permits to exploit useful
  /// already implemented operations. As for fe_dg, the polynomial order is
  /// always 1 because it is only needed for geometric operations not related to
  /// the degrees of freedom.
  std::unique_ptr<dealii::FEValues<dim>> fe_values;

  /// A condition to inform if the class is initialized on an element or not.
  bool initialized = false;

public:
  /// Constructor.
  VolumeHandlerDG<dim>(const unsigned int degree)
    : n_quad_points_1D(degree + 2)
    , fe_dg(std::make_unique<dealii::FE_SimplexDGP<dim>>(1))
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
  VolumeHandlerDG<dim>(VolumeHandlerDG<dim> &VolumeHandlerDG) = default;

  /// Default const copy constructor.
  VolumeHandlerDG<dim>(const VolumeHandlerDG<dim> &VolumeHandlerDG) = default;

  /// Default move constructor.
  VolumeHandlerDG<dim>(VolumeHandlerDG<dim> &&VolumeHandlerDG) = default;

  /// Reinit objects on the current new_cell.
  void
  reinit(
    const typename dealii::DoFHandler<dim>::active_cell_iterator &new_cell);

  /// Return the @f$q@f$-th spatial quadrature point position on the actual
  /// element.
  virtual dealii::Point<dim>
  quadrature_real(const unsigned int q) const;

  /// Return the @f$q@f$-th spatial quadrature point position on the reference
  /// element.
  virtual dealii::Point<dim>
  quadrature_ref(const unsigned int q) const;

  /// Return the quadrature weight associated to the @f$q@f$-th quadrature
  /// point.
  virtual double
  quadrature_weight(const unsigned int q) const;

  /// Inverse of the Jacobian of the reference-to-actual transformation.
  dealii::Tensor<2, dim>
  get_jacobian_inverse() const;

  /// Get the number of quadrature points on the current element.
  unsigned int
  get_n_quad_points() const;

  /// Destructor.
  virtual ~VolumeHandlerDG() = default;
};

template <unsigned int dim>
void
VolumeHandlerDG<dim>::reinit(
  const typename dealii::DoFHandler<dim>::active_cell_iterator &new_cell)
{
  cell = new_cell;
  fe_values->reinit(new_cell);
  initialized = true;
}

template <unsigned int dim>
dealii::Point<dim>
VolumeHandlerDG<dim>::quadrature_real(const unsigned int q) const
{
  AssertThrow(initialized, dealii::StandardExceptions::ExcNotInitialized());

  AssertThrow(q < pow(n_quad_points_1D, dim),
              dealii::StandardExceptions::ExcMessage(
                "Index of quadrature point outside the limit."));

  return fe_values->quadrature_point(q);
}

template <unsigned int dim>
dealii::Point<dim>
VolumeHandlerDG<dim>::quadrature_ref(const unsigned int q) const
{
  AssertThrow(initialized, dealii::StandardExceptions::ExcNotInitialized());

  AssertThrow(q < pow(n_quad_points_1D, dim),
              dealii::StandardExceptions::ExcMessage(
                "Index of quadrature point outside the limit."));

  return mapping->transform_real_to_unit_cell(cell, quadrature_real(q));
}

template <unsigned int dim>
double
VolumeHandlerDG<dim>::quadrature_weight(const unsigned int q) const
{
  AssertThrow(initialized, dealii::StandardExceptions::ExcNotInitialized());

  AssertThrow(q < pow(n_quad_points_1D, dim),
              dealii::StandardExceptions::ExcMessage(
                "Index of quadrature point outside the limit."));

  return QGLpoints.weight(q);
}

template <unsigned int dim>
dealii::Tensor<2, dim>
VolumeHandlerDG<dim>::get_jacobian_inverse() const
{
  AssertThrow(initialized, dealii::StandardExceptions::ExcNotInitialized());

  dealii::Tensor<2, dim> BJinv;

  const dealii::DerivativeForm<1, dim, dim> jac_inv =
    fe_values->inverse_jacobian(0);

  // We now copy jac_inv in BJinv.
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
VolumeHandlerDG<dim>::get_n_quad_points() const
{
  AssertThrow(initialized, dealii::StandardExceptions::ExcNotInitialized());

  return fe_values.get_n_quad_points.size();
}

#endif /* VOLUME_HANDLER_DG_HPP_*/
