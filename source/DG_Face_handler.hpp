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

#ifndef DGFaceHandler_HPP_
#define DGFaceHandler_HPP_

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_q.h>

#include <cmath>
#include <memory>
#include <utility>

#include "DG_Volume_handler.hpp"
#include "QGaussLegendreSimplex.hpp"

/**
 * @brief Class for the main operations on the faces of a discontinuous
 * Galerkin element.
 */
template <unsigned int dim>
class DGFaceHandler : public DGVolumeHandler<dim>
{
private:
  /// Index of the actual face.
  unsigned int edge;

  /// Tolerance.
  const double tol = 1e-10;

  /// Quadrature formula for face elements.
  const QGaussLegendreSimplex<dim - 1> QGLpoints_face;

  /// Object for FEM basis for face elements.
  std::unique_ptr<dealii::FEFaceValues<dim>> fe_face_values;

public:
  /// Constructor.
  DGFaceHandler<dim>(const unsigned int degree)
    : DGVolumeHandler<dim>(degree)
    , QGLpoints_face(this->n_quad_points_1D)
    , fe_face_values(std::make_unique<dealii::FEFaceValues<dim>>(
        *(this->mapping),
        *(this->fe_dg),
        QGLpoints_face,
        dealii::update_quadrature_points | dealii::update_normal_vectors |
          dealii::update_JxW_values | dealii::update_jacobians))
  {}

  /// Default copy constructor.
  DGFaceHandler<dim>(DGFaceHandler<dim> &DGFaceHandler) = default;

  /// Default const copy constructor.
  DGFaceHandler<dim>(const DGFaceHandler<dim> &DGFaceHandler) = default;

  /// Default move constructor.
  DGFaceHandler<dim>(DGFaceHandler<dim> &&DGFaceHandler) = default;

  /// Reinit objects on the current new_cell and new_edge.
  void
  reinit(const typename dealii::DoFHandler<dim>::active_cell_iterator &new_cell,
         const unsigned int new_edge);

  /// Spatial quadrature point position (actual element).
  dealii::Point<dim>
  quadrature_real(const unsigned int q) const override;

  /// Spatial quadrature point position (reference element).
  dealii::Point<dim>
  quadrature_ref(const unsigned int q) const override;

  /// Quadrature weight associated to a quadrature node.
  double
  quadrature_weight(const unsigned int quadrature_point_no) const override;

  /// If needed, to manually obtain the associated quadrature point in the
  /// neighbor element on the shared face.
  int
  corresponding_neigh_index(
    const unsigned int        q,
    const DGFaceHandler<dim> &DGFaceHandler_neigh) const;

  /// Outward normal vector on the current element and face.
  dealii::Tensor<1, dim>
  get_normal() const;

  /// Measure of face.
  double
  get_measure() const;

  /// Destructor.
  virtual ~DGFaceHandler() = default;
};

template <unsigned int dim>
void
DGFaceHandler<dim>::reinit(
  const typename dealii::DoFHandler<dim>::active_cell_iterator &new_cell,
  const unsigned int                                            new_edge)
{
  this->cell = new_cell;
  edge       = new_edge;
  fe_face_values->reinit(new_cell, new_edge);
  this->fe_values->reinit(new_cell);
  this->initialized = true;
}

template <unsigned int dim>
dealii::Point<dim>
DGFaceHandler<dim>::quadrature_real(const unsigned int q) const
{
  AssertThrow(this->initialized,
              dealii::StandardExceptions::ExcNotInitialized());

  AssertThrow(q < pow(this->n_quad_points_1D, dim - 1),
              dealii::StandardExceptions::ExcMessage(
                "Index of quadrature point outside the limit."));

  return fe_face_values->quadrature_point(q);
}

template <unsigned int dim>
dealii::Point<dim>
DGFaceHandler<dim>::quadrature_ref(const unsigned int q) const
{
  AssertThrow(this->initialized,
              dealii::StandardExceptions::ExcNotInitialized());

  AssertThrow(q < pow(this->n_quad_points_1D, dim - 1),
              dealii::StandardExceptions::ExcMessage(
                "Index of quadrature point outside the limit."));

  return this->mapping->transform_real_to_unit_cell(this->cell,
                                                    quadrature_real(q));
}

template <unsigned int dim>
double
DGFaceHandler<dim>::quadrature_weight(
  const unsigned int quadrature_point_no) const
{
  AssertThrow(this->initialized,
              dealii::StandardExceptions::ExcNotInitialized());

  AssertThrow(quadrature_point_no < pow(this->n_quad_points_1D, dim - 1),
              dealii::StandardExceptions::ExcMessage(
                "Index of quadrature point outside the limit."));

  return QGLpoints_face.weight(quadrature_point_no);
}

template <unsigned int dim>
int
DGFaceHandler<dim>::corresponding_neigh_index(
  const unsigned int        q,
  const DGFaceHandler<dim> &DGFaceHandler_neigh) const
{
  AssertThrow(this->initialized,
              dealii::StandardExceptions::ExcNotInitialized());

  AssertThrow(q < pow(this->n_quad_points_1D, dim - 1),
              dealii::StandardExceptions::ExcMessage(
                "Index of quadrature point outside the limit."));

  const unsigned int n_quad_points_face =
    static_cast<int>(std::pow(this->n_quad_points_1D, dim - 1));
  const dealii::Point<dim> P_q = quadrature_real(q);

  int quad = -1;
  for (unsigned int nq = 0; nq < n_quad_points_face; ++nq)
    {
      const dealii::Point<dim> P_nq = DGFaceHandler_neigh.quadrature_real(nq);

      if ((P_nq - P_q).norm() < tol)
        {
          quad = nq;
        }
    }

  return quad;
}

template <unsigned int dim>
dealii::Tensor<1, dim>
DGFaceHandler<dim>::get_normal() const
{
  AssertThrow(this->initialized,
              dealii::StandardExceptions::ExcNotInitialized());

  return fe_face_values->normal_vector(0);
}

/// Specialization to measure the face in two dimensions (i.e., length of a
/// segment).
template <>
double
DGFaceHandler<2>::get_measure() const
{
  AssertThrow(this->initialized,
              dealii::StandardExceptions::ExcNotInitialized());

  return this->cell->face(edge)->measure();
}

/// Specialization to measure the area of the face for a three dimensional
/// tethraedron (i.e., area of a triangle). The method exploits the Erone's
/// formula. Note that dealII does not have so far a version for triangles.
template <>
double
DGFaceHandler<3>::get_measure() const
{
  AssertThrow(this->initialized,
              dealii::StandardExceptions::ExcNotInitialized());

  const auto   face     = this->cell->face(edge);
  const double semi_per = (face->line(0)->measure() + face->line(1)->measure() +
                           face->line(2)->measure()) /
                          2;

  return std::sqrt(semi_per * (semi_per - face->line(0)->measure()) *
                   (semi_per - face->line(1)->measure()) *
                   (semi_per - face->line(2)->measure())); // Erone's formula
}

#endif /* DGFaceHandler_HPP_*/
