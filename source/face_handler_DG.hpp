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

#ifndef FACE_HANDLER_DG_HPP_
#define FACE_HANDLER_DG_HPP_

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_q.h>

#include <cmath>
#include <memory>
#include <utility>

#include "QGaussLegendreSimplex.hpp"
#include "volume_handler_DG.hpp"

/**
 * @brief Class for the main operations on the faces of a discontinuous
 * Galerkin element.
 */
template <unsigned int dim>
class FaceHandlerDG : public VolumeHandlerDG<dim>
{
private:
  /// Index of the actual face.
  unsigned int edge;

  /// Default tolerance.
  const double tol = 1e-10;

  /// Quadrature formula for face elements.
  const QGaussLegendreSimplex<dim - 1> QGLpoints_face;

  /// Internal FEM basis class for face elements. This internal member permits
  /// to exploit useful already implemented operations. The polynomial order is
  /// always 1 because it is only needed for geometric operations not related to
  /// the degrees of freedom.
  std::unique_ptr<dealii::FEFaceValues<dim>> fe_face_values;

public:
  /// Constructor.
  FaceHandlerDG<dim>(const unsigned int degree)
    : VolumeHandlerDG<dim>(degree)
    , QGLpoints_face(this->n_quad_points_1D)
    , fe_face_values(std::make_unique<dealii::FEFaceValues<dim>>(
        *(this->mapping),
        *(this->fe_dg),
        QGLpoints_face,
        dealii::update_quadrature_points | dealii::update_normal_vectors |
          dealii::update_JxW_values | dealii::update_jacobians))
  {}

  /// Default copy constructor.
  FaceHandlerDG<dim>(FaceHandlerDG<dim> &FaceHandlerDG) = default;

  /// Default const copy constructor.
  FaceHandlerDG<dim>(const FaceHandlerDG<dim> &FaceHandlerDG) = default;

  /// Default move constructor.
  FaceHandlerDG<dim>(FaceHandlerDG<dim> &&FaceHandlerDG) = default;

  /// Reinit objects on the current new_cell and new_edge.
  void
  reinit(const typename dealii::DoFHandler<dim>::active_cell_iterator &new_cell,
         const unsigned int new_edge);

  /// Return the @f$q@f$-th spatial quadrature point position on the actual
  /// element.
  dealii::Point<dim>
  quadrature_real(const unsigned int q) const override;

  /// Return the @f$q@f$-th spatial quadrature point position on the reference
  /// element.
  dealii::Point<dim>
  quadrature_ref(const unsigned int q) const override;

  /// Return the quadrature weight associated to the @f$q@f$-th quadrature
  /// point.
  double
  quadrature_weight(const unsigned int q) const override;

  /// To manually obtain the associated quadrature point index in the
  /// neighbor element on the shared face.
  int
  corresponding_neigh_index(
    const unsigned int        q,
    const FaceHandlerDG<dim> &FaceHandlerDG_neigh) const;

  /// Outward normal vector on the current element and face.
  dealii::Tensor<1, dim>
  get_normal() const;

  /// Measure of face.
  double
  get_measure() const;

  /// Destructor.
  virtual ~FaceHandlerDG() = default;
};

template <unsigned int dim>
void
FaceHandlerDG<dim>::reinit(
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
FaceHandlerDG<dim>::quadrature_real(const unsigned int q) const
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
FaceHandlerDG<dim>::quadrature_ref(const unsigned int q) const
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
FaceHandlerDG<dim>::quadrature_weight(const unsigned int q) const
{
  AssertThrow(this->initialized,
              dealii::StandardExceptions::ExcNotInitialized());

  AssertThrow(q < pow(this->n_quad_points_1D, dim - 1),
              dealii::StandardExceptions::ExcMessage(
                "Index of quadrature point outside the limit."));

  return QGLpoints_face.weight(q);
}

template <unsigned int dim>
int
FaceHandlerDG<dim>::corresponding_neigh_index(
  const unsigned int        q,
  const FaceHandlerDG<dim> &FaceHandlerDG_neigh) const
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
      const dealii::Point<dim> P_nq = FaceHandlerDG_neigh.quadrature_real(nq);

      // If the distance between P_nq and P_q is negligible, then we have found
      // the corresponding quadrature point on the neighbor element.
      if ((P_nq - P_q).norm() < tol)
        {
          quad = nq;
        }
    }

  return quad;
}

template <unsigned int dim>
dealii::Tensor<1, dim>
FaceHandlerDG<dim>::get_normal() const
{
  AssertThrow(this->initialized,
              dealii::StandardExceptions::ExcNotInitialized());

  return fe_face_values->normal_vector(0);
}

/// Specialization to measure the face in two dimensions (i.e., length of a
/// segment).
template <>
double
FaceHandlerDG<2>::get_measure() const
{
  AssertThrow(this->initialized,
              dealii::StandardExceptions::ExcNotInitialized());

  return this->cell->face(edge)->measure();
}

/// Specialization to measure the area of the face for a three dimensional
/// tethraedron (i.e., area of a triangle). The method exploits the Erone's
/// formula since deal.II does not have so far a version for triangles.
template <>
double
FaceHandlerDG<3>::get_measure() const
{
  AssertThrow(this->initialized,
              dealii::StandardExceptions::ExcNotInitialized());

  const auto face = this->cell->face(edge);

  // Semi-perimeter of the triangle.
  const double semi_per = (face->line(0)->measure() + face->line(1)->measure() +
                           face->line(2)->measure()) /
                          2;

  // Erone's formula.
  return std::sqrt(semi_per * (semi_per - face->line(0)->measure()) *
                   (semi_per - face->line(1)->measure()) *
                   (semi_per - face->line(2)->measure()));
}

#endif /* FACE_HANDLER_DG_HPP_*/
