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

#ifndef QGAUSSLEGENDRESIMPLEX_HPP_
#define QGAUSSLEGENDRESIMPLEX_HPP_

#include <deal.II/base/config.h>

#include <deal.II/base/quadrature.h>

#include <utility>
#include <vector>

/// This routine computes the n Gauss-Legendre nodes and weights on a given
/// interval (a,b)
extern std::pair<std::vector<dealii::Point<1>>, std::vector<double>>
gauleg(const double       left_position,
       const double       right_position,
       const unsigned int n_quadrature_points)
{
  AssertThrow(left_position < right_position,
              dealii::StandardExceptions::ExcMessage(
                "(a,b) is an interval, hence a must be lower than b."));

  // The following lines follow the definition of Gauss-Legendre nodes on an
  // interval.

  const double middle_point = 0.5 * (right_position + left_position);
  const double half_length  = 0.5 * (right_position - left_position);

  std::vector<dealii::Point<1>> coords(n_quadrature_points);
  std::vector<double>           weights(n_quadrature_points);

  double z, z1, p1, p2, p3, pp = 0.0;

  for (unsigned int i = 1; i <= (n_quadrature_points + 1) / 2.; ++i)
    {
      z  = std::cos(M_PI * (i - 0.25) / (n_quadrature_points + 0.5));
      z1 = z + 1.;

      while (!(std::abs(z - z1) < 1e-10))
        {
          p1 = 1.0;
          p2 = 0.0;

          for (unsigned int j = 1; j <= n_quadrature_points; ++j)
            {
              p3 = p2;
              p2 = p1;
              p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j;
            }

          pp = n_quadrature_points * (z * p1 - p2) / (pow(z, 2) - 1);
          z1 = z;
          z  = z1 - p1 / pp;
        }

      const dealii::Point<1> P1(middle_point - half_length * z);
      const dealii::Point<1> P2(middle_point + half_length * z);
      coords[i - 1]                   = P1;
      coords[n_quadrature_points - i] = P2;
      weights[i - 1] = 2.0 * half_length / ((1.0 - z * z) * pp * pp);
      weights[n_quadrature_points - i] = weights[i - 1];
    }

  return {coords, weights};
}

/**
 * @brief Class representing the Gauss Legendre quadrature formula on simplex
 * elements.
 */
template <unsigned int dim>
class QGaussLegendreSimplex : public dealii::Quadrature<dim>
{
public:
  /// Constructor.
  QGaussLegendreSimplex<dim>(const unsigned int n_quadrature_points);
};

/// To create @f$n@f$ quadrature points and weights in the segment @f$(0,1)@f$.
template <>
QGaussLegendreSimplex<1>::QGaussLegendreSimplex(
  const unsigned int n_quadrature_points)
  : dealii::Quadrature<1>(n_quadrature_points)
{
  std::vector<dealii::Point<1>> coords_1D(n_quadrature_points);
  std::vector<double>           weights_1D(n_quadrature_points);
  std::tie(coords_1D, weights_1D) = gauleg(0, 1, n_quadrature_points);

  this->quadrature_points = coords_1D;
  this->weights           = weights_1D;
}

/// To create @f$n^2@f$ quadrature points and weights on the triangle @f$(0,0)
/// , (0.1) , (1,0)@f$.
template <>
QGaussLegendreSimplex<2>::QGaussLegendreSimplex(
  const unsigned int n_quadrature_points)
  : dealii::Quadrature<2>(n_quadrature_points)
{
  std::vector<dealii::Point<1>> coords_1D(n_quadrature_points);
  std::vector<double>           weights_1D(n_quadrature_points);
  std::tie(coords_1D, weights_1D) = gauleg(-1, 1, n_quadrature_points);

  std::vector<dealii::Point<2>> coords_2D;
  std::vector<double>           weights_2D;

  for (unsigned int i = 0; i < n_quadrature_points; ++i)
    {
      for (unsigned int j = 0; j < n_quadrature_points; ++j)
        {
          const dealii::Point<2> P((1 + coords_1D[i][0]) / 2,
                                   (1 - coords_1D[i][0]) *
                                     (1 + coords_1D[j][0]) / 4);

          coords_2D.push_back(P);
          weights_2D.push_back((1 - coords_1D[i][0]) * weights_1D[i] *
                               weights_1D[j] / 8);
        }
    }

  this->quadrature_points = coords_2D;
  this->weights           = weights_2D;
}

/// To create @f$n^3@f$ quadrature points and weights on the tetrahedron
/// @f$(0,0,0) , (1,0,0) , (0,1,0) , (0,0,1)@f$.
template <>
QGaussLegendreSimplex<3>::QGaussLegendreSimplex(
  const unsigned int n_quadrature_points)
  : dealii::Quadrature<3>(n_quadrature_points)
{
  std::vector<dealii::Point<1>> coords_1D(n_quadrature_points);
  std::vector<double>           weights_1D(n_quadrature_points);
  std::tie(coords_1D, weights_1D) = gauleg(-1, 1, n_quadrature_points);

  std::vector<dealii::Point<3>> coords_3D;
  std::vector<double>           weights_3D;

  for (unsigned int i = 0; i < n_quadrature_points; ++i)
    {
      for (unsigned int j = 0; j < n_quadrature_points; ++j)
        {
          for (unsigned int k = 0; k < n_quadrature_points; ++k)
            {
              const dealii::Point<3> new_coords((coords_1D[i][0] + 1) *
                                                  (coords_1D[j][0] - 1) *
                                                  (coords_1D[k][0] - 1) / 8,
                                                (1 - coords_1D[k][0]) *
                                                  (1 + coords_1D[j][0]) / 4,
                                                (coords_1D[k][0] + 1) / 2);

              coords_3D.push_back(new_coords);
              weights_3D.push_back((1 - coords_1D[k][0]) *
                                   (1 - coords_1D[k][0]) *
                                   (1 - coords_1D[j][0]) * weights_1D[i] *
                                   weights_1D[j] * weights_1D[k] / 64);
            }
        }
    }

  this->quadrature_points = coords_3D;
  this->weights           = weights_3D;
}

#endif /* QGAUSSLEGENDRESIMPLEX_HPP_*/
