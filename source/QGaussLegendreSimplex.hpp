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

#ifndef LIFEX_QGAUSSLEGENDRESIMPLEX_HPP_
#define LIFEX_QGAUSSLEGENDRESIMPLEX_HPP_

#include <deal.II/base/config.h>

#include <deal.II/base/quadrature.h>

#include <utility>
#include <vector>

/// This routine computes the n Gauss-Ledendre nodes and weights on a given
/// interval (a,b)
extern std::pair<std::vector<dealii::Point<1>>, std::vector<double>>
gauleg(const double a, const double b, const unsigned int n)
{
  AssertThrow(a < b,
              dealii::StandardExceptions::ExcMessage(
                "(a,b) is an interval, hence a must be lower than b."));

  const double m  = (n + 1.) / 2.;
  const double xm = 0.5 * (b + a);
  const double xl = 0.5 * (b - a);

  std::vector<dealii::Point<1>> x(n);
  std::vector<double>           w(n);

  double z, z1, p1, p2, p3, pp = 0.0;

  for (unsigned int i = 1; i <= m; ++i)
    {
      z  = std::cos(M_PI * (i - 0.25) / (n + 0.5));
      z1 = z + 1.;

      while (!(std::abs(z - z1) < 1e-10))
        {
          p1 = 1.0;
          p2 = 0.0;

          for (unsigned int j = 1; j <= n; ++j)
            {
              p3 = p2;
              p2 = p1;
              p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j;
            }

          pp = n * (z * p1 - p2) / (pow(z, 2) - 1);
          z1 = z;
          z  = z1 - p1 / pp;
        }

      const dealii::Point<1> P1(xm - xl * z);
      const dealii::Point<1> P2(xm + xl * z);
      x[i - 1] = P1;
      x[n - i] = P2;
      w[i - 1] = 2.0 * xl / ((1.0 - z * z) * pp * pp);
      w[n - i] = w[i - 1];
    }

  return {x, w};
}


/**
 * @brief Class representing the Gauss Legendre quadrature formula on simplex elements.
 */
template <unsigned int dim>
class QGaussLegendreSimplex : public dealii::Quadrature<dim>
{
public:
  /// Constructor.
  QGaussLegendreSimplex<dim>(const unsigned int n);
};


/// To create @f$n@f$ quadrature points and weigths in the segment @f$(0,1)@f$.
template <>
QGaussLegendreSimplex<1>::QGaussLegendreSimplex(const unsigned int n)
  : dealii::Quadrature<1>(n)
{
  std::vector<dealii::Point<1>> x_1D(n);
  std::vector<double>           w_1D(n);
  std::tie(x_1D, w_1D) = gauleg(0, 1, n);

  this->quadrature_points = x_1D;
  this->weights           = w_1D;
}


/// To create @f$n^2@f$ quadrature points and weights in the triangle @f$(0,0) ,
/// (0.1) , (1,0)@f$.
template <>
QGaussLegendreSimplex<2>::QGaussLegendreSimplex(const unsigned int n)
  : dealii::Quadrature<2>(n)
{
  std::vector<dealii::Point<1>> x_1D(n);
  std::vector<double>           w_1D(n);
  std::tie(x_1D, w_1D) = gauleg(0, 1, n);

  std::vector<dealii::Point<1>> x(n);
  std::vector<double>           w(n);
  std::tie(x, w) = gauleg(-1, 1, n);

  std::vector<dealii::Point<2>> x_2D;
  std::vector<double>           w_2D;

  for (unsigned int i = 0; i < n; ++i)
    {
      for (unsigned int j = 0; j < n; ++j)
        {
          const dealii::Point<2> P((1 + x[i][0]) / 2,
                                   (1 - x[i][0]) * (1 + x[j][0]) / 4);

          x_2D.push_back(P);
          w_2D.push_back((1 - x[i][0]) * w[i] * w[j] / 8);
        }
    }

  this->quadrature_points = x_2D;
  this->weights           = w_2D;
}


/// To create @f$n^3@f$ quadrature points and weights in the tetrahedron
/// @f$(0,0,0) , (1,0,0) , (0,1,0) , (0,0,1)@f$.
template <>
QGaussLegendreSimplex<3>::QGaussLegendreSimplex(const unsigned int n)
  : dealii::Quadrature<3>(n)
{
  std::vector<dealii::Point<1>> x_1D(n);
  std::vector<double>           w_1D(n);
  std::tie(x_1D, w_1D) = gauleg(0, 1, n);

  std::vector<dealii::Point<1>> x(n);
  std::vector<double>           w(n);
  std::tie(x, w) = gauleg(-1, 1, n);

  std::vector<dealii::Point<3>> x_3D;
  std::vector<double>           w_3D;

  for (unsigned int i = 0; i < n; ++i)
    {
      for (unsigned int j = 0; j < n; ++j)
        {
          for (unsigned int k = 0; k < n; ++k)
            {
              const dealii::Point<3> P((x[i][0] + 1) * (x[j][0] - 1) *
                                         (x[k][0] - 1) / 8,
                                       (1 - x[k][0]) * (1 + x[j][0]) / 4,
                                       (x[k][0] + 1) / 2);

              x_3D.push_back(P);
              w_3D.push_back((1 - x[k][0]) * (1 - x[k][0]) * (1 - x[j][0]) *
                             w[i] * w[j] * w[k] / 64);
            }
        }
    }

  this->quadrature_points = x_3D;
  this->weights           = w_3D;
}


#endif /* LIFEX_QGAUSSLEGENDRESIMPLEX_HPP_*/
