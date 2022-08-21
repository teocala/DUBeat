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

#ifndef DUBValues_HPP_
#define DUBValues_HPP_

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_q.h>

#include <math.h>

#include <cmath>
#include <utility>

/**
 * @brief
 * Class representing the Dubiner basis functions definitions. The main utility
 * is the evaluations of these functions and their gradients on the reference
 * cell. Instead, for their evaluations on the real cells see VolumeHandlerDG
 * and FaceHandlerDG.
 */
template <unsigned int dim>
class DUBValues
{
protected:
  /// Polynomial degree used.
  const unsigned int poly_degree;

  /// Default tolerance.
  const double tol = 1e-10;

  /// Evaluation of the @f$(n,\alpha,\beta)@f$-Jacobi polynomial (used to
  /// evaluate Dubiner basis).
  double
  eval_jacobi_polynomial(const unsigned int n,
                         const unsigned int alpha,
                         const unsigned int beta,
                         const double       eval_point) const;

  /// Conversion from the FEM basis taxonomy (@f$1^{st}@f$ basis function,
  /// @f$2^{nd}@f$ basis function@f$\ldots@f$) to the Dubiner basis taxonomy
  /// (where every basis function is indexed by a tuple of indexes @f$(i,j)@f$
  /// in 2D and three indexes @f$(i,j,k)@f$ in 3D). It is needed to order
  /// sequentially all the Dubiner basis functions of a certain element.
  std::array<unsigned int, dim>
  fun_coeff_conversion(const unsigned int n) const;

public:
  /// Constructor.
  DUBValues<dim>(const unsigned int degree)
    : poly_degree(degree)
  {
    dofs_per_cell = get_dofs_per_cell();
  }

  /// Number of degrees of freedom.
  unsigned int dofs_per_cell;

  /// Default copy constructor.
  DUBValues<dim>(DUBValues<dim> &DUBValues) = default;

  /// Default const copy constructor.
  DUBValues<dim>(const DUBValues<dim> &DUBValues) = default;

  /// Default move constructor.
  DUBValues<dim>(DUBValues<dim> &&DUBValues) = default;

  /// Return the number of degrees of freedom per element.
  unsigned int
  get_dofs_per_cell() const;

  /// Same as get_dofs_per_cell() but with variable space degree.
  unsigned int
  get_dofs_per_cell(const unsigned int degree) const;

  /// Evaluation of the Dubiner basis functions.
  double
  shape_value(const unsigned int        function_no,
              const dealii::Point<dim> &quadrature_point) const;

  /// Evaluation of the Dubiner basis functions gradients.
  dealii::Tensor<1, dim>
  shape_grad(const unsigned int        function_no,
             const dealii::Point<dim> &quadrature_point) const;

  /// Default destructor.
  virtual ~DUBValues() = default;
};

/// Specialization in two dimensions, i.e. from an integer @f$n@f$ returns a
/// tuple of indexes @f$(i,j)@f$ such that the @f$n@f$-th Dubiner function
/// corresponds to the function with indexes @f$(i,j)@f$.
template <>
std::array<unsigned int, 2>
DUBValues<2>::fun_coeff_conversion(const unsigned int n) const
{
  unsigned int i, j;
  unsigned int c     = poly_degree + 1;
  unsigned int c_seq = c;
  while (n > c_seq - 1)
    {
      c -= 1;
      c_seq += c;
    }

  i = n - (c_seq - c);
  j = poly_degree + 1 - c;

  return {{i, j}};
}

/// Specialization in three dimensions, i.e. from an integer @f$n@f$ returns
/// three indexes @f$(i,j,k)@f$ such that the @f$n@f$-th Dubiner function
/// corresponds to the function with indexes @f$(i,j,k)@f$.
template <>
std::array<unsigned int, 3>
DUBValues<3>::fun_coeff_conversion(const unsigned int n) const
{
  unsigned int i, j, k;
  unsigned int p      = poly_degree;
  unsigned int c1     = (p + 1) * (p + 2) / 2;
  unsigned int c1_seq = c1;

  while (n > c1_seq - 1)
    {
      p -= 1;
      c1 = (p + 1) * (p + 2) / 2;
      c1_seq += c1;
    }

  i                   = poly_degree - p;
  unsigned int n2     = n - (c1_seq - c1) + 1;
  unsigned int c2     = p + 1;
  unsigned int c2_seq = c2;

  while (n2 > c2_seq)
    {
      c2 -= 1;
      c2_seq += c2;
    }

  j = poly_degree - i + 1 - c2;
  k = n2 - (c2_seq - c2) - 1;

  return {{i, j, k}};
}

template <unsigned int dim>
double
DUBValues<dim>::eval_jacobi_polynomial(const unsigned int n,
                                       const unsigned int alpha,
                                       const unsigned int beta,
                                       const double       eval_point) const
{
  if (n == 0)
    return 1.0;
  else if (n == 1)
    return static_cast<double>(
      (alpha - beta + (alpha + beta + 2.0) * eval_point) / 2.0);
  else
    {
      const double apb    = alpha + beta;
      double       poly   = 0.0;
      double       polyn2 = 1.0;
      double       polyn1 = static_cast<double>(
        (alpha - beta + (alpha + beta + 2.0) * eval_point) / 2.0);
      double a1, a2, a3, a4;

      for (unsigned int k = 2; k <= n; ++k)
        {
          a1 = 2.0 * static_cast<double>(k) * static_cast<double>(k + apb) *
               static_cast<double>(2.0 * k + apb - 2.0);
          a2 = static_cast<double>(2.0 * k + apb - 1.0) *
               static_cast<double>(alpha * alpha - beta * beta);
          a3 = static_cast<double>(2.0 * k + apb - 2.0) *
               static_cast<double>(2.0 * k + apb - 1.0) *
               static_cast<double>(2.0 * k + apb);
          a4 = 2.0 * static_cast<double>(k + alpha - 1.0) *
               static_cast<double>(k + beta - 1.0) *
               static_cast<double>(2.0 * k + apb);

          a2 = a2 / a1;
          a3 = a3 / a1;
          a4 = a4 / a1;

          poly   = (a2 + a3 * eval_point) * polyn1 - a4 * polyn2;
          polyn2 = polyn1;
          polyn1 = poly;
        }

      return poly;
    }
}

template <unsigned int dim>
unsigned int
DUBValues<dim>::get_dofs_per_cell() const
{
  // The analytical formula is:
  // n_dof_per_cell = (p+1)*(p+2)*...(p+d) / d!,
  // where p is the space order and d the space dimension..

  unsigned int denominator = 1;
  unsigned int nominator   = 1;

  for (unsigned int i = 1; i <= dim; i++)
    {
      denominator *= i;
      nominator *= poly_degree + i;
    }

  return (int)(nominator / denominator);
}

template <unsigned int dim>
unsigned int
DUBValues<dim>::get_dofs_per_cell(const unsigned int degree) const
{
  // The analytical formula is:
  // n_dof_per_cell = (p+1)*(p+2)*...(p+d) / d!,
  // where p is the space order and d the space dimension..

  unsigned int denominator = 1;
  unsigned int nominator   = 1;

  for (unsigned int i = 1; i <= dim; i++)
    {
      denominator *= i;
      nominator *= degree + i;
    }

  return (int)(nominator / denominator);
}

/// Evaluation of the function basis in two dimensions.
template <>
double
DUBValues<2>::shape_value(const unsigned int      function_no,
                          const dealii::Point<2> &quadrature_point) const
{
  AssertThrow(function_no < dofs_per_cell,
              dealii::StandardExceptions::ExcMessage(
                "function_no outside the limit."));

  const auto fun_coeff = fun_coeff_conversion(function_no);

  const int    i   = fun_coeff[0];
  const int    j   = fun_coeff[1];
  const double csi = quadrature_point[0];
  const double eta = quadrature_point[1];

  // If the point is outside the reference cell, the evaluation is zero.
  if (csi+eta>1.0+tol || csi < -tol || eta <-tol)
    return 0.0;

  double a, b;
  a = b = 0;

  if (std::abs(1 - eta) > this->tol)
    a = 2.0 * csi / (1 - eta) - 1.0;
  b = 2.0 * eta - 1.0;

  const double r  = sqrt((2 * i + 1) * 2 * (i + j + 1) / pow(4, i));
  const double pi = this->eval_jacobi_polynomial(i, 0, 0, a);
  const double pj = this->eval_jacobi_polynomial(j, 2 * i + 1, 0, b);

  return r * pow(2, i) * pow(1 - eta, i) * pi * pj;
}

/// Evaluation of the function basis in three dimensions.
template <>
double
DUBValues<3>::shape_value(const unsigned int      function_no,
                          const dealii::Point<3> &quadrature_point) const
{
  AssertThrow(function_no < dofs_per_cell,
              dealii::StandardExceptions::ExcMessage(
                "function_no outside the limit."));

  const auto fun_coeff = fun_coeff_conversion(function_no);

  const int    i   = fun_coeff[0];
  const int    j   = fun_coeff[1];
  const int    k   = fun_coeff[2];
  const double csi = quadrature_point[0];
  const double eta = quadrature_point[1];
  const double ni  = quadrature_point[2];

  // If the point is outside the reference cell, the evaluation is zero.
  if (csi+eta+ni>1.0+tol || csi < -tol || eta <-tol || ni<-tol)
    return 0.0;

  double a, b, c;
  a = b = c = 0;

  if (std::abs(-eta - ni + 1.) > this->tol)
    a = 2. * csi / (-eta - ni + 1.) - 1.;
  if (std::abs(1 - ni) > this->tol)
    b = 2. * eta / (1. - ni) - 1.;
  c = 2. * ni - 1.;

  const double r =
    sqrt(8 * pow(2, 4 * i + 2 * j + 3) * (2 * i + 2 * j + 2 * k + 3) *
         (2 * i + 2 * j + 2) * (2 * i + 1) / pow(2, 4 * i + 2 * j + 6));
  const double pi = this->eval_jacobi_polynomial(i, 0, 0, a);
  const double pj = this->eval_jacobi_polynomial(j, 2 * i + 1, 0, b);
  const double pk = this->eval_jacobi_polynomial(k, 2 * i + 2 * j + 2, 0, c);

  return r * pi * pow((1. - b) / 2., i) * pj * pow((1. - c) / 2., i + j) * pk;
}

/// Evaluation of the gradient of the function basis in two dimensions.
template <>
dealii::Tensor<1, 2>
DUBValues<2>::shape_grad(const unsigned int      function_no,
                         const dealii::Point<2> &quadrature_point) const
{
  AssertThrow(function_no < dofs_per_cell,
              dealii::StandardExceptions::ExcMessage(
                "function_no outside the limit."));

  const auto fun_coeff = fun_coeff_conversion(function_no);

  const int    i   = fun_coeff[0];
  const int    j   = fun_coeff[1];
  const double csi = quadrature_point[0];
  const double eta = quadrature_point[1];

  dealii::Tensor<1, 2> grad;

  // If the point is outside the reference cell, the evaluation is zero.
  if (csi+eta>1.0+tol || csi < -tol || eta <-tol)
    return grad;

  double a, b;
  a = b = 0;

  if (std::abs(1 - eta) > this->tol)
    a = 2.0 * csi / (1 - eta) - 1.0;
  b = 2.0 * eta - 1.0;

  const double r  = sqrt((2 * i + 1) * 2 * (i + j + 1) / pow(4, i));
  const double pi = this->eval_jacobi_polynomial(i, 0, 0, a);
  const double pj = this->eval_jacobi_polynomial(j, 2 * i + 1, 0, b);


  if (i == 0 && j == 0)
    {
      grad[0] = 0.0;
      grad[1] = 0.0;
    }
  else if (i == 0 && j != 0)
    {
      grad[0] = 0.0;
      grad[1] = r * (j + 2) * eval_jacobi_polynomial(j - 1, 2, 1, b);
    }
  else if (i != 0 && j == 0)
    {
      grad[0] = r * pow(2, i) * pow(1 - eta, i - 1) * (i + 1) *
                eval_jacobi_polynomial(i - 1, 1, 1, a);
      grad[1] = r * pow(2, i) *
                (-i * pow(1 - eta, i - 1) * pi +
                 csi * pow(1 - eta, i - 2) * (i + 1) *
                   eval_jacobi_polynomial(i - 1, 1, 1, a));
    }
  else
    {
      grad[0] = r * pow(2, i) * pow(1 - eta, i - 1) * (i + 1) *
                eval_jacobi_polynomial(i - 1, 1, 1, a) *
                eval_jacobi_polynomial(j, 2 * i + 1, 0, b);
      grad[1] = r * pow(2, i) *
                (-i * pow(1 - eta, i - 1) * pi * pj +
                 csi * pow(1 - eta, i - 2) * (i + 1) *
                   eval_jacobi_polynomial(i - 1, 1, 1, a) * pj +
                 pow(1 - eta, i) * (2 * i + j + 2) * pi *
                   eval_jacobi_polynomial(j - 1, 2 * i + 2, 1, b));
    }

  return grad;
}

/// Evaluation of the gradient of the function basis in three dimensions.
template <>
dealii::Tensor<1, 3>
DUBValues<3>::shape_grad(const unsigned int      function_no,
                         const dealii::Point<3> &quadrature_point) const
{
  AssertThrow(function_no < dofs_per_cell,
              dealii::StandardExceptions::ExcMessage(
                "function_no outside the limit."));

  const auto fun_coeff = fun_coeff_conversion(function_no);

  const int    i   = fun_coeff[0];
  const int    j   = fun_coeff[1];
  const int    k   = fun_coeff[2];
  const double csi = quadrature_point[0];
  const double eta = quadrature_point[1];
  const double ni  = quadrature_point[2];

   dealii::Tensor<1, 3> grad;

  // If the point is outside the reference cell, the evaluation is zero.
  if (csi+eta+ni>1.0+tol || csi < -tol || eta <-tol || ni<-tol)
    return grad;

  double a, b, c;
  a = b = c = 0.0;
  if (std::abs(-eta - ni + 1.) > this->tol)
    a = 2. * csi / (-eta - ni + 1.) - 1;
  if (std::abs(1. - ni) > this->tol)
    b = 2. * eta / (1. - ni) - 1.;
  c = 2. * ni - 1.;

  const double r =
    sqrt(8 * pow(2, 4 * i + 2 * j + 3) * (2 * i + 2 * j + 2 * k + 3) *
         (2 * i + 2 * j + 2) * (2 * i + 1) / pow(2, 4 * i + 2 * j + 6));
  const double pi = this->eval_jacobi_polynomial(i, 0, 0, a);
  const double pj = this->eval_jacobi_polynomial(j, 2 * i + 1, 0, b);
  const double pk = this->eval_jacobi_polynomial(k, 2 * i + 2 * j + 2, 0, c);

 

  double dPi_csi = 0.0, dPi_eta = 0.0, dPi_ni = 0.0;
  double db_csi = 0.0, db_eta = 0.0, db_ni = 0.0;
  double dPj_csi = 0.0, dPj_eta = 0.0, dPj_ni = 0.0;
  double dc_csi = 0.0, dc_eta = 0.0, dc_ni = 0.0;
  double dPk_csi = 0.0, dPk_eta = 0.0, dPk_ni = 0.0;

  if (i != 0)
    {
      dPi_csi = static_cast<double>(i + 1.) * 1. / (-eta - ni + 1.) *
                eval_jacobi_polynomial(i - 1, 1, 1, a);
      dPi_eta = static_cast<double>(i + 1.) * csi / pow(-eta - ni + 1., 2.) *
                eval_jacobi_polynomial(i - 1, 1, 1, a);
      dPi_ni = static_cast<double>(i + 1.) * csi / pow(-eta - ni + 1., 2.) *
               eval_jacobi_polynomial(i - 1, 1, 1, a);

      db_csi = 0.0;
      db_eta = -static_cast<double>(i) / (1. - ni) * pow((1. - b) / 2., i - 1.);
      db_ni  = -static_cast<double>(i) * eta / pow(1. - ni, 2) *
              pow((1. - b) / 2, i - 1.);
    }

  if (j != 0)
    {
      dPj_csi = 0.0;
      dPj_eta = static_cast<double>(2. * i + j + 2.) / (1. - ni) *
                eval_jacobi_polynomial(j - 1, 2 * i + 2, 1, b);
      dPj_ni = eta * static_cast<double>(2. * i + j + 2.) / pow(1. - ni, 2.) *
               eval_jacobi_polynomial(j - 1, 2 * i + 2, 1, b);
    }

  if ((i + j) != 0)
    {
      dc_csi = 0.0;
      dc_eta = 0.0;
      dc_ni  = -static_cast<double>(i + j) * pow((1. - c) / 2., i + j - 1.);
    }

  if (k != 0)
    {
      dPk_csi = 0.0;
      dPk_eta = 0.0;
      dPk_ni  = static_cast<double>(2. * i + 2. * j + k + 3.) *
               eval_jacobi_polynomial(k - 1, 2 * i + 2 * j + 3, 1, c);
    }

  grad[0] =
    r * (dPi_csi * pow((1. - b) / 2., i) * pj * pow((1. - c) / 2., i + j) * pk +
         pi * db_csi * pj * pow((1. - c) / 2., i + j) * pk +
         pi * pow((1. - b) / 2., i) * dPj_csi * pow((1. - c) / 2., i + j) * pk +
         pi * pow((1. - b) / 2., i) * pj * dc_csi * pk +
         pi * pow((1. - b) / 2., i) * pj * pow((1. - c) / 2., i + j) * dPk_csi);

  grad[1] =
    r * (dPi_eta * pow((1. - b) / 2., i) * pj * pow((1. - c) / 2., i + j) * pk +
         pi * db_eta * pj * pow((1. - c) / 2., i + j) * pk +
         pi * pow((1. - b) / 2., i) * dPj_eta * pow((1. - c) / 2., i + j) * pk +
         pi * pow((1. - b) / 2., i) * pj * dc_eta * pk +
         pi * pow((1. - b) / 2., i) * pj * pow((1. - c) / 2., i + j) * dPk_eta);

  grad[2] =
    r * (dPi_ni * pow((1. - b) / 2., i) * pj * pow((1. - c) / 2., i + j) * pk +
         pi * db_ni * pj * pow((1. - c) / 2., i + j) * pk +
         pi * pow((1. - b) / 2., i) * dPj_ni * pow((1. - c) / 2., i + j) * pk +
         pi * pow((1. - b) / 2., i) * pj * dc_ni * pk +
         pi * pow((1. - b) / 2., i) * pj * pow((1. - c) / 2., i + j) * dPk_ni);

  return grad;
}

#endif /* DUBValues_HPP_*/
