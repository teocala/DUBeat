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

#ifndef ComputeErrorsDG_HPP_
#define ComputeErrorsDG_HPP_

#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping_q1_eulerian.h>

#include <deal.II/lac/trilinos_vector.h>

#include <cmath>
#include <vector>

#include "DUBValues.hpp"


/**
 * @brief Class to compute the errors between the numerical solution (solution_owned) and the exact solution (solution_ex_owned).
 * Four definitions of error are implemented:
 *
 * @f[ \|u-u_h\|_{L^\infty(\Omega)}:=  \sup_{x\in \Omega} |u(x)-u_h(x)|  @f]
 * @f[ \|u-u_h\|_{L^2(\Omega)}^2 := \int_{\Omega} |u(x)-u_h(x)|^2 \, dx  @f]
 * @f[ \|u-u_h\|_{H^1(\Omega)}^2 := \|u-u_h\|_{L^2(\Omega)}^2 + \|\nabla
 * u-\nabla u_h\|_{L^2(\Omega)}^2  @f]
 * @f[ \|u-u_h\|_{DG(\Omega)}^2 := \|\nabla u-\nabla u_h\|_{L^2(\Omega)}^2
 * +\gamma \|[[ u-\nabla u_h]]\|_{L^2(\mathcal{F})}^2  @f] where @f$\gamma @f$
 * is the stability coefficient and the @f$L^2(\mathcal{F}) @f$ norm is computed
 * on the faces instead of the volume.
 */

template <class basis>
class Compute_Errors_DG
{
public:
  /// Constructor.
  Compute_Errors_DG<basis>(const unsigned int degree,
                           const double       stability_coeff,
                           const unsigned int local_dofs,
                           const dealii::DoFHandler<lifex::dim> &dof_hand)
    : dof_handler(dof_hand)
    , n_quad_points(static_cast<int>(std::pow(degree + 2, lifex::dim)))
    , n_quad_points_face(static_cast<int>(std::pow(degree + 2, lifex::dim - 1)))
    , dofs_per_cell(local_dofs)
    , fe_degree(degree)
    , stability_coefficient(stability_coeff)
    , basis_ptr(std::make_unique<basis>(degree))
    , solution_name((char *)"u")
    , errors({0, 0, 0, 0})
    , dub_fem_values(
        std::make_shared<DUBFEMHandler<lifex::dim>>(degree, dof_handler))
  {}

  /// Default copy constructor.
  Compute_Errors_DG<basis>(Compute_Errors_DG<basis> &ComputeErrorsDG) = default;

  /// Default const copy constructor.
  Compute_Errors_DG<basis>(const Compute_Errors_DG<basis> &ComputeErrorsDG) =
    default;

  /// Default move constructor.
  Compute_Errors_DG<basis>(Compute_Errors_DG<basis> &&ComputeErrorsDG) =
    default;

  /// Reinitialization with new computed and exact solutions.
  void
  reinit(const lifex::LinAlg::MPI::Vector &                   sol_owned,
         const lifex::LinAlg::MPI::Vector &                   sol_ex_owned,
         const std::shared_ptr<dealii::Function<lifex::dim>> &u_ex_input,
         const std::shared_ptr<dealii::Function<lifex::dim>> &grad_u_ex_input,
         const char *solution_name_input);

  /// Compute errors following the preferences in the list. E.g.,
  /// errors_defs={"L2"} will only compute the @f$L^2@f$ error.
  void
  compute_errors(
    std::list<const char *> errors_defs = {"inf", "L2", "H1", "DG"});

  /// Output of the errors following the preferences in the list. E.g.,
  /// errors_defs={"L2"} will only output the @f$L^2@f$ error. The output vector
  /// contains the errors in the order of the input list.
  std::vector<double>
  output_errors(
    std::list<const char *> errors_defs = {"inf", "L2", "H1", "DG"}) const;

private:
  /// Compute the @f$L^\infty@f$ error.
  void
  compute_error_inf();

  /// Compute the @f$L^2@f$ error.
  void
  compute_error_L2();

  /// Compute the @f$H^1@f$ error.
  void
  compute_error_H1();

  /// Compute the @f$DG@f$ error.
  void
  compute_error_DG();

  /// Dof handler object of the problem.
  const dealii::DoFHandler<lifex::dim> &dof_handler;

  /// Number of quadrature points in the volume element.
  /// By default: @f$(degree+2)^{dim}@f$.
  const unsigned int n_quad_points;

  /// Number of quadrature points on the face element.
  /// By default: @f$(degree+2)^{dim-1}@f$.
  const unsigned int n_quad_points_face;

  /// Number of degrees of freedom per cell.
  const unsigned int dofs_per_cell;

  /// Stability coefficient, needed for the computation of the @f$DG@f$ error.
  const double stability_coefficient;

  /// Polynomial degree.
  const unsigned int fe_degree;

  /// Pointer to the basis handler.
  const std::unique_ptr<basis> basis_ptr;

  /// Computed solution.
  lifex::LinAlg::MPI::Vector solution_owned;

  /// Exact solution to compare with the numerical one.
  lifex::LinAlg::MPI::Vector solution_ex_owned;

  /// Analytical exact solution.
  std::shared_ptr<dealii::Function<lifex::dim>> u_ex;

  /// Gradient of the analytical exact solution.
  std::shared_ptr<dealii::Function<lifex::dim>> grad_u_ex;

  /// String that contains the solution name (to write on output files, default
  /// "u").
  char *solution_name;

  /// Array that contains the current errors (in order, @f$L^\infty@f$,
  /// @f$L^2@f$, @f$H^1@f$, @f$DG@f$).
  std::array<double, 4> errors;

  /// Member to compute conversion to FEM basis, needed for L^inf error.
  std::shared_ptr<DUBFEMHandler<lifex::dim>> dub_fem_values;
};

template <class basis>
void
Compute_Errors_DG<basis>::reinit(
  const lifex::LinAlg::MPI::Vector &                   sol_owned,
  const lifex::LinAlg::MPI::Vector &                   sol_ex_owned,
  const std::shared_ptr<dealii::Function<lifex::dim>> &u_ex_input,
  const std::shared_ptr<dealii::Function<lifex::dim>> &grad_u_ex_input,
  const char *                                         solution_name_input)
{
  solution_owned    = sol_owned;
  solution_ex_owned = sol_ex_owned;
  u_ex              = u_ex_input;
  grad_u_ex         = grad_u_ex_input;
  solution_name     = (char *)solution_name_input;
}


template <class basis>
void
Compute_Errors_DG<basis>::compute_errors(std::list<const char *> errors_defs)
{
  AssertThrow(u_ex != nullptr,
              dealii::StandardExceptions::ExcMessage(
                "No valid pointer to the exact solution."));

  AssertThrow(grad_u_ex != nullptr,
              dealii::StandardExceptions::ExcMessage(
                "No valid pointer to the gradient of the exact solution."));

  AssertThrow(solution_owned.size() == solution_ex_owned.size(),
              dealii::StandardExceptions::ExcMessage(
                "The exact solution vector and the approximate solution vector "
                "must have the same length."));

  if (std::find(errors_defs.begin(), errors_defs.end(), "inf") !=
      errors_defs.end())
    {
      this->compute_error_inf();
    }

  // We need to respect the following order because the H1 semi error
  // contributes to the DG error and the L2 error contributes to the H1 error.
  if (std::find(errors_defs.begin(), errors_defs.end(), "DG") !=
      errors_defs.end())
    {
      this->compute_error_L2();
      this->compute_error_H1();
      this->compute_error_DG();
    }
  else if (std::find(errors_defs.begin(), errors_defs.end(), "H1") !=
           errors_defs.end())
    {
      this->compute_error_L2();
      this->compute_error_H1();
    }
  else if (std::find(errors_defs.begin(), errors_defs.end(), "L2") !=
           errors_defs.end())
    {
      this->compute_error_L2();
    }
}


template <class basis>
std::vector<double>
Compute_Errors_DG<basis>::output_errors(
  std::list<const char *> errors_defs) const
{
  std::vector<double> output_errors = {};

  for (auto error_def = errors_defs.begin(); error_def != errors_defs.end();
       error_def++)
    {
      AssertThrow(*error_def == "inf" || *error_def == "L2" ||
                    *error_def == "H1" || *error_def == "DG",
                  dealii::StandardExceptions::ExcMessage(
                    "Error definition must be inf, L2, H1 or DG."));

      if (*error_def == "inf")
        output_errors.push_back(errors[0]);
      else if (*error_def == "L2")
        output_errors.push_back(errors[1]);
      else if (*error_def == "H1")
        output_errors.push_back(errors[2]);
      else
        output_errors.push_back(errors[3]);
    }
  return output_errors;
}


template <class basis>
void
Compute_Errors_DG<basis>::compute_error_inf()
{
  lifex::LinAlg::MPI::Vector difference = solution_owned;
  difference -= solution_ex_owned;

  errors[0] = difference.linfty_norm();
}

/// Specialized version for Dubiner basis. Before computing the @f$L^\infty@f$
/// error, the vector solutions are transformed in terms of FEM coefficients.
template <>
void
Compute_Errors_DG<DUBValues<lifex::dim>>::compute_error_inf()
{
  lifex::LinAlg::MPI::Vector solution_fem =
    dub_fem_values->dubiner_to_fem(solution_owned);
  lifex::LinAlg::MPI::Vector solution_ex_fem =
    dub_fem_values->dubiner_to_fem(solution_ex_owned);

  lifex::LinAlg::MPI::Vector difference = solution_fem;
  difference -= solution_ex_fem;

  errors[0] = difference.linfty_norm();
}


template <class basis>
void
Compute_Errors_DG<basis>::compute_error_L2()
{
  double error_L2 = 0;

  DGVolumeHandler<lifex::dim>                 vol_handler(fe_degree);
  std::vector<lifex::types::global_dof_index> dof_indices(dofs_per_cell);

  double local_approx;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      vol_handler.reinit(cell);

      if (cell->is_locally_owned())
        {
          const dealii::Tensor<2, lifex::dim> BJinv =
            vol_handler.get_jacobian_inverse();
          const double det = 1 / determinant(BJinv);

          cell->get_dof_indices(dof_indices);

          for (unsigned int q = 0; q < n_quad_points; ++q)
            {
              local_approx = 0;

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  local_approx +=
                    basis_ptr->shape_value(i, vol_handler.quadrature_ref(q)) *
                    solution_owned[dof_indices[i]];
                }

              error_L2 +=
                pow(local_approx - u_ex->value(vol_handler.quadrature_real(q)),
                    2) *
                det * vol_handler.quadrature_weight(q);
            }
        }
    }

  errors[1] = sqrt(error_L2);
}

template <class basis>
void
Compute_Errors_DG<basis>::compute_error_H1()
{
  double                        error_semi_H1 = 0;
  dealii::Tensor<1, lifex::dim> local_approx_gradient;
  dealii::Tensor<1, lifex::dim> local_grad_exact;
  dealii::Tensor<1, lifex::dim> pointwise_diff;

  DGVolumeHandler<lifex::dim>                 vol_handler(fe_degree);
  std::vector<lifex::types::global_dof_index> dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      vol_handler.reinit(cell);

      if (cell->is_locally_owned())
        {
          const dealii::Tensor<2, lifex::dim> BJinv =
            vol_handler.get_jacobian_inverse();
          const double det = 1 / determinant(BJinv);

          cell->get_dof_indices(dof_indices);

          for (unsigned int q = 0; q < n_quad_points; ++q)
            {
              local_grad_exact      = 0;
              local_approx_gradient = 0;
              pointwise_diff        = 0;

              for (unsigned int j = 0; j < lifex::dim; ++j)
                {
                  local_grad_exact[j] =
                    grad_u_ex->value(vol_handler.quadrature_real(q), j);

                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      local_approx_gradient[j] +=
                        basis_ptr->shape_grad(
                          i, vol_handler.quadrature_ref(q))[j] *
                        solution_owned[dof_indices[i]];
                    }
                }

              pointwise_diff =
                local_grad_exact - (local_approx_gradient * BJinv);

              error_semi_H1 += pointwise_diff * pointwise_diff * det *
                               vol_handler.quadrature_weight(q);
            }
        }
    }

  errors[2] = sqrt(errors[1] * errors[1] + error_semi_H1);
}


template <class basis>
void
Compute_Errors_DG<basis>::compute_error_DG()
{
  double error_DG = 0;

  DGVolumeHandler<lifex::dim>                 vol_handler(fe_degree);
  std::vector<lifex::types::global_dof_index> dof_indices(dofs_per_cell);
  DGFaceHandler<lifex::dim>                   face_handler(fe_degree);
  DGFaceHandler<lifex::dim>                   face_handler_neigh(fe_degree);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      vol_handler.reinit(cell);

      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(dof_indices);

          for (const auto &edge : cell->face_indices())
            {
              face_handler.reinit(cell, edge);

              const double face_measure      = face_handler.get_measure();
              const double unit_face_measure = (4.0 - lifex::dim) / 2;
              const double measure_ratio     = face_measure / unit_face_measure;
              const double h_local           = (cell->measure()) / face_measure;

              lifex::LinAlg::MPI::Vector difference = solution_owned;
              difference -= solution_ex_owned;

              const double local_stability_coeff =
                (stability_coefficient * pow(fe_degree, 2)) / h_local;

              std::vector<lifex::types::global_dof_index> dof_indices_neigh(
                dofs_per_cell);

              for (unsigned int q = 0; q < n_quad_points_face; ++q)
                {
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                          error_DG +=
                            difference[dof_indices[i]] *
                            difference[dof_indices[j]] * local_stability_coeff *
                            basis_ptr->shape_value(
                              i, face_handler.quadrature_ref(q)) *
                            basis_ptr->shape_value(
                              j, face_handler.quadrature_ref(q)) *
                            face_handler.quadrature_weight(q) * measure_ratio;

                          if (!cell->at_boundary(edge))
                            {
                              const auto neighcell = cell->neighbor(edge);
                              const auto neighedge =
                                cell->neighbor_face_no(edge);
                              neighcell->get_dof_indices(dof_indices_neigh);
                              face_handler_neigh.reinit(neighcell, neighedge);

                              const unsigned int nq =
                                face_handler.corresponding_neigh_index(
                                  q, face_handler_neigh);

                              error_DG -=
                                difference[dof_indices[i]] *
                                difference[dof_indices_neigh[j]] *
                                local_stability_coeff *
                                basis_ptr->shape_value(
                                  i, face_handler.quadrature_ref(q)) *
                                basis_ptr->shape_value(
                                  j, face_handler_neigh.quadrature_ref(nq)) *
                                face_handler.quadrature_weight(q) *
                                measure_ratio;
                            }
                        }
                    }
                }
            }
        }
    }
  double error_semi_H1 = errors[2] * errors[2] - errors[1] * errors[1];
  errors[3]            = sqrt(error_semi_H1 + error_DG);
}


#endif /* ComputeErrorsDG_HPP_*/
