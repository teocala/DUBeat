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

#ifndef DGAssemble_HPP_
#define DGAssemble_HPP_

#include <deal.II/base/parameter_handler.h>

#include <deal.II/fe/mapping_q1_eulerian.h>

#include <deal.II/lac/full_matrix.h>

#include <memory>
#include <utility>
#include <vector>

#include "DG_Face_handler.hpp"
#include "DG_Volume_handler.hpp"
#include "DUB_FEM_handler.hpp"
#include "source/core_model.hpp"
#include "source/geometry/mesh_handler.hpp"
#include "source/init.hpp"
#include "source/io/data_writer.hpp"
#include "source/numerics/bc_handler.hpp"
#include "source/numerics/linear_solver_handler.hpp"
#include "source/numerics/preconditioner_handler.hpp"
#include "source/numerics/tools.hpp"

/**
 * @brief Class for the assembly of the main local matrices for discontinuous
 * Galerkin methods.
 *
 * Definition of the main terms:
 *  - @f$\mathcal{K}@f$ as the volume of a generic element.
 *  - @f$\mathcal{F}@f$ as a generic face of one element.
 *  - @f$\varphi_i@f$ as the @f$i@f$-th basis function of the element.
 *  - The @f$+@f$ superscript to indicate that a function evaluated on the face
 * is the one associated to the same element.
 *  - The @f$-@f$ superscript to indicate that a function evaluated on the face
 * is the one associated to the neighbor element.
 *  - @f$n^+, n^-@f$ as the outward/inward normal vector with respect to one
 * element face.
 */
template <class basis>
class DGAssemble
{
private:
  /// Basis function class.
  const std::unique_ptr<basis> basis_ptr;

  /// Volume element handler.
  DGVolumeHandler<lifex::dim> vol_handler;

  /// Face element handler.
  DGFaceHandler<lifex::dim> face_handler;

  /// Face element handler of the neighbor face.
  DGFaceHandler<lifex::dim> face_handler_neigh;

  /// Degree of freedom for each cell.
  unsigned int dofs_per_cell;

  /// Number of quadrature points in 1D.
  const unsigned int n_quad_points_1D;

  /// Number of quadrature points in the volume: (n_quad_points_1D)^(dim).
  const unsigned int n_quad_points;

  /// Number of quadrature points in the face: (n_quad_points_1D)^(dim-1).
  const unsigned int n_quad_points_face;

  /// Polynomial degree.
  const unsigned int poly_degree;

  /// Cell object.
  typename dealii::DoFHandler<lifex::dim>::active_cell_iterator cell;

public:
  /// Constructor.
  DGAssemble<basis>(const unsigned int degree)
    : basis_ptr(std::make_unique<basis>(degree))
    , vol_handler(DGVolumeHandler<lifex::dim>(degree))
    , face_handler(DGFaceHandler<lifex::dim>(degree))
    , face_handler_neigh(DGFaceHandler<lifex::dim>(degree))
    , n_quad_points_1D(degree + 2)
    , n_quad_points(static_cast<int>(std::pow(n_quad_points_1D, lifex::dim)))
    , n_quad_points_face(
        static_cast<int>(std::pow(n_quad_points_1D, lifex::dim - 1)))
    , poly_degree(degree)
  {
    dofs_per_cell = this->get_dofs_per_cell();
  }

  /// Default copy constructor.
  DGAssemble<basis>(DGAssemble<basis> &) = default;

  /// Default const copy constructor.
  DGAssemble<basis>(const DGAssemble<basis> &) = default;

  /// Default move constructor.
  DGAssemble<basis>(DGAssemble<basis> &&) = default;

  /// Reinitialize object on the current new_edge of the new_cell.
  void
  reinit(const typename dealii::DoFHandler<lifex::dim>::active_cell_iterator
           &                new_cell,
         const unsigned int new_edge);

  /// Reinitialize object on the current new_cell.
  void
  reinit(const typename dealii::DoFHandler<lifex::dim>::active_cell_iterator
           &new_cell);

  /// Return the number of degrees of freedom per element.
  unsigned int
  get_dofs_per_cell() const;

  /// Assembly of stiffness matrix: @f[V(i,j)=\int_{\mathcal{K}} \nabla
  /// \varphi_j \cdot \nabla \varphi_i \,dx @f]
  dealii::FullMatrix<double>
  local_V() const;

  /// Assembly of local mass matrix: @f[M(i,j)=\int_{\mathcal{K}} \varphi_j
  /// \varphi_i
  /// \, dx @f]
  dealii::FullMatrix<double>
  local_M() const;

  /// Assembly of the local forcing term: @f[F(i)=\int_{\mathcal{K}} f
  /// \varphi_i
  /// \, dx
  /// @f]
  /// where @f$ f @f$ is the known forcing term of the problem.
  dealii::Vector<double>
  local_rhs(const std::shared_ptr<dealii::Function<lifex::dim>> &f_ex) const;

  /// Assembly of the local matrix associated to non-homogeneous Dirichlet
  /// boundary conditions: @f[F(i)=\int_{\mathcal{F}} \gamma u \varphi_i^{+} -
  /// u \nabla \varphi_i^{+}  \cdot n \,ds @f] where @f$\gamma@f$ is the
  /// regularity coeficient and @f$u@f$ the known solution of the problem.
  dealii::Vector<double>
  local_rhs_edge_dirichlet(
    const double stability_coefficient,
    const std::shared_ptr<lifex::utils::FunctionDirichlet> &u_ex) const;

  /// Assembly of the right hand side term associated to non-homogeneous
  /// Neumann boundary conditions: @f[F(i)=\int_{\mathcal{F}} \varphi_i g \, ds
  /// @f] where @f$g@f$ is the gradient of the known solution of the problem.
  dealii::Vector<double>
  local_rhs_edge_neumann(
    const std::shared_ptr<dealii::Function<lifex::dim>> &g) const;

  /// Assembly of the local matrix S: @f[S(i,j)=\int_{\mathcal{F}} \gamma
  /// \varphi_j^{+} \varphi_i^{+} \, ds @f] where @f$ \gamma @f$ is the
  /// regularity coefficient.
  dealii::FullMatrix<double>
  local_S(const double stability_coefficient) const;

  /// Assembly of the local matrix S at the interface with the adjacent
  /// element:
  /// @f[ SN(i,j)=- \int_{\mathcal{F}} \gamma \varphi_j^{+} \varphi_i^{-} \, ds
  /// @f]
  dealii::FullMatrix<double>
  local_SN(const double stability_coefficient) const;

  /// Assembly of the local matrix I and its transpose:
  ///@f[
  /// \begin{aligned}
  /// I(i,j)= & \frac{1}{2}
  /// \int_{\mathcal{F}} \nabla \varphi_i^{+} \cdot  n^{+} \varphi_j^{+}  \, ds
  /// \\ I_T(i,j)= & - \frac{\theta}{2} \int_{\mathcal{F}} \nabla \varphi_j^{+}
  /// \cdot  n^{+} \varphi_i^{+}  \, ds \end{aligned} @f] where @f$\theta@f$ is
  /// the penalty coefficient.
  std::pair<dealii::FullMatrix<double>, dealii::FullMatrix<double>>
  local_I(const double theta) const;

  /// Assembly of the local matrix I and its transpose on the boundary edges:
  /// @f[\begin{aligned}
  /// IB(i,j)=& \: \theta \int_{\mathcal{F}} \nabla \varphi_i^{+} \cdot n^{+}
  /// \varphi_j^{+} \, ds \\ IB_T(i,j)=& - \int_{\mathcal{F}} \nabla
  /// \varphi_j^{+} \cdot n^{+} \varphi_i^{+} \, ds \end{aligned} @f] where
  /// @f$\theta@f$ is the penalty coefficient.
  std::pair<dealii::FullMatrix<double>, dealii::FullMatrix<double>>
  local_IB(const double theta) const;

  /// Assembly of the local matrix I and its transpose at the interface with
  /// the adjacent element: @f[\begin{aligned} IN(i,j)=& \frac{\theta}{2}
  /// \int_{\mathcal{F}} \nabla \varphi_i^{+} \cdot n^{+} \varphi_j^{-}  \, ds
  /// \\ IN_T(i,j)=& -\frac{1}{2} \int_{\mathcal{F}} \nabla \varphi_j^{+} \cdot
  /// n^{+} \varphi_i^{-}  \, ds \end{aligned} @f] where @f$\theta@f$ is the
  /// penalty coefficient.
  std::pair<dealii::FullMatrix<double>, dealii::FullMatrix<double>>
  local_IN(const double theta) const;

  /// Assembly of the right hand side term associated to the previous time-step
  /// solution in time-dependent problems:
  /// @f[F(i)= \int_{\mathcal{K}} u^n \varphi_i \, dx @f]
  /// where @f$u^n@f$ is the solution at a generic previous step @f$ n@f$.
  dealii::Vector<double>
  local_u0_M_rhs(
    const lifex::LinAlg::MPI::Vector &                  u0,
    const std::vector<dealii::types::global_dof_index> &dof_indices) const;

  /// Assembly of the right hand side term associated to the previous time-step
  /// gating variable solution (for monodomain problem):
  /// @f[F(i)=\int_{\mathcal{K}} w^n \varphi_i \, dx @f]
  /// where @f$w^n@f$ is the gating variable solution at a generic previous
  /// step
  /// @f$ n@f$.
  dealii::Vector<double>
  local_w0_M_rhs(
    const lifex::LinAlg::MPI::Vector &                  u0,
    const std::vector<dealii::types::global_dof_index> &dof_indices) const;

  /// Assembly of the non-linear local matrix of the Fitzhugh-Nagumo model:
  /// @f[C(i,j)=\int_{\mathcal{K}} \chi_m k
  /// (u^n-1)(u^n-a) \varphi_j \varphi_i \, dx @f]
  /// where @f$\chi_m@f$ and @f$k@f$ are parameters of the monodomain equation
  /// and @f$u^n@f$ is the solution at a generic previous step @f$ n@f$.
  dealii::FullMatrix<double>
  local_non_linear_fitzhugh(
    const lifex::LinAlg::MPI::Vector &                  u0,
    const double                                        a,
    const std::vector<dealii::types::global_dof_index> &dof_indices) const;

  /// Destructor.
  virtual ~DGAssemble() = default;
};

template <class basis>
void
DGAssemble<basis>::reinit(
  const typename dealii::DoFHandler<lifex::dim>::active_cell_iterator &new_cell,
  const unsigned int                                                   new_edge)
{
  cell = new_cell;
  face_handler.reinit(new_cell, new_edge);

  if (!cell->at_boundary(new_edge))
    {
      const auto neighcell = cell->neighbor(new_edge);
      const auto neighedge = cell->neighbor_face_no(new_edge);

      face_handler_neigh.reinit(neighcell, neighedge);
    }
}

template <class basis>
void
DGAssemble<basis>::reinit(
  const typename dealii::DoFHandler<lifex::dim>::active_cell_iterator &new_cell)
{
  cell = new_cell;
  vol_handler.reinit(new_cell);
}

template <class basis>
unsigned int
DGAssemble<basis>::get_dofs_per_cell() const
{
  // The analytical formula is:
  // n_dof_per_cell = (p+1)*(p+2)*...(p+d) / d!,
  // where p is the space order and d the space dimension..

  unsigned int denominator = 1;
  unsigned int nominator   = 1;

  for (unsigned int i = 1; i <= lifex::dim; i++)
    {
      denominator *= i;
      nominator *= poly_degree + i;
    }

  return (int)(nominator / denominator);
}

template <class basis>
dealii::FullMatrix<double>
DGAssemble<basis>::local_V() const
{
  dealii::FullMatrix<double>          V(dofs_per_cell, dofs_per_cell);
  const dealii::Tensor<2, lifex::dim> BJinv =
    vol_handler.get_jacobian_inverse();
  const double det = 1 / determinant(BJinv);

  for (unsigned int q = 0; q < n_quad_points; ++q)
    {
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              V(i, j) +=
                (basis_ptr->shape_grad(i, vol_handler.quadrature_ref(q)) *
                 BJinv) *
                (basis_ptr->shape_grad(j, vol_handler.quadrature_ref(q)) *
                 BJinv) *
                vol_handler.quadrature_weight(q) * det;
            }
        }
    }

  return V;
}

template <class basis>
dealii::FullMatrix<double>
DGAssemble<basis>::local_M() const
{
  dealii::FullMatrix<double>          M(dofs_per_cell, dofs_per_cell);
  const dealii::Tensor<2, lifex::dim> BJinv =
    vol_handler.get_jacobian_inverse();
  const double det = 1 / determinant(BJinv);

  for (unsigned int q = 0; q < n_quad_points; ++q)
    {
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              M(i, j) +=
                (basis_ptr->shape_value(i, vol_handler.quadrature_ref(q))) *
                (basis_ptr->shape_value(j, vol_handler.quadrature_ref(q))) *
                vol_handler.quadrature_weight(q) * det;
            }
        }
    }

  return M;
}

template <class basis>
dealii::Vector<double>
DGAssemble<basis>::local_rhs(
  const std::shared_ptr<dealii::Function<lifex::dim>> &f_ex) const
{
  AssertThrow(f_ex != nullptr,
              dealii::StandardExceptions::ExcMessage(
                "Not valid pointer to the source term."));

  dealii::Vector<double>              cell_rhs(dofs_per_cell);
  const dealii::Tensor<2, lifex::dim> BJinv =
    vol_handler.get_jacobian_inverse();
  const double det = 1 / determinant(BJinv);

  for (unsigned int q = 0; q < n_quad_points; ++q)
    {
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          cell_rhs(i) +=
            f_ex->value(vol_handler.quadrature_real(q)) *
            basis_ptr->shape_value(i, vol_handler.quadrature_ref(q)) *
            vol_handler.quadrature_weight(q) * det;
        }
    }

  return cell_rhs;
}

template <class basis>
dealii::Vector<double>
DGAssemble<basis>::local_rhs_edge_dirichlet(
  const double                                            stability_coefficient,
  const std::shared_ptr<lifex::utils::FunctionDirichlet> &u_ex) const
{
  AssertThrow(u_ex != nullptr,
              dealii::StandardExceptions::ExcMessage(
                "Not valid pointer to the exact solution."));

  dealii::Vector<double>              cell_rhs(dofs_per_cell);
  const dealii::Tensor<2, lifex::dim> BJinv =
    face_handler.get_jacobian_inverse();

  const double face_measure      = face_handler.get_measure();
  const double unit_face_measure = (4.0 - lifex::dim) / 2;
  const double measure_ratio     = face_measure / unit_face_measure;
  const double h_local           = (cell->measure()) / face_measure;

  const double local_pen_coeff =
    (stability_coefficient * pow(poly_degree, 2)) / h_local;

  for (unsigned int q = 0; q < n_quad_points_face; ++q)
    {
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          cell_rhs(i) +=
            local_pen_coeff *
            basis_ptr->shape_value(i, face_handler.quadrature_ref(q)) *
            u_ex->value(face_handler.quadrature_real(q)) *
            face_handler.quadrature_weight(q) * measure_ratio;

          cell_rhs(i) -=
            ((basis_ptr->shape_grad(i, face_handler.quadrature_ref(q)) *
              BJinv) *
             face_handler.get_normal()) *
            u_ex->value(face_handler.quadrature_real(q)) *
            face_handler.quadrature_weight(q) * measure_ratio;
        }
    }

  return cell_rhs;
}

template <class basis>
dealii::Vector<double>
DGAssemble<basis>::local_rhs_edge_neumann(
  const std::shared_ptr<dealii::Function<lifex::dim>> &g) const
{
  AssertThrow(g != nullptr,
              dealii::StandardExceptions::ExcMessage(
                "Not valid pointer to the Neumann function."));

  dealii::Vector<double> cell_rhs(dofs_per_cell);

  const double face_measure      = face_handler.get_measure();
  const double unit_face_measure = (4.0 - lifex::dim) / 2;
  const double measure_ratio     = face_measure / unit_face_measure;

  for (unsigned int q = 0; q < n_quad_points_face; ++q)
    {
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          cell_rhs(i) +=
            basis_ptr->shape_value(i, face_handler.quadrature_ref(q)) *
            g->value(face_handler.quadrature_real(q)) *
            face_handler.quadrature_weight(q) * measure_ratio;
        }
    }

  return cell_rhs;
}

template <class basis>
dealii::FullMatrix<double>
DGAssemble<basis>::local_S(const double stability_coefficient) const
{
  const double face_measure      = face_handler.get_measure();
  const double unit_face_measure = (4.0 - lifex::dim) / 2;
  const double measure_ratio     = face_measure / unit_face_measure;
  const double h_local           = (cell->measure()) / face_measure;

  const double local_pen_coeff =
    (stability_coefficient * pow(poly_degree, 2)) / h_local;

  dealii::FullMatrix<double> S(dofs_per_cell, dofs_per_cell);

  for (unsigned int q = 0; q < n_quad_points_face; ++q)
    {
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              S(i, j) +=
                local_pen_coeff *
                basis_ptr->shape_value(i, face_handler.quadrature_ref(q)) *
                basis_ptr->shape_value(j, face_handler.quadrature_ref(q)) *
                face_handler.quadrature_weight(q) * measure_ratio;
            }
        }
    }

  return S;
}

template <class basis>
dealii::FullMatrix<double>
DGAssemble<basis>::local_SN(const double stability_coefficient) const
{
  const double face_measure      = face_handler.get_measure();
  const double unit_face_measure = (4.0 - lifex::dim) / 2;
  const double measure_ratio     = face_measure / unit_face_measure;
  const double h_local           = (cell->measure()) / face_measure;

  const double local_pen_coeff =
    (stability_coefficient * pow(poly_degree, 2)) / h_local;

  dealii::FullMatrix<double> SN(dofs_per_cell, dofs_per_cell);

  for (unsigned int q = 0; q < n_quad_points_face; ++q)
    {
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              const unsigned int nq =
                face_handler.corresponding_neigh_index(q, face_handler_neigh);

              SN(i, j) -=
                local_pen_coeff *
                basis_ptr->shape_value(i, face_handler.quadrature_ref(q)) *
                basis_ptr->shape_value(j,
                                       face_handler_neigh.quadrature_ref(nq)) *
                face_handler.quadrature_weight(q) * measure_ratio;
            }
        }
    }

  return SN;
}

template <class basis>
std::pair<dealii::FullMatrix<double>, dealii::FullMatrix<double>>
DGAssemble<basis>::local_I(const double theta) const
{
  AssertThrow(theta == 1. || theta == 0. || theta == -1.,
              dealii::StandardExceptions::ExcMessage(
                "Penalty coefficient must be 1 (SIP method) or 0 (IIP method) "
                "or -1 (NIP method)."));

  const dealii::Tensor<2, lifex::dim> BJinv =
    face_handler.get_jacobian_inverse();

  const double face_measure      = face_handler.get_measure();
  const double unit_face_measure = (4.0 - lifex::dim) / 2;
  const double measure_ratio     = face_measure / unit_face_measure;

  dealii::FullMatrix<double> I(dofs_per_cell, dofs_per_cell);
  dealii::FullMatrix<double> I_t(dofs_per_cell, dofs_per_cell);

  for (unsigned int q = 0; q < n_quad_points_face; ++q)
    {
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              I(i, j) +=
                0.5 *
                ((basis_ptr->shape_grad(i, face_handler.quadrature_ref(q)) *
                  BJinv) *
                 face_handler.get_normal()) *
                basis_ptr->shape_value(j, face_handler.quadrature_ref(q)) *
                face_handler.quadrature_weight(q) * measure_ratio;
            }
        }
    }

  I_t.copy_transposed(I);
  I_t *= (-1);
  I *= theta;

  return {I, I_t};
}

template <class basis>
std::pair<dealii::FullMatrix<double>, dealii::FullMatrix<double>>
DGAssemble<basis>::local_IB(const double theta) const
{
  AssertThrow(theta == 1. || theta == 0. || theta == -1.,
              dealii::StandardExceptions::ExcMessage(
                "Penalty coefficient must be 1 (SIP method) or 0 (IIP method) "
                "or -1 (NIP method)."));

  const dealii::Tensor<2, lifex::dim> BJinv =
    face_handler.get_jacobian_inverse();

  const double face_measure      = face_handler.get_measure();
  const double unit_face_measure = (4.0 - lifex::dim) / 2;
  const double measure_ratio     = face_measure / unit_face_measure;

  dealii::FullMatrix<double> IB(dofs_per_cell, dofs_per_cell);
  dealii::FullMatrix<double> IB_t(dofs_per_cell, dofs_per_cell);

  for (unsigned int q = 0; q < n_quad_points_face; ++q)
    {
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              IB(i, j) +=
                ((basis_ptr->shape_grad(i, face_handler.quadrature_ref(q)) *
                  BJinv) *
                 face_handler.get_normal()) *
                basis_ptr->shape_value(j, face_handler.quadrature_ref(q)) *
                face_handler.quadrature_weight(q) * measure_ratio;
            }
        }
    }

  IB_t.copy_transposed(IB);
  IB_t *= (-1);
  IB *= theta;

  return {IB, IB_t};
}

template <class basis>
std::pair<dealii::FullMatrix<double>, dealii::FullMatrix<double>>
DGAssemble<basis>::local_IN(const double theta) const
{
  AssertThrow(theta == 1. || theta == 0. || theta == -1.,
              dealii::StandardExceptions::ExcMessage(
                "Penalty coefficient must be 1 (SIP method) or 0 (IIP method) "
                "or -1 (NIP method)."));

  const dealii::Tensor<2, lifex::dim> BJinv =
    face_handler.get_jacobian_inverse();

  const double face_measure      = face_handler.get_measure();
  const double unit_face_measure = (4.0 - lifex::dim) / 2;
  const double measure_ratio     = face_measure / unit_face_measure;

  dealii::FullMatrix<double> IN(dofs_per_cell, dofs_per_cell);
  dealii::FullMatrix<double> IN_t(dofs_per_cell, dofs_per_cell);

  for (unsigned int q = 0; q < n_quad_points_face; ++q)
    {
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              const unsigned int nq =
                face_handler.corresponding_neigh_index(q, face_handler_neigh);

              IN(i, j) -=
                0.5 *
                ((basis_ptr->shape_grad(i, face_handler.quadrature_ref(q)) *
                  BJinv) *
                 face_handler.get_normal()) *
                basis_ptr->shape_value(j,
                                       face_handler_neigh.quadrature_ref(nq)) *
                face_handler.quadrature_weight(q) * measure_ratio;
            }
        }
    }

  IN_t.copy_transposed(IN);
  IN_t *= (-1);
  IN *= theta;

  return {IN, IN_t};
}

template <class basis>
dealii::Vector<double>
DGAssemble<basis>::local_u0_M_rhs(
  const lifex::LinAlg::MPI::Vector &                  u0,
  const std::vector<dealii::types::global_dof_index> &dof_indices) const
{
  std::vector<double> u_bdf_loc(n_quad_points);
  std::fill(u_bdf_loc.begin(), u_bdf_loc.end(), 0);

  for (unsigned int q = 0; q < n_quad_points; ++q)
    {
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          u_bdf_loc[q] +=
            u0[dof_indices[i]] *
            basis_ptr->shape_value(i, vol_handler.quadrature_ref(q));
        }
    }

  dealii::Vector<double>              cell_rhs(dofs_per_cell);
  const dealii::Tensor<2, lifex::dim> BJinv =
    vol_handler.get_jacobian_inverse();
  const double det = 1 / determinant(BJinv);

  for (unsigned int q = 0; q < n_quad_points; ++q)
    {
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          cell_rhs(i) +=
            u_bdf_loc[q] *
            basis_ptr->shape_value(i, vol_handler.quadrature_ref(q)) *
            vol_handler.quadrature_weight(q) * det;
        }
    }

  return cell_rhs;
}

template <class basis>
dealii::Vector<double>
DGAssemble<basis>::local_w0_M_rhs(
  const lifex::LinAlg::MPI::Vector &                  u0,
  const std::vector<dealii::types::global_dof_index> &dof_indices) const
{
  dealii::Vector<double>              cell_rhs(dofs_per_cell);
  const dealii::Tensor<2, lifex::dim> BJinv =
    vol_handler.get_jacobian_inverse();
  const double det = 1 / determinant(BJinv);

  for (unsigned int q = 0; q < n_quad_points; ++q)
    {
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              cell_rhs(i) +=
                (basis_ptr->shape_value(i, vol_handler.quadrature_ref(q))) *
                (basis_ptr->shape_value(j, vol_handler.quadrature_ref(q))) *
                vol_handler.quadrature_weight(q) * det * u0[dof_indices[j]];
            }
        }
    }

  return cell_rhs;
}

template <class basis>
dealii::FullMatrix<double>
DGAssemble<basis>::local_non_linear_fitzhugh(
  const lifex::LinAlg::MPI::Vector &                  u0,
  const double                                        a,
  const std::vector<dealii::types::global_dof_index> &dof_indices) const
{
  dealii::FullMatrix<double>          C(dofs_per_cell, dofs_per_cell);
  const dealii::Tensor<2, lifex::dim> BJinv =
    vol_handler.get_jacobian_inverse();
  const double det = 1 / determinant(BJinv);


  double non_lu;
  double non_lin;

  for (unsigned int q = 0; q < n_quad_points; ++q)
    {
      non_lu = 0.0;

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          non_lu += u0[dof_indices[i]] *
                    basis_ptr->shape_value(i, vol_handler.quadrature_ref(q));
        }

      non_lin = (non_lu - 1) * (non_lu - a);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              C(i, j) +=
                non_lin *
                basis_ptr->shape_value(i, vol_handler.quadrature_ref(q)) *
                basis_ptr->shape_value(j, vol_handler.quadrature_ref(q)) *
                vol_handler.quadrature_weight(q) * det;
            }
        }
    }

  return C;
}

#endif /* DGAssemble_HPP_*/
