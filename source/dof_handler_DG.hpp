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

#ifndef DOF_HANDLER_DG_HPP_
#define DOF_HANDLER_DG_HPP_

#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping_q1_eulerian.h>

#include <deal.II/lac/trilinos_vector.h>

#include <cmath>
#include <map>
#include <vector>

#include "DUBValues.hpp"
#include "source/init.hpp"


using ActiveSelector = dealii::internal::DoFHandlerImplementation::
  Iterators<lifex::dim, lifex::dim, false>;
using active_cell_iterator = typename ActiveSelector::active_cell_iterator;

/**
 * @brief
 * Class to work with global and local degrees of freedom and their mapping.
 * DUBeat exploits this class instead of the deal.II DoFHandler class because
 * the latter, at the moment, cannot distribute dofs on tethraedra with
 * polynomial orders greater than 2. This implementation permits to overcome
 * this issue thanks to the use of an internal dof_map and the definition of
 * Dubiner basis of every order (even >2) from DUBValues. On the other hand, it
 * is not possible to do the same with DGFEM because the basis functions, in
 * this case, come directly from the deal.II FiniteElement classes. From the
 * previous observations, it is so clear that Dubiner basis can be used with
 * every order while DGFEM with order at most 2.
 */
template <class basis>
class DoFHandlerDG : public dealii::DoFHandler<lifex::dim>
{
private:
  /// Polynomial space degree.
  unsigned int degree;

  /// Local to global dof map.
  std::map<active_cell_iterator, std::vector<unsigned int>> dof_map;

public:
  /// Constructor.
  DoFHandlerDG<basis>()
    : dealii::DoFHandler<lifex::dim>()
    , degree(0)
  {}

  /// Default copy constructor.
  DoFHandlerDG<basis>(DoFHandlerDG<basis> &DoFHandlerDG) = default;

  /// Default const copy constructor.
  DoFHandlerDG<basis>(const DoFHandlerDG<basis> &DOFHandlerDG) = default;

  /// Default move constructor.
  DoFHandlerDG<basis>(DoFHandlerDG<basis> &&DoFHandlerDG) = default;

  /// Return copy of n_dofs_per_cell.
  unsigned int
  n_dofs_per_cell() const;

  /// Return copy of n_dofs.
  unsigned int
  n_dofs() const;

  /// Distribute dofs through the elements. It overwrites the method in
  /// dealii::DoFHandler in order to let higher order polynomials available in
  /// the case of Dubiner basis.
  void
  distribute_dofs(const dealii::FE_SimplexDGP<lifex::dim> &fe);

  /// Same method but avoids to use FiniteElement classes that might be invalid
  /// with higher order polynomials.
  void
  distribute_dofs(const unsigned int degree);

  /// Returns the global dofs referred to the input cell.
  std::vector<lifex::types::global_dof_index>
  get_dof_indices(active_cell_iterator cell) const;

  /// Return a set of all the locally owned dofs (for the time being, it is
  /// equivalent to return all the dofs).
  dealii::IndexSet
  locally_owned_dofs() const;
};

template <class basis>
unsigned int
DoFHandlerDG<basis>::n_dofs_per_cell() const
{
  AssertThrow(degree > 0,
              dealii::StandardExceptions::ExcMessage(
                "Dofs have not been distributed yet. Please, use "
                "distribute_dofs before."));

  // The analytical formula is:
  // n_dof_per_cell = (p+1)*(p+2)*...(p+d) / d!,
  // where p is the space order and d the space dimension..

  unsigned int denominator = 1;
  unsigned int nominator   = 1;

  for (unsigned int i = 1; i <= lifex::dim; i++)
    {
      denominator *= i;
      nominator *= degree + i;
    }

  return (int)(nominator / denominator);
}

template <class basis>
unsigned int
DoFHandlerDG<basis>::n_dofs() const
{
  AssertThrow(degree > 0,
              dealii::StandardExceptions::ExcMessage(
                "Dofs have not been distributed yet. Please, use "
                "distribute_dofs before."));

  unsigned int n_cells = 0;

  for (const auto &cell : this->active_cell_iterators())
    n_cells++;

  // The global nuber of dofs is the number of dofs per cell times the total
  // number of cells since every cell has the same number of dofs.
  return n_cells * n_dofs_per_cell();
}


/// Specialized version for DGFEM, it does not use dof_map but exploits instead
/// the deal.II original distribute_dofs. Limited to order 2.
template <>
void
DoFHandlerDG<dealii::FE_SimplexDGP<lifex::dim>>::distribute_dofs(
  const dealii::FE_SimplexDGP<lifex::dim> &fe)
{
  AssertThrow(fe.degree < 3,
              dealii::StandardExceptions::ExcMessage(
                "deal.II library does not provide yet FEM spaces with "
                "polynomial order > 2."));
  degree = fe.degree;
  dealii::DoFHandler<lifex::dim>::distribute_dofs(fe);
}

/// Specialized version for Dubiner basis, this version uses instead the
/// internal dof_map. Because of the input argument, the space order is still
/// limited to 2.
template <>
void
DoFHandlerDG<DUBValues<lifex::dim>>::distribute_dofs(
  const dealii::FE_SimplexDGP<lifex::dim> &fe)
{
  AssertThrow(fe.degree < 3,
              dealii::StandardExceptions::ExcMessage(
                "deal.II library does not provide yet FEM spaces with "
                "polynomial order > 2."));
  dealii::DoFHandler<lifex::dim>::distribute_dofs(fe);
  degree = fe.degree;

  std::vector<lifex::types::global_dof_index> dof_indices(
    this->n_dofs_per_cell());
  dof_map.clear();

  for (const auto &cell : this->active_cell_iterators())
    {
      // In the dof_map, for every cell we assign the global dofs.
      cell->get_dof_indices(dof_indices);
      dof_map.emplace(cell, dof_indices);
    }
}

/// Specialized version for DGFEM, it exploits the original distribute_dofs and
/// so it is limited to space order at most 2.
template <>
void
DoFHandlerDG<dealii::FE_SimplexDGP<lifex::dim>>::distribute_dofs(
  const unsigned int degree)
{
  AssertThrow(degree,
              dealii::StandardExceptions::ExcMessage(
                "deal.II library does not provide yet DGFEM spaces with "
                "polynomial order > 2."));
  dealii::FE_SimplexDGP<lifex::dim> fe(degree);
  this->distribute_dofs(fe);
}


/// Specialized version for Dubiner basis. Since the input is the polynomial
/// order and it works only with the internal dof_map (no FiniteElement
/// classes), this version of distribute_dofs is the only one to accept every
/// polynomial order.
template <>
void
DoFHandlerDG<DUBValues<lifex::dim>>::distribute_dofs(const unsigned int degree)
{
  this->degree                 = degree;
  unsigned int n_dofs_per_cell = this->n_dofs_per_cell();
  dof_map.clear();
  unsigned int n = 0;
  for (const auto &cell : this->active_cell_iterators())
    {
      std::vector<unsigned int> local_dofs;

      for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
        local_dofs.push_back(n + i);

      dof_map.emplace(cell, local_dofs);

      n = n + n_dofs_per_cell;
    }
}

/// Specialized version for DGFEM. Hence, it exploits the original deal.II
/// methods.
template <>
std::vector<lifex::types::global_dof_index>
DoFHandlerDG<dealii::FE_SimplexDGP<lifex::dim>>::get_dof_indices(
  active_cell_iterator cell) const
{
  std::vector<lifex::types::global_dof_index> dof_indices(
    this->n_dofs_per_cell());
  cell->get_dof_indices(dof_indices);
  return dof_indices;
}


/// Specialized version for Dubiner basis. Hence, the dof_indices are obtained
/// directly from the internal dof_map.
template <>
std::vector<lifex::types::global_dof_index>
DoFHandlerDG<DUBValues<lifex::dim>>::get_dof_indices(
  active_cell_iterator cell) const
{
  std::vector<lifex::types::global_dof_index> dof_indices(
    this->n_dofs_per_cell());
  dof_indices = dof_map.at(cell);
  return dof_indices;
}

template <class basis>
dealii::IndexSet
DoFHandlerDG<basis>::locally_owned_dofs() const
{
  dealii::IndexSet owned_dofs(this->n_dofs());
  // For the time being, this function returns all the dofs.
  owned_dofs.add_range(0, this->n_dofs());
  return owned_dofs;
}


#endif /* DOF_HANDLER_DG_HPP_*/
