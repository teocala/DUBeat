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

#include "../models/heat_dg.hpp"
#include "../models/laplace_dg.hpp"
#include "../models/monodomain_fitzhugh_dg.hpp"
#include "source/init.hpp"

/// This file is an example of main execution script. The model below can be
/// defined using the example models and adopting either DGFEM or Dubiner basis.

int
main(int argc, char **argv)
{
  lifex::lifex_init lifex_initializer(argc, argv, 1);

  try
    {
      // Choose the model from the models folder and the basis from:
      // Dubiner basis:    DUBValues<lifex::dim>
      // FEM basis:        dealii::FE_SimplexDGP<lifex::dim>
      lifex::examples::Monodomain_fitzhugh_DG<DUBValues<lifex::dim>> model;

      model.main_run_generate();
    }
  LIFEX_CATCH_EXC();

  return EXIT_SUCCESS;
}
