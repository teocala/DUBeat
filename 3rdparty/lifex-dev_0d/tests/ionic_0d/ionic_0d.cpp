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
 * @author Michele Bucelli <michele.bucelli@polimi.it>.
 */

#include "core/source/init.hpp"

#include "core/source/io/csv_test.hpp"

#include "source/ionic_0d.hpp"

/// Perform a test on a 0D ionic model, computing its solution and comparing it
/// to a reference one.
int
main(int argc, char **argv)
{
  lifex::lifex_init lifex_initializer(argc, argv, 1);

  try
    {
      using namespace lifex::utils;

      CSVTest<lifex::IonicModel0D, WithLinAlg::No> test("Ionic model");

      test.main_run_generate([&test]() {
        test.generate_parameters_from_json({"ttp06_endo"}, "ttp06_endo");
        test.generate_parameters_from_json({"ttp06_epi"}, "ttp06_epi");
        test.generate_parameters_from_json({"ttp06_myo"}, "ttp06_myo");
      });
    }
  LIFEX_CATCH_EXC();

  return EXIT_SUCCESS;
}
