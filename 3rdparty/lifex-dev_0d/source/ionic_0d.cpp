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

#include "source/ttp06.hpp"
#include "source/ionic_0d.hpp"

namespace lifex
{
  IonicModel0D::IonicModel0D(const std::string &subsection_ionic_)
    : CoreModel(subsection_ionic_)
    , subsection_ionic(subsection_ionic_)
  {}

  void
  IonicModel0D::declare_parameters(ParamHandler &params) const
  {
    params.enter_subsection_path(prm_subsection_path);

    params.declare_entry_selection(
      "Ionic model",
      TTP06::label,
      Ionic::IonicFactory::get_registered_keys_prm());

    params.declare_entry("Verbose",
                         "true",
                         Patterns::Bool(),
                         "Toggle verbosity.");

    params.leave_subsection_path();

    // Dependencies.
    {
      Ionic::IonicFactory::declare_children_parameters_0d(params,
                                                          subsection_ionic);
    }
  }

  void
  IonicModel0D::parse_parameters(ParamHandler &params)
  {
    // Parse input file.
    params.parse();

    // Read input parameters.
    params.enter_subsection_path(prm_subsection_path);

    prm_ionic_model = params.get("Ionic model");
    prm_verbose     = params.get_bool("Verbose");

    params.leave_subsection_path();

    // Dependencies.
    {
      ionic = Ionic::IonicFactory::parse_child_parameters_0d(params,
                                                             prm_ionic_model,
                                                             subsection_ionic);
    }
  }

  void
  IonicModel0D::run()
  {
    ionic->run_0d(prm_verbose);
  }

} // namespace lifex
