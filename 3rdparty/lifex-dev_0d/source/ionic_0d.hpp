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

#ifndef LIFEX_PHYSICS_IONIC_0D_HPP_
#define LIFEX_PHYSICS_IONIC_0D_HPP_

#include "source/ionic.hpp"

#include <memory>
#include <string>

namespace lifex
{
  /// @brief Class for 0D ionic model simulation.
  ///
  /// Allows to select with a parameter which ionic model to simulate.
  class IonicModel0D : public CoreModel
  {
  public:
    /// Constructor.
    IonicModel0D(const std::string &subsection_ionic_);

    /// Declare parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override;

    /// Parse input parameters.
    virtual void
    parse_parameters(ParamHandler &params) override;

    /// Run the simulation.
    virtual void
    run() override;

    /// Return the name of the file the output has been written to.
    std::string
    get_output_csv_filename() const
    {
      return ionic->get_output_csv_filename();
    }

  protected:
    std::unique_ptr<Ionic> ionic; ///< Ionic model to be simulated.

    /// @name Parameters read from file.
    /// @{

    std::string prm_ionic_model; ///< Name of the ionic model.

    bool prm_verbose; ///< Verbosity flag.

    /// @}

    std::string subsection_ionic; ///< File and subsection for the parameters of
                                  ///< the ionic model.
  };
} // namespace lifex

#endif /* LIFEX_PHYSICS_IONIC_0D_HPP_ */
