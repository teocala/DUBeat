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
 * @author Matteo Salvador <matteo1.salvador@polimi.it>.
 * @author Francesco Regazzoni <francesco.regazzoni@polimi.it>.
 */

#ifndef LIFEX_IONIC_TTP06_HPP_
#define LIFEX_IONIC_TTP06_HPP_

#include "source/ionic.hpp"

#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace lifex0d
{
  /**
   * @brief ten Tusscher-Panfilov (2006) ionic model.
   *
   * Reference for the ionic model:
   * https://doi.org/10.1152/ajpheart.00109.2006.
   *
   * Reference for ischemic regions remodeling:
   * http://dx.doi.org/10.1038/ncomms11437
   *
   */
  class TTP06 : public Ionic
  {
  public:
    /// This class needs to access the value of prm_implicit_Iion.
    friend class Electrophysiology;

    /// Ionic model label.
    static inline constexpr auto label = "TTP06";

    /// Alias for array of currents.
    template <class NumberType>
    using CurrentType = std::array<NumberType, 12>;

    /// Constructor.
    TTP06(const std::string &subsection);

    /// Declare parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override;

    /// Parse parameters.
    virtual void
    parse_parameters(ParamHandler &params) override;

    /// Set the initial conditions for the ionic variables.
    virtual std::vector<double>
    setup_initial_conditions() const override;

    /// Set the initial conditions for the transmembrane potential,
    /// to be used in the associated electrophysiology problem.
    virtual double
    setup_initial_transmembrane_potential() const override;

    /// Solve 0D time step, given the transmembrane potential.
    virtual std::pair<std::vector<double>, unsigned int>
    solve_time_step_0d(const double &             u,
                       const double &             alpha_bdf,
                       const std::vector<double> &w_bdf,
                       const std::vector<double> &w_ext,
                       const double &             cell_type,
                       const double &             Iapp) override;

    /// Evaluate the ionic current given the transmembrane potential and the
    /// ionic model variables.
    virtual std::pair<double, double>
    Iion(const double &             u,
         const double &             u_old,
         const std::vector<double> &w,
         const double &             cell_type) override;

    /// @copydoc Ionic::get_repolarization_upper_threshold
    virtual double
    get_repolarization_upper_threshold() const override
    {
      return -0.03;
    }

    /// @copydoc Ionic::get_repolarization_lower_threshold
    virtual double
    get_repolarization_lower_threshold() const override
    {
      return -0.05;
    }

  protected:
    /// Compute the raw calcium concentration (i.e. pre-rescaling) given the
    /// ionic variables.
    virtual double
    compute_calcium_raw(const std::vector<double> &w) const override;

    /// Return a properly formatted string for the output of solver iterations.
    virtual std::string
    iterations_log_string(const unsigned int &n_iter) override;

    /// Compute all the currents of the model.
    template <class NumberType>
    void
    compute_currents(const NumberType &       VV,
                     CurrentType<NumberType> &currents);

    /// Compute the derivatives of the concentration variables of the model.
    void
    compute_concentration_derivatives(const double &Iapp);

    /// Compute constants useful for the evolution time of the gating variables
    /// of the model.
    void
    compute_gating_constants(const double &CaSS);


    /// @name Parameters read from file.
    /// @{

    bool prm_implicit_Iion; ///< If true, dIion_du is assembled using automatic
                            ///< differentiation.
    double prm_membrane_capacitance; ///< @f$[\mathrm{1}]@f$.
    double prm_capacitance;          ///< @f$[\mathrm{1}]@f$.
    double prm_Ko;                   ///< @f$[\mathrm{mM}]@f$.
    double prm_Cao;                  ///< @f$[\mathrm{mM}]@f$.
    double prm_Nao;                  ///< @f$[\mathrm{mM}]@f$.
    double prm_Vc;                   ///< @f$[\mathrm{m^3}]@f$.
    double prm_Vsr;                  ///< @f$[\mathrm{m^3}]@f$.
    double prm_Vss;                  ///< @f$[\mathrm{m^3}]@f$.
    double prm_Bufc;                 ///< @f$[\mathrm{mM}]@f$.
    double prm_Kbufc;                ///< @f$[\mathrm{mM}]@f$.
    double prm_Bufsr;                ///< @f$[\mathrm{mM}]@f$.
    double prm_Kbufsr;               ///< @f$[\mathrm{mM}]@f$.
    double prm_Bufss;                ///< @f$[\mathrm{mM}]@f$.
    double prm_Kbufss;               ///< @f$[\mathrm{mM}]@f$.
    double prm_Vmaxup;               ///< @f$[\mathrm{mM/ms}]@f$.
    double prm_Kup;                  ///< @f$[\mathrm{mM}]@f$.
    double prm_Vrel;                 ///< @f$[\mathrm{mM/ms}]@f$.
    double prm_k1_;                  ///< @f$[\mathrm{mM^{-2}ms^{-1}}]@f$.
    double prm_k2_;                  ///< @f$[\mathrm{mM^{-1}ms^{-1}}]@f$.
    double prm_k3;                   ///< @f$[\mathrm{ms^{-1}}]@f$.
    double prm_k4;                   ///< @f$[\mathrm{ms^{-1}}]@f$.
    double prm_EC;                   ///< @f$[\mathrm{mM}]@f$.
    double prm_maxsr;                ///< @f$[\mathrm{1}]@f$.
    double prm_minsr;                ///< @f$[\mathrm{1}]@f$.
    double prm_Vleak;                ///< @f$[\mathrm{mM/ms}]@f$.
    double prm_Vxfer;                ///< @f$[\mathrm{mM/ms}]@f$.
    double prm_R;                    ///< @f$[\mathrm{JK^{-1}mmol^{-1}}]@f$.
    double prm_F;                    ///< @f$[\mathrm{C/mol}]@f$.
    double prm_T;                    ///< @f$[\mathrm{K}]@f$.
    double RT_F;                     ///< @f$[\mathrm{mV}]@f$.
    double prm_Gkr;                  ///< @f$[\mathrm{nS/pF}]@f$.
    double prm_pKNa;                 ///< @f$[\mathrm{1}]@f$.
    double Gks;                      ///< @f$[\mathrm{]@f$.
    double prm_GK1;                  ///< @f$[\mathrm{nS/pF}]@f$.
    double Gto;                      ///< @f$[\mathrm{}]@f$.
    double prm_GNa;                  ///< @f$[\mathrm{nS/pF}]@f$.
    double prm_GbNa;                 ///< @f$[\mathrm{nS/pF}]@f$.
    double prm_KmK;                  ///< @f$[\mathrm{mM}]@f$.
    double prm_KmNa;                 ///< @f$[\mathrm{mM}]@f$.
    double prm_knak;                 ///< @f$[\mathrm{pA/pF}]@f$.
    double prm_GCaL;      ///< @f$[\mathrm{cmms^{-1}\micro F^{-1}}]@f$.
    double prm_GbCa;      ///< @f$[\mathrm{nS/pF}]@f$.
    double prm_knaca;     ///< @f$[\mathrm{pA/pF}]@f$.
    double prm_KmNai;     ///< @f$[\mathrm{mM}]@f$.
    double prm_KmCa;      ///< @f$[\mathrm{mM}]@f$.
    double prm_ksat;      ///< @f$[\mathrm{1}]@f$.
    double prm_n;         ///< @f$[\mathrm{1}]@f$.
    double prm_GpCa;      ///< @f$[\mathrm{nS/pF}]@f$.
    double prm_KpCa;      ///< @f$[\mathrm{nS/pF}]@f$.
    double prm_GpK;       ///< @f$[\mathrm{nS/pF}]@f$.
    double inverse_VcF2;  ///< @f$[\mathrm{C^{-1}/mol^{-1}m^{-3}}]@f$.
    double inverse_VcF;   ///< @f$[\mathrm{C^{-1}/mol^{-1}m^{-3}}]@f$.
    double inverse_vssF2; ///< @f$[\mathrm{C^{-1}/mol^{-1}m^{-3}}]@f$.

    /// @}

    std::variant<double, double_AD> VV; ///< Voltage variable.

    std::array<double, 12> tau;  ///< Time constants.
    std::array<double, 12> wInf; ///< Final values of the gating variables.

    std::variant<CurrentType<double>, CurrentType<double_AD>>
      currents; ///< Vector of ionic currents.

    std::array<double, 18> ww_ext; ///< Extrapolated ionic variables.
    std::array<double, 6>
      dConc; ///< Derivatives of the concentration equations.
  };

  LIFEX_REGISTER_CHILD_INTO_FACTORY(Ionic,
                                    TTP06,
                                    TTP06::label,
                                    const std::string &);

} // namespace lifex

#endif /* LIFEX_IONIC_TTP06_HPP_ */
