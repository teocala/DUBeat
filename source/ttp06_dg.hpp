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

#ifndef LIFEX_IONIC_TTP06_DG_HPP_
#define LIFEX_IONIC_TTP06_DG_HPP_

#include <string>
#include <utility>
#include <vector>

#include "ionic_dg.hpp"

namespace lifex
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

  template <class basis>
  class TTP06_DG : public Ionic_DG<basis>
  {
  public:
    /// Ionic model label.
    static inline constexpr auto label = "TTP06 DG";

    /// Alias for array of currents.
    template <class NumberType>
    using CurrentType = std::array<NumberType, 12>;

    /// Constructor.
    TTP06_DG(const std::string &subsection, const bool &standalone_)
      : Ionic_DG<basis>(18, subsection + " / " + label, standalone_)
    {
      this->compute_I_app = true;
    }

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
                       const double &             ischemic_region,
                       const double &             Iapp) override;

    /// Evaluate the ionic current given the transmembrane potential and the
    /// ionic model variables.
    virtual std::pair<double, double>
    Iion(const double &             u,
         const double &             u_old,
         const std::vector<double> &w,
         const double &             ischemic_region,
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
    compute_currents(const double &           ischemic_region,
                     const NumberType &       VV,
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

  template <class basis>
  void
  TTP06_DG<basis>::declare_parameters(ParamHandler &params) const
  {
    Ionic_DG<basis>::declare_parameters(params);

    // Declare parameters.
    params.set_verbosity(VerbosityParam::Full);
    params.enter_subsection_path(this->prm_subsection_path);
    {
      params.declare_entry("Implicit Iion",
                           "false",
                           Patterns::Bool(),
                           "Treat Iion implicitly w.r.to u and compute "
                           "dIion/du via automatic differentiation.");

      params.enter_subsection("Physical constants");
      {
        params.declare_entry("Membrane capacitance",
                             "1.0",
                             Patterns::Double(0),
                             "Membrane capacitance.");

        params.declare_entry("Capacitance",
                             "0.185",
                             Patterns::Double(0),
                             "Capacitance.");

        params.declare_entry("Ko",
                             "5.4",
                             Patterns::Double(0),
                             "Extracellular K+ concentration.");

        params.declare_entry("Cao",
                             "2.0",
                             Patterns::Double(0),
                             "Extracellular Ca2+ concentration.");

        params.declare_entry("Nao",
                             "140.0",
                             Patterns::Double(0),
                             "Extracellular Na+ concentration.");

        params.declare_entry("Vc",
                             "0.016404",
                             Patterns::Double(0),
                             "Cytoplasmic volume.");

        params.declare_entry("Vsr",
                             "0.001094",
                             Patterns::Double(0),
                             "Sarcoplasmic reticulum volume.");

        params.declare_entry("Vss",
                             "0.00005468",
                             Patterns::Double(0),
                             "Subspace volume.");

        params.declare_entry("Bufc",
                             "0.2",
                             Patterns::Double(0),
                             "Total cytoplasmic buffer concentration.");

        params.declare_entry(
          "Kbufc",
          "0.001",
          Patterns::Double(),
          "Cai half-saturation constant for cytoplasmic buffer.");

        params.declare_entry("Bufsr",
                             "10.0",
                             Patterns::Double(0),
                             "Total sarcoplasmic buffer concentration.");

        params.declare_entry(
          "Kbufsr",
          "0.3",
          Patterns::Double(0),
          "CaSR half-saturation constant for sarcoplasmic buffer.");

        params.declare_entry("Bufss",
                             "0.4",
                             Patterns::Double(0),
                             "Total subspace buffer concentration.");

        params.declare_entry(
          "Kbufss",
          "0.00025",
          Patterns::Double(0),
          "CaSS half-saturation constant for subspace buffer.");

        params.declare_entry("Vmaxup",
                             "0.006375",
                             Patterns::Double(0),
                             "Maximal Iup conductance.");

        params.declare_entry("Kup",
                             "0.00025",
                             Patterns::Double(0),
                             "Half-saturation constant Iup.");

        params.declare_entry("Vrel",
                             "0.102",
                             Patterns::Double(0),
                             "Maximal Irel conductance.");

        params.declare_entry("k1",
                             "0.15",
                             Patterns::Double(0),
                             "R to O and RI to I Irel transition rate.");

        params.declare_entry("k2",
                             "0.045",
                             Patterns::Double(0),
                             "O to I and R to RI Irel transition rate.");

        params.declare_entry("k3",
                             "0.060",
                             Patterns::Double(0),
                             "O to R and I to RI Irel transition rate.");

        params.declare_entry("k4",
                             "0.005",
                             Patterns::Double(0),
                             "I to O and RI to I Irel transition rate.");

        params.declare_entry("EC",
                             "1.5",
                             Patterns::Double(0),
                             "CaSR half-saturation constant of kCaSR.");

        params.declare_entry("maxsr",
                             "2.5",
                             Patterns::Double(0),
                             "Maximum value of kCaSR.");

        params.declare_entry("minsr",
                             "1.0",
                             Patterns::Double(0),
                             "Minimum value of kCaSR.");

        params.declare_entry("Vleak",
                             "0.00036",
                             Patterns::Double(0),
                             "Maximal Ileak conductance.");

        params.declare_entry("Vxfer",
                             "0.0038",
                             Patterns::Double(0),
                             "Maximal Ixfer conductance.");

        params.declare_entry("R",
                             "8314.472",
                             Patterns::Double(0),
                             "Gas constant.");

        params.declare_entry("F",
                             "96485.3415",
                             Patterns::Double(0),
                             "Faraday constant.");

        params.declare_entry("T", "310.0", Patterns::Double(0), "Temperature.");

        params.declare_entry("Gkr",
                             "0.153",
                             Patterns::Double(0),
                             "Maximal Ikr conductance.");

        params.declare_entry("pKNa",
                             "0.03",
                             Patterns::Double(0),
                             "Relative Iks permeability to Na+.");

        params.declare_entry("GK1",
                             "5.405",
                             Patterns::Double(0),
                             "Maximal IK1 conductance.");

        params.declare_entry("GNa",
                             "14.838",
                             Patterns::Double(0),
                             "Maximal INa conductance.");

        params.declare_entry("GbNa",
                             "0.00029",
                             Patterns::Double(0),
                             "Maximal IbNa conductance.");

        params.declare_entry("KmK",
                             "1.0",
                             Patterns::Double(0),
                             "K0 half-saturation constant of INaK.");

        params.declare_entry("KmNa",
                             "40.0",
                             Patterns::Double(0),
                             "Nai half-saturation constant of INaK.");

        params.declare_entry("knak",
                             "2.724",
                             Patterns::Double(0),
                             "Maximal INaK.");

        params.declare_entry("GCaL",
                             "0.00003980",
                             Patterns::Double(0),
                             "Maximal ICaL.");

        params.declare_entry("GbCa",
                             "0.000592",
                             Patterns::Double(0),
                             "Maximal IbCa conductance.");

        params.declare_entry("knaca",
                             "1000.0",
                             Patterns::Double(0),
                             "Maximal INaCa.");

        params.declare_entry("KmNai",
                             "87.5",
                             Patterns::Double(0),
                             "Nai half-saturation constant of INaK.");

        params.declare_entry("KmCa",
                             "1.38",
                             Patterns::Double(0),
                             "Cai half-saturation constant for INaCa.");

        params.declare_entry("ksat",
                             "0.1",
                             Patterns::Double(0),
                             "Saturation factor for INaCa.");

        params.declare_entry("n",
                             "0.35",
                             Patterns::Double(0),
                             "Voltage dependence parameter of INaCa.");

        params.declare_entry("GpCa",
                             "0.1238",
                             Patterns::Double(0),
                             "Maximal IpCa conductance.");

        params.declare_entry("KpCa",
                             "0.0005",
                             Patterns::Double(0),
                             "Half-saturation constant of IpCa.");

        params.declare_entry("GpK",
                             "0.0146",
                             Patterns::Double(0),
                             "Maximal IpK conductance.");
      }
      params.leave_subsection();
    }
    params.leave_subsection_path();
    params.reset_verbosity();
  }

  template <class basis>
  void
  TTP06_DG<basis>::parse_parameters(ParamHandler &params)
  {
    // Parse input file.
    params.parse();

    Ionic_DG<basis>::parse_parameters(params);

    // Read input parameters.
    params.enter_subsection_path(this->prm_subsection_path);
    {
      prm_implicit_Iion = params.get_bool("Implicit Iion");

      params.enter_subsection("Physical constants");
      {
        prm_membrane_capacitance = params.get_double("Membrane capacitance");

        prm_capacitance = params.get_double("Capacitance");
        prm_Ko          = params.get_double("Ko");
        prm_Cao         = params.get_double("Cao");
        prm_Nao         = params.get_double("Nao");
        prm_Vc          = params.get_double("Vc");
        prm_Vsr         = params.get_double("Vsr");
        prm_Vss         = params.get_double("Vss");
        prm_Bufc        = params.get_double("Bufc");
        prm_Kbufc       = params.get_double("Kbufc");
        prm_Bufsr       = params.get_double("Bufsr");
        prm_Kbufsr      = params.get_double("Kbufsr");
        prm_Bufss       = params.get_double("Bufss");
        prm_Kbufss      = params.get_double("Kbufss");
        prm_Vmaxup      = params.get_double("Vmaxup");
        prm_Kup         = params.get_double("Kup");
        prm_Vrel        = params.get_double("Vrel");
        prm_k1_         = params.get_double("k1");
        prm_k2_         = params.get_double("k2");
        prm_k3          = params.get_double("k3");
        prm_k4          = params.get_double("k4");
        prm_EC          = params.get_double("EC");
        prm_maxsr       = params.get_double("maxsr");
        prm_minsr       = params.get_double("minsr");
        prm_Vleak       = params.get_double("Vleak");
        prm_Vxfer       = params.get_double("Vxfer");
        prm_R           = params.get_double("R");
        prm_F           = params.get_double("F");
        prm_T           = params.get_double("T");
        RT_F            = (prm_R * prm_T) / prm_F;
        prm_Gkr         = params.get_double("Gkr");
        prm_pKNa        = params.get_double("pKNa");

        this->cell_type_dof = this->prm_cell_type;

        if (this->prm_cell_type == "Epicardium")
          {
            Gks = 0.392;
            Gto = 0.294;
          }
        else if (this->prm_cell_type == "Endocardium")
          {
            Gks = 0.392;
            Gto = 0.073;
          }
        else if (this->prm_cell_type == "Myocardium")
          {
            Gks = 0.098;
            Gto = 0.294;
          }
        // else if (prm_cell_type == "All")
        // cell_type_dof is updated in solve_timestep_0d().

        prm_GK1   = params.get_double("GK1");
        prm_GNa   = params.get_double("GNa");
        prm_GbNa  = params.get_double("GbNa");
        prm_KmK   = params.get_double("KmK");
        prm_KmNa  = params.get_double("KmNa");
        prm_knak  = params.get_double("knak");
        prm_GCaL  = params.get_double("GCaL");
        prm_GbCa  = params.get_double("GbCa");
        prm_knaca = params.get_double("knaca");
        prm_KmNai = params.get_double("KmNai");
        prm_KmCa  = params.get_double("KmCa");
        prm_ksat  = params.get_double("ksat");
        prm_n     = params.get_double("n");
        prm_GpCa  = params.get_double("GpCa");
        prm_KpCa  = params.get_double("KpCa");
        prm_GpK   = params.get_double("GpK");

        inverse_VcF2  = 1.0 / (2 * prm_Vc * prm_F);
        inverse_VcF   = 1.0 / (prm_Vc * prm_F);
        inverse_vssF2 = 1.0 / (2 * prm_Vss * prm_F);
      }
      params.leave_subsection();
    }
    params.leave_subsection_path();
  }

  template <class basis>
  std::string
  TTP06_DG<basis>::iterations_log_string(const unsigned int & /* n_iter */)
  {
    return std::string(label) + ": direct solver";
  }

  template <class basis>
  std::vector<double>
  TTP06_DG<basis>::setup_initial_conditions() const
  {
    std::vector<double> w(this->n_variables);
    w[0]  = 0.0;     // M
    w[1]  = 0.75;    // H
    w[2]  = 0.75;    // J
    w[3]  = 0.0;     // Xr1
    w[4]  = 1.0;     // Xr2
    w[5]  = 0.0;     // Xs
    w[6]  = 1.0;     // S
    w[7]  = 0.0;     // R
    w[8]  = 0.0;     // D
    w[9]  = 1.0;     // F
    w[10] = 1.0;     // F2
    w[11] = 1.0;     // FCass
    w[12] = 0.1;     // Cai
    w[13] = 3.35;    // CaSR
    w[14] = 0.00007; // CaSS
    w[15] = 7.67;    // Nai
    w[16] = 138.3;   // Ki
    w[17] = 1.0;     // RR

    return w;
  }

  template <class basis>
  double
  TTP06_DG<basis>::setup_initial_transmembrane_potential() const
  {
    return -84e-3;
  }

  template <class basis>
  double
  TTP06_DG<basis>::compute_calcium_raw(const std::vector<double> &w) const
  {
    return w[12];
  }

  template <class basis>
  std::pair<std::vector<double>, unsigned int>
  TTP06_DG<basis>::solve_time_step_0d(const double &             u,
                                      const double &             alpha_bdf,
                                      const std::vector<double> &w_bdf,
                                      const std::vector<double> &w_ext,
                                      const double &             cell_type,
                                      const double &ischemic_region,
                                      const double &Iapp)
  {
    for (unsigned int j = 0; j < this->n_variables; ++j)
      {
        ww_ext[j] = w_ext[j];
      }
    VV.emplace<double>(u);

    // See whether a distribution of action potentials from epicardium
    // to endocardium is employed or not.
    if (this->prm_cell_type == "All")
      {
        if (utils::is_zero(cell_type, this->prm_endo_tolerance))
          {
            Gks                 = 0.392;
            Gto                 = 0.073;
            this->cell_type_dof = "Endocardium";
          }
        else if (utils::is_equal(cell_type, 1.0, this->prm_epi_tolerance))
          {
            Gks                 = 0.392;
            Gto                 = 0.294;
            this->cell_type_dof = "Epicardium";
          }
        else
          {
            Gks                 = 0.098;
            Gto                 = 0.294;
            this->cell_type_dof = "Myocardium";
          }
      }

    currents = CurrentType<double>();
    compute_currents(ischemic_region,
                     std::get<double>(VV),
                     std::get<CurrentType<double>>(currents));

    compute_gating_constants(ww_ext[14]);

    compute_concentration_derivatives(Iapp);

    // Update concentration variables at a specific DOF.
    std::vector<double> w_val(this->n_variables);
    for (unsigned int j = 12; j < this->n_variables; ++j)
      {
        w_val[j] = (w_bdf[j] + this->time_step * dConc[j - 12]) / alpha_bdf;
      }

    // Update gating variables at a specific DOF.
    for (unsigned int j = 0; j < 12; ++j)
      {
        w_val[j] = (wInf[j] * this->time_step + w_bdf[j] * tau[j]) /
                   (this->time_step + alpha_bdf * tau[j]);
      }

    if (prm_implicit_Iion)
      {
        currents = CurrentType<double_AD>();
      }

    unsigned int n_iter = 1;

    return std::make_pair(w_val, n_iter);
  }

  template <class basis>
  template <class NumberType>
  void
  TTP06_DG<basis>::compute_currents(const double &           ischemic_region,
                                    const NumberType &       VV,
                                    CurrentType<NumberType> &currents)
  {
    // NB: the output is in mV/ms !

    double     Cai_mM = ww_ext[12] * 1e-3; // from microM to milliM
    NumberType V_mV   = VV * 1e3;          // from V to mV

    // Needed to compute currents
    double     Ek  = RT_F * (std::log((prm_Ko / ww_ext[16])));
    double     Ena = RT_F * (std::log((prm_Nao / ww_ext[15])));
    double     Eks = RT_F * (std::log((prm_Ko + prm_pKNa * prm_Nao) /
                                  (ww_ext[16] + prm_pKNa * ww_ext[15])));
    double     Eca = 0.5 * RT_F * (std::log((prm_Cao / Cai_mM)));
    NumberType Ak1 = 0.1 / (1.0 + std::exp(0.06 * (V_mV - Ek - 200)));
    NumberType Bk1 = (3.0 * std::exp(0.0002 * (V_mV - Ek + 100)) +
                      std::exp(0.1 * (V_mV - Ek - 10))) /
                     (1.0 + std::exp(-0.5 * (V_mV - Ek)));
    NumberType rec_iK1 = Ak1 / (Ak1 + Bk1);
    NumberType rec_iNaK =
      (1.0 / (1.0 + 0.1245 * std::exp(-0.1 * V_mV * prm_F / (prm_R * prm_T)) +
              0.0353 * std::exp(-V_mV * prm_F / (prm_R * prm_T))));
    NumberType rec_ipK = 1.0 / (1.0 + std::exp((25 - V_mV) / 5.98));

    // Compute currents
    currents[0] =
      (0.7 * (ischemic_region - this->prm_ischemic_region_threshold) /
         (1.0 - this->prm_ischemic_region_threshold) +
       0.3) *
      prm_Gkr * std::sqrt(prm_Ko / 5.4) * ww_ext[3] * ww_ext[4] * (V_mV - Ek);

    currents[1] =
      (0.8 * (ischemic_region - this->prm_ischemic_region_threshold) /
         (1.0 - this->prm_ischemic_region_threshold) +
       0.2) *
      Gks * ww_ext[5] * ww_ext[5] * (V_mV - Eks);

    currents[2] = prm_GK1 * rec_iK1 * std::sqrt(prm_Ko / 5.4) * (V_mV - Ek);

    currents[3] = Gto * ww_ext[7] * ww_ext[6] * (V_mV - Ek);

    currents[4] =
      (0.62 * (ischemic_region - this->prm_ischemic_region_threshold) /
         (1.0 - this->prm_ischemic_region_threshold) +
       0.38) *
      prm_GNa * ww_ext[0] * ww_ext[0] * ww_ext[0] * ww_ext[1] * ww_ext[2] *
      (V_mV - Ena);

    currents[5] = prm_GbNa * (V_mV - Ena);

    currents[6] =
      (0.69 * (ischemic_region - this->prm_ischemic_region_threshold) /
         (1.0 - this->prm_ischemic_region_threshold) +
       0.31) *
      prm_GCaL * ww_ext[8] * ww_ext[9] * ww_ext[10] * ww_ext[11] * 4 *
      (V_mV - 15) * (prm_F * prm_F / (prm_R * prm_T)) *
      (0.25 * std::exp(2 * (V_mV - 15) * prm_F / (prm_R * prm_T)) * ww_ext[14] -
       prm_Cao) /
      (std::exp(2 * (V_mV - 15) * prm_F / (prm_R * prm_T)) - 1.0);

    currents[7] = prm_GbCa * (V_mV - Eca);

    currents[8] = prm_knak * (prm_Ko / (prm_Ko + prm_KmK)) *
                  (ww_ext[15] / (ww_ext[15] + prm_KmNa)) * rec_iNaK;

    currents[9] =
      prm_knaca *
      (1.0 /
       (prm_KmNai * prm_KmNai * prm_KmNai + prm_Nao * prm_Nao * prm_Nao)) *
      (1.0 / (prm_KmCa + prm_Cao)) *
      (1.0 / (1 + prm_ksat *
                    std::exp((prm_n - 1) * V_mV * prm_F / (prm_R * prm_T)))) *
      (std::exp(prm_n * V_mV * prm_F / (prm_R * prm_T)) * ww_ext[15] *
         ww_ext[15] * ww_ext[15] * prm_Cao -
       std::exp((prm_n - 1) * V_mV * prm_F / (prm_R * prm_T)) * prm_Nao *
         prm_Nao * prm_Nao * Cai_mM * 2.5);

    currents[10] = prm_GpCa * Cai_mM / (prm_KpCa + Cai_mM);

    currents[11] = prm_GpK * rec_ipK * (V_mV - Ek);
  }

  template <class basis>
  void
  TTP06_DG<basis>::compute_concentration_derivatives(const double &Iapp)
  {
    auto &curr = std::get<CurrentType<double>>(currents);

    double Cai_mM     = ww_ext[12] * 1e-3; // from microM to milliM
    double Iapp_mV_ms = Iapp * prm_membrane_capacitance *
                        (-1.0); // from V/s to mV/mS (plus change of sign)

    double kCaSR =
      prm_maxsr - ((prm_maxsr - prm_minsr) /
                   (1 + (prm_EC / ww_ext[13]) * (prm_EC / ww_ext[13])));
    double k1  = prm_k1_ / kCaSR;
    double k2  = prm_k2_ * kCaSR;
    double dRR = prm_k4 * (1 - ww_ext[17]) - k2 * ww_ext[14] * ww_ext[17];

    double sOO = k1 * ww_ext[14] * ww_ext[14] * ww_ext[17] /
                 (prm_k3 + k1 * ww_ext[14] * ww_ext[14]);

    double Irel  = prm_Vrel * sOO * (ww_ext[13] - ww_ext[14]);
    double Ileak = prm_Vleak * (ww_ext[13] - Cai_mM);
    double Iup = prm_Vmaxup / (1.0 + ((prm_Kup * prm_Kup) / (Cai_mM * Cai_mM)));
    double Ixfer = prm_Vxfer * (ww_ext[14] - Cai_mM);

    double dVec_CaSR_tot = (Iup - Irel - Ileak);
    double dVec_CaSR =
      dVec_CaSR_tot / (1 + prm_Bufsr * prm_Kbufsr / (ww_ext[13] + prm_Kbufsr) /
                             (ww_ext[13] + prm_Kbufsr));

    double dVec_CaSS_tot =
      (-Ixfer * (prm_Vc / prm_Vss) + Irel * (prm_Vsr / prm_Vss) +
       (-curr[6] * inverse_vssF2 * prm_capacitance));
    double dVec_CaSS =
      dVec_CaSS_tot / (1 + prm_Bufss * prm_Kbufss / (ww_ext[14] + prm_Kbufss) /
                             (ww_ext[14] + prm_Kbufss));

    double dVec_Cai_tot =
      ((-(curr[7] + curr[10] - 2 * curr[9]) * inverse_VcF2 * prm_capacitance) -
       (Iup - Ileak) * (prm_Vsr / prm_Vc) + Ixfer);
    double dVec_Cai =
      dVec_Cai_tot /
      (1 + prm_Bufc * prm_Kbufc / (Cai_mM + prm_Kbufc) / (Cai_mM + prm_Kbufc));

    double dNai = -(curr[4] + curr[5] + 3 * curr[8] + 3 * curr[9]) *
                  inverse_VcF * prm_capacitance;

    double dKi = -(Iapp_mV_ms + curr[2] + curr[3] + curr[0] + curr[1] -
                   2 * curr[8] + curr[11]) *
                 inverse_VcF * prm_capacitance;

    dConc[0] = dVec_Cai * 1e6;  // from milliM ms^{-1} to microM s^{-1}
    dConc[1] = dVec_CaSR * 1e3; // from * ms^{-1} to * s^{-1}
    dConc[2] = dVec_CaSS * 1e3; // from * ms^{-1} to * s^{-1}
    dConc[3] = dNai * 1e3;      // from * ms^{-1} to * s^{-1}
    dConc[4] = dKi * 1e3;       // from * ms^{-1} to * s^{-1}
    dConc[5] = dRR * 1e3;       // from * ms^{-1} to * s^{-1}
  }

  template <class basis>
  void
  TTP06_DG<basis>::compute_gating_constants(const double &CaSS)
  {
    double V_mV = std::get<double>(VV) * 1e3; // from dimensionless to mV

    double AM = 1.0 / (1.0 + std::exp((-60.0 - V_mV) / 5.0));
    double BM = 0.1 / (1.0 + std::exp((V_mV + 35.0) / 5.0)) +
                0.10 / (1.0 + std::exp((V_mV - 50.0) / 200.0));
    double TAU_M = AM * BM;
    double INF   = 1.0 / ((1.0 + std::exp((-56.86 - V_mV) / 9.03)) *
                        (1.0 + std::exp((-56.86 - V_mV) / 9.03)));
    double TAU_H = 0;
    if (V_mV >= -40.0)
      {
        double AH_1 = 0.0;
        double BH_1 = (0.77 / (0.13 * (1. + std::exp(-(V_mV + 10.66) / 11.1))));
        TAU_H       = 1.0 / (AH_1 + BH_1);
      }
    else
      {
        double AH_2 = (0.057 * std::exp(-(V_mV + 80.0) / 6.8));
        double BH_2 =
          (2.7 * std::exp(0.079 * V_mV) + (3.1e5) * std::exp(0.3485 * V_mV));
        TAU_H = 1.0 / (AH_2 + BH_2);
      }
    double H_INF = 1.0 / ((1.0 + std::exp((V_mV + 71.55) / 7.43)) *
                          (1.0 + std::exp((V_mV + 71.55) / 7.43)));
    double TAU_J = 0.0;
    if (V_mV >= -40.0)
      {
        double AJ_1 = 0.0;
        double BJ_1 = (0.6 * std::exp((0.057) * V_mV) /
                       (1.0 + std::exp(-0.1 * (V_mV + 32.0))));
        TAU_J       = 1.0 / (AJ_1 + BJ_1);
      }
    else
      {
        double AJ_2 =
          (((-2.5428e4) * std::exp(0.2444 * V_mV) -
            (6.948e-6) * std::exp(-0.04391 * V_mV)) *
           (V_mV + 37.78) / (1. + std::exp(0.311 * (V_mV + 79.23))));
        double BJ_2 = (0.02424 * std::exp(-0.01052 * V_mV) /
                       (1.0 + std::exp(-0.1378 * (V_mV + 40.14))));
        TAU_J       = 1.0 / (AJ_2 + BJ_2);
      }
    double J_INF = H_INF;

    double Xr1_INF = 1.0 / (1.0 + std::exp((-26.0 - V_mV) / 7.0));
    double axr1    = 450.0 / (1.0 + std::exp((-45.0 - V_mV) / 10.0));
    double bxr1    = 6.0 / (1.0 + std::exp((V_mV - (-30.0)) / 11.5));
    double TAU_Xr1 = axr1 * bxr1;
    double Xr2_INF = 1.0 / (1.0 + std::exp((V_mV - (-88.0)) / 24.0));
    double axr2    = 3.0 / (1.0 + std::exp((-60.0 - V_mV) / 20.0));
    double bxr2    = 1.12 / (1.0 + std::exp((V_mV - 60.0) / 20.0));
    double TAU_Xr2 = axr2 * bxr2;

    double Xs_INF = 1.0 / (1.0 + std::exp((-5.0 - V_mV) / 14.0));
    double Axs    = (1400.0 / (std::sqrt(1. + std::exp((5.0 - V_mV) / 6))));
    double Bxs    = (1.0 / (1. + std::exp((V_mV - 35.0) / 15.0)));
    double TAU_Xs = Axs * Bxs + 80;

    double R_INF = 1.0 / (1.0 + std::exp((20 - V_mV) / 6.0));
    double S_INF = 0.0;
    double TAU_R =
      9.5 * std::exp(-(V_mV + 40.0) * (V_mV + 40.0) / 1800.0) + 0.8;
    double TAU_S = 0.0;
    if (this->cell_type_dof == "Epicardium")
      {
        S_INF = 1.0 / (1.0 + std::exp((V_mV + 20) / 5.0));
        TAU_S = 85.0 * std::exp(-(V_mV + 45.0) * (V_mV + 45.0) / 320.0) +
                5.0 / (1.0 + std::exp((V_mV - 20.0) / 5.0)) + 3.0;
      }
    else if (this->cell_type_dof == "Endocardium")
      {
        S_INF = 1.0 / (1.0 + std::exp((V_mV + 28) / 5.0));
        TAU_S = 1000.0 * std::exp(-(V_mV + 67) * (V_mV + 67) / 1000.0) + 8.0;
      }
    else if (this->cell_type_dof == "Myocardium")
      {
        S_INF = 1.0 / (1.0 + std::exp((V_mV + 20) / 5.0));
        TAU_S = 85.0 * std::exp(-(V_mV + 45.0) * (V_mV + 45.0) / 320.0) +
                5.0 / (1.0 + std::exp((V_mV - 20.0) / 5.0)) + 3.0;
      }

    double D_INF     = 1.0 / (1.0 + std::exp((-8 - V_mV) / 7.5));
    double Ad        = 1.4 / (1.0 + std::exp((-35 - V_mV) / 13)) + 0.25;
    double Bd        = 1.4 / (1.0 + std::exp((V_mV + 5) / 5));
    double Cd        = 1.0 / (1.0 + std::exp((50 - V_mV) / 20));
    double TAU_D     = Ad * Bd + Cd;
    double F_INF     = 1.0 / (1.0 + std::exp((V_mV + 20) / 7));
    double Af        = 1102.5 * std::exp(-(V_mV + 27) * (V_mV + 27) / 225);
    double Bf        = 200.0 / (1 + std::exp((13 - V_mV) / 10.));
    double Cf        = (180.0 / (1 + std::exp((V_mV + 30) / 10))) + 20;
    double TAU_F     = Af + Bf + Cf;
    double F2_INF    = 0.67 / (1. + std::exp((V_mV + 35) / 7)) + 0.33;
    double Af2       = 600 * std::exp(-(V_mV + 25) * (V_mV + 25) / 170);
    double Bf2       = 31 / (1.0 + std::exp((25 - V_mV) / 10));
    double Cf2       = 16 / (1.0 + std::exp((V_mV + 30) / 10));
    double TAU_F2    = Af2 + Bf2 + Cf2;
    double FCaSS_INF = 0.6 / (1 + (CaSS / 0.05) * (CaSS / 0.05)) + 0.4;
    double TAU_FCaSS = 80.0 / (1 + (CaSS / 0.05) * (CaSS / 0.05)) + 2.0;

    tau[0]  = TAU_M * 1e-3;     // From ms to s.
    tau[1]  = TAU_H * 1e-3;     // From ms to s.
    tau[2]  = TAU_J * 1e-3;     // From ms to s.
    tau[3]  = TAU_Xr1 * 1e-3;   // From ms to s.
    tau[4]  = TAU_Xr2 * 1e-3;   // From ms to s.
    tau[5]  = TAU_Xs * 1e-3;    // From ms to s.
    tau[6]  = TAU_S * 1e-3;     // From ms to s.
    tau[7]  = TAU_R * 1e-3;     // From ms to s.
    tau[8]  = TAU_D * 1e-3;     // From ms to s.
    tau[9]  = TAU_F * 1e-3;     // From ms to s.
    tau[10] = TAU_F2 * 1e-3;    // From ms to s.
    tau[11] = TAU_FCaSS * 1e-3; // From ms to s.

    wInf[0]  = INF;
    wInf[1]  = H_INF;
    wInf[2]  = J_INF;
    wInf[3]  = Xr1_INF;
    wInf[4]  = Xr2_INF;
    wInf[5]  = Xs_INF;
    wInf[6]  = S_INF;
    wInf[7]  = R_INF;
    wInf[8]  = D_INF;
    wInf[9]  = F_INF;
    wInf[10] = F2_INF;
    wInf[11] = FCaSS_INF;
  }

  template <class basis>
  std::pair<double, double>
  TTP06_DG<basis>::Iion(const double &             u,
                        const double &             u_old,
                        const std::vector<double> &w,
                        const double &             ischemic_region,
                        const double & /*cell_type*/)
  {
    for (unsigned int j = 0; j < this->n_variables; ++j)
      {
        ww_ext[j] = w[j];
      }


    auto compute_Iion = [this, ischemic_region](auto u) {
      using NumberType = std::decay_t<decltype(u)>;

      VV.emplace<NumberType>(u);

      compute_currents(ischemic_region,
                       std::get<NumberType>(VV),
                       std::get<CurrentType<NumberType>>(currents));

      NumberType Iion_val = 0.0;
      for (unsigned int j = 0; j < 12; ++j)
        {
          Iion_val += std::get<CurrentType<NumberType>>(currents)[j];
        }

      Iion_val /= prm_membrane_capacitance;

      return Iion_val;
    };


    if (prm_implicit_Iion)
      {
        return utils::compute_value_and_derivative(compute_Iion, u);
      }
    else
      {
        double Iion_val     = compute_Iion(u_old);
        double dIion_du_val = 0.0;

        return std::make_pair(Iion_val, dIion_du_val);
      }
  }

} // namespace lifex

#endif /* LIFEX_IONIC_TTP06_DG_HPP_ */
