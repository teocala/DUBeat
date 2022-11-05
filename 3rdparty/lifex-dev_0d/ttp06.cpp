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

#include "ttp06.hpp"

#include "source/numerics/numbers.hpp"

namespace lifex0d
{
  TTP06::TTP06(const std::string &subsection)
    : Ionic(18, subsection + " / " + label)
  {}

  void
  TTP06::declare_parameters(ParamHandler &params) const
  {
    Ionic::declare_parameters(params);

    // Declare parameters.
    params.set_verbosity(VerbosityParam::Full);
    params.enter_subsection_path(prm_subsection_path);
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

  void
  TTP06::parse_parameters(ParamHandler &params)
  {
    // Parse input file.
    params.parse();

    Ionic::parse_parameters(params);

    // Read input parameters.
    params.enter_subsection_path(prm_subsection_path);
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

        cell_type_dof = prm_cell_type;

        if (prm_cell_type == "Epicardium")
          {
            Gks = 0.392;
            Gto = 0.294;
          }
        else if (prm_cell_type == "Endocardium")
          {
            Gks = 0.392;
            Gto = 0.073;
          }
        else if (prm_cell_type == "Myocardium")
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

  std::string
  TTP06::iterations_log_string(const unsigned int & /* n_iter */)
  {
    return std::string(label) + ": direct solver";
  }

  std::vector<double>
  TTP06::setup_initial_conditions() const
  {
    std::vector<double> w(n_variables);
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

  double
  TTP06::setup_initial_transmembrane_potential() const
  {
    return -84e-3;
  }

  double
  TTP06::compute_calcium_raw(const std::vector<double> &w) const
  {
    return w[12];
  }

  std::pair<std::vector<double>, unsigned int>
  TTP06::solve_time_step_0d(const double              &u,
                            const double              &alpha_bdf,
                            const std::vector<double> &w_bdf,
                            const std::vector<double> &w_ext,
                            const double              &cell_type,
                            const double              &Iapp)
  {
    for (unsigned int j = 0; j < n_variables; ++j)
      {
        ww_ext[j] = w_ext[j];
      }
    VV.emplace<double>(u);

    // See whether a distribution of action potentials from epicardium
    // to endocardium is employed or not.
    if (prm_cell_type == "All")
      {
        if (utils::is_zero(cell_type, prm_endo_tolerance))
          {
            Gks           = 0.392;
            Gto           = 0.073;
            cell_type_dof = "Endocardium";
          }
        else if (utils::is_equal(cell_type, 1.0, prm_epi_tolerance))
          {
            Gks           = 0.392;
            Gto           = 0.294;
            cell_type_dof = "Epicardium";
          }
        else
          {
            Gks           = 0.098;
            Gto           = 0.294;
            cell_type_dof = "Myocardium";
          }
      }

    currents = CurrentType<double>();
    compute_currents(std::get<double>(VV),
                     std::get<CurrentType<double>>(currents));

    compute_gating_constants(ww_ext[14]);

    compute_concentration_derivatives(Iapp);

    // Update concentration variables at a specific DOF.
    std::vector<double> w_val(n_variables);
    for (unsigned int j = 12; j < n_variables; ++j)
      {
        w_val[j] = (w_bdf[j] + time_step * dConc[j - 12]) / alpha_bdf;
      }

    // Update gating variables at a specific DOF.
    for (unsigned int j = 0; j < 12; ++j)
      {
        w_val[j] = (wInf[j] * time_step + w_bdf[j] * tau[j]) /
                   (time_step + alpha_bdf * tau[j]);
      }

    if (prm_implicit_Iion)
      {
        currents = CurrentType<double_AD>();
      }

    unsigned int n_iter = 1;

    return std::make_pair(w_val, n_iter);
  }

  template <class NumberType>
  void
  TTP06::compute_currents(const NumberType        &VV,
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
      prm_Gkr * std::sqrt(prm_Ko / 5.4) * ww_ext[3] * ww_ext[4] * (V_mV - Ek);

    currents[1] = Gks * ww_ext[5] * ww_ext[5] * (V_mV - Eks);

    currents[2] = prm_GK1 * rec_iK1 * std::sqrt(prm_Ko / 5.4) * (V_mV - Ek);

    currents[3] = Gto * ww_ext[7] * ww_ext[6] * (V_mV - Ek);

    currents[4] = prm_GNa * ww_ext[0] * ww_ext[0] * ww_ext[0] * ww_ext[1] *
                  ww_ext[2] * (V_mV - Ena);

    currents[5] = prm_GbNa * (V_mV - Ena);

    currents[6] =
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

  void
  TTP06::compute_concentration_derivatives(const double &Iapp)
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

  void
  TTP06::compute_gating_constants(const double &CaSS)
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
    if (cell_type_dof == "Epicardium")
      {
        S_INF = 1.0 / (1.0 + std::exp((V_mV + 20) / 5.0));
        TAU_S = 85.0 * std::exp(-(V_mV + 45.0) * (V_mV + 45.0) / 320.0) +
                5.0 / (1.0 + std::exp((V_mV - 20.0) / 5.0)) + 3.0;
      }
    else if (cell_type_dof == "Endocardium")
      {
        S_INF = 1.0 / (1.0 + std::exp((V_mV + 28) / 5.0));
        TAU_S = 1000.0 * std::exp(-(V_mV + 67) * (V_mV + 67) / 1000.0) + 8.0;
      }
    else if (cell_type_dof == "Myocardium")
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

  std::pair<double, double>
  TTP06::Iion(const double              &u,
              const double              &u_old,
              const std::vector<double> &w,
              const double & /*cell_type*/)
  {
    for (unsigned int j = 0; j < n_variables; ++j)
      {
        ww_ext[j] = w[j];
      }


    auto compute_Iion = [this](auto u) {
      using NumberType = std::decay_t<decltype(u)>;

      VV.emplace<NumberType>(u);

      compute_currents(std::get<NumberType>(VV),
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

  /// Explicit instantiation.
  template void
  TTP06::compute_currents<>(const double &VV, CurrentType<double> &currents);

  /// Explicit instantiation.
  template void
  TTP06::compute_currents<>(const double_AD        &VV,
                            CurrentType<double_AD> &currents);
} // namespace lifex0d
