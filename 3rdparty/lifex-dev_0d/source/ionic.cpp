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
 * @author Pasquale Claudio Africa <pasqualeclaudio.africa@polimi.it>.
 * @author Michele Bucelli <michele.bucelli@polimi.it>.
 * @author Francesco Regazzoni <francesco.regazzoni@polimi.it>.
 * @author Matteo Salvador <matteo1.salvador@polimi.it>.
 * @author Gloria Marziali <gloria.marziali@mail.polimi.it>.
 * @author Giulia Gualtieri <giulia.gualtieri@mail.polimi.it>.
 */

#include "core/source/numerics/numbers.hpp"

#include "source/ionic.hpp"

#include <algorithm>
#include <limits>
#include <set>

namespace lifex
{
  Ionic::Ionic(const size_t &     n_variables_,
               const std::string &subsection)
    : CoreModel(subsection)
    , QuadratureEvaluationFEMScalar()
    , n_variables(n_variables_)
    , prm_bdf_order(1)
    , csv_writer(mpi_rank == 0)
    , do_0d_simulation_before_initialization(false)
  {
    // When entering declare_parameters* methods, the current subsection is the
    // one specified in the derived class. So we go up one level w.r.to the
    // current ionic model section.
    const size_t found = prm_subsection_path.find_last_of("/");
    subsection_path    = prm_subsection_path.substr(0, found);
    subsection_label = "Ionic model /" + prm_subsection_path.substr(found + 1);
  }

  void
  Ionic::declare_parameters(ParamHandler &params) const
  {
    params.enter_subsection_path(subsection_path);
    {
      // Declare parameters.
      params.declare_entry_selection("Cell type",
                                     "Epicardium",
                                     "Epicardium|Endocardium|Myocardium|All");

      params.set_verbosity(VerbosityParam::Standard);
      params.declare_entry("Calcium rescale factor",
                           "1",
                           Patterns::Double(0),
                           "Calcium rescale factor [-].");

      params.declare_entry("Epicardium tolerance",
                           "0.01",
                           Patterns::Double(0),
                           "Epicardium tolerance.");

      params.declare_entry("Endocardium tolerance",
                           "0.01",
                           Patterns::Double(0),
                           "Endocardium tolerance.");
      params.reset_verbosity();

      params.enter_subsection("Applied current 0D");
      {
        params.declare_entry("Initial times",
                             "0.0",
                             Patterns::List(Patterns::Double(0)),
                             "Initial times [s].");

        params.declare_entry("Durations",
                             "0.001",
                             Patterns::List(Patterns::Double(0)),
                             "Durations [s].");

        params.declare_entry(
          "Amplitudes",
          "50",
          Patterns::List(Patterns::Double(0)),
          "Impulse amplitude [V/s]. Please notice that in case a "
          "phenomenological ionic model (see documentation of the specific "
          "ionic model) is used, this input might be "
          "interpreted in dimensionless form.");

        params.declare_entry("Period",
                             "1.0",
                             Patterns::Double(0),
                             "Impulse period [s]. To apply a non-periodic "
                             "impulse, set Period = 0.");
      }
      params.leave_subsection();

      params.enter_subsection("Output 0D");
      {
        params.declare_entry("Enable output",
                             "true",
                             Patterns::Bool(),
                             "Enable/disable output.");

        params.declare_entry("Filename",
                             "ionic_model_0d.csv",
                             Patterns::FileName(
                               Patterns::FileName::FileType::output),
                             "Output file.");

        params.declare_entry("Save every n timesteps",
                             "1",
                             Patterns::Integer(1),
                             "Save every n timesteps.");

        params.declare_entry("First time to export",
                             "0",
                             Patterns::Double(0),
                             "First time to consider in exported file [s].");

        params.declare_entry(
          "Shift time",
          "true",
          Patterns::Bool(),
          "If true, exported time is shifted so that it starts at time zero.");
      }
      params.leave_subsection();
    }
    params.leave_subsection_path();
  }

  void
  Ionic::parse_parameters(ParamHandler &params)
  {
    params.enter_subsection_path(subsection_path);
    {
      // Read input parameters.
      prm_cell_type = params.get("Cell type");

      prm_calcium_rescale = params.get_double("Calcium rescale factor");

      prm_epi_tolerance  = params.get_double("Epicardium tolerance");
      prm_endo_tolerance = params.get_double("Endocardium tolerance");

      params.enter_subsection("Applied current 0D");
      {
        prm_I_app_0d_initial_times = params.get_vector<double>("Initial times");
        prm_I_app_0d_durations =
          params.get_vector<double>("Durations",
                                    prm_I_app_0d_initial_times.size());
        prm_I_app_0d_amplitudes =
          params.get_vector<double>("Amplitudes",
                                    prm_I_app_0d_initial_times.size());
        prm_I_app_0d_period = params.get_double("Period");
      }
      params.leave_subsection();

      params.enter_subsection("Output 0D");
      {
        prm_enable_output = params.get_bool("Enable output");
        prm_output_every_n_timesteps =
          params.get_integer("Save every n timesteps");
        prm_first_time_to_export = params.get_double("First time to export");
        prm_shift_time           = params.get_bool("Shift time");
        prm_output_basename      = params.get("Filename");
      }
      params.leave_subsection();
    }
    params.leave_subsection_path();
  }

  void
  Ionic::declare_parameters_0d(ParamHandler &params) const
  {
    params.enter_subsection_path(subsection_path);
    {
      params.enter_subsection("Time solver");
      {
        params.declare_entry("BDF order",
                             "1",
                             Patterns::Integer(1),
                             "BDF order.");

        params.declare_entry("Initial time",
                             "0.0",
                             Patterns::Double(0),
                             "Initial time [s].");

        params.declare_entry("Final time",
                             "1.0",
                             Patterns::Double(0),
                             "Final time [s].");

        params.declare_entry("Time step",
                             "1e-4",
                             Patterns::Double(0),
                             "Time step [s].");
      }
      params.leave_subsection();
    }
    params.leave_subsection_path();
  }

  void
  Ionic::parse_parameters_0d(ParamHandler &params)
  {
    params.enter_subsection_path(subsection_path);
    {
      params.enter_subsection("Time solver");
      {
        prm_bdf_order    = params.get_integer("BDF order");
        prm_initial_time = params.get_double("Initial time");
        prm_final_time   = params.get_double("Final time");
        prm_time_step    = params.get_double("Time step");
      }
      params.leave_subsection();
    }
    params.leave_subsection_path();
  }

  void
  Ionic::solve_electrophysiology_time_step_0d(const bool &verbose)
  {
    double              dIion_du_val;
    double              Iion_val;
    const double &      alpha_bdf = bdf_handler_0d[0].get_alpha();
    std::vector<double> ww_bdf(n_variables);
    std::vector<double> ww_ext(n_variables);

    double time_relative = time;
    if (prm_I_app_0d_period > 0.0)
      {
        time_relative = std::fmod(time, prm_I_app_0d_period);
      }
    double Iapp = 0.0;
    for (unsigned int i = 0; i < prm_I_app_0d_initial_times.size(); ++i)
      {
        Iapp += prm_I_app_0d_amplitudes[i] *
                (prm_I_app_0d_initial_times[i] <= time_relative &&
                 time_relative <
                   prm_I_app_0d_initial_times[i] + prm_I_app_0d_durations[i]);
      }

    bdf_handler_0d[0].time_advance(u_0d, true);
    u_bdf_0d = *bdf_handler_0d[0].get_sol_bdf_ptr();
    u_ext_0d = *bdf_handler_0d[0].get_sol_extrapolation_ptr();
    for (size_t n = 0; n < n_variables; ++n)
      {
        bdf_handler_0d[n + 1].time_advance(w_0d[n], true);
        w_bdf_0d[n] = bdf_handler_0d[n + 1].get_sol_bdf_ptr();
        w_ext_0d[n] = bdf_handler_0d[n + 1].get_sol_extrapolation_ptr();
        ww_bdf[n]   = *w_bdf_0d[n];
        ww_ext[n]   = *w_ext_0d[n];
      }

    unsigned int n_iter = 0;
    std::tie(w_0d, n_iter) =
      solve_time_step_0d(u_ext_0d, alpha_bdf, ww_bdf, ww_ext, 0, Iapp);

    calcium_0d = compute_calcium(w_0d);

    if (verbose)
      pcout << "\t" << iterations_log_string(n_iter) << std::endl;

    std::tie(Iion_val, dIion_du_val) = Iion(u_0d, u_ext_0d, w_0d, 0.5);


    u_0d = (u_bdf_0d - prm_time_step / dt_fact * (Iion_val - Iapp)) / alpha_bdf;

    time += prm_time_step;
  }

  void
  Ionic::setup_system_0d()
  {
    TimerOutput::Scope timer_section(timer_output,
                                     subsection_label + " / Setup system 0D");

    w_bdf_0d.resize(n_variables);
    w_ext_0d.resize(n_variables);

    u_0d = setup_initial_transmembrane_potential();

    w_0d       = setup_initial_conditions();
    calcium_0d = compute_calcium(w_0d);

    // One BDF handler for each variable, plus one for the transmembrane
    // potential.
    bdf_handler_0d.resize(n_variables + 1);
    bdf_handler_0d[0].initialize(prm_bdf_order,
                                 std::vector<double>(prm_bdf_order, u_0d));
    for (size_t n = 0; n < n_variables; ++n)
      {
        bdf_handler_0d[n + 1].initialize(prm_bdf_order,
                                         std::vector<double>(prm_bdf_order,
                                                             w_0d[n]));
      }

    time      = prm_initial_time;
    time_step = prm_time_step;

    if (prm_enable_output)
      {
        // Declare entries for CSV output.
        std::vector<std::string> csv_entries;
        csv_entries.push_back("t");
        csv_entries.push_back("u");
        csv_entries.push_back("Ca");
        for (size_t n = 0; n < n_variables; ++n)
          {
            csv_entries.push_back("w" + std::to_string(n) + "");
          }

        csv_writer.declare_entries(csv_entries);
        csv_writer.set_format_double(std::ios::fixed, 15);
        csv_writer.open(get_output_csv_filename(), ',');
        output_results_0d();
      }
  }

  void
  Ionic::run_0d(const bool &verbose)
  {
    unsigned int timestep_number = 1;

    setup_system_0d();

    {
      TimerOutput::Scope timer_section(timer_output,
                                       subsection_label +
                                         " / Solve electrophysiology 0D");

      pcout << "Ionic model 0D simulation, BDF" << prm_bdf_order << std::endl;

      // Prevent performing one additional iteration due to finite arithmetic
      // errors.
      while (time < prm_final_time - prm_time_step * 1e-2)
        {
          if (verbose)
            pcout << "Time step " << std::setw(6) << timestep_number
                  << " at t = " << std::setw(8) << std::fixed
                  << std::setprecision(6) << time;

          solve_electrophysiology_time_step_0d(verbose);

          if (prm_enable_output &&
              timestep_number % prm_output_every_n_timesteps == 0)
            {
              output_results_0d();
            }

          ++timestep_number;
        }
    }
  }

  double
  Ionic::compute_calcium(const std::vector<double> &w) const
  {
    return prm_calcium_rescale * compute_calcium_raw(w);
  }

  void
  Ionic::output_results_0d()
  {
    if (time < prm_first_time_to_export - prm_time_step * 1e-2)
      return;

    TimerOutput::Scope timer_section(timer_output,
                                     subsection_label + " / Output results");

    double time_to_export    = time;
    double time_ini_expected = prm_first_time_to_export;
    double time_end_expected = prm_final_time;
    if (prm_shift_time)
      {
        time_to_export -= prm_first_time_to_export;
        time_ini_expected = 0;
        time_end_expected = prm_final_time - prm_first_time_to_export;
      }
    if (std::abs(time_to_export - time_ini_expected) < prm_time_step * 1e-2)
      time_to_export = time_ini_expected;
    if (std::abs(time_to_export - time_end_expected) < prm_time_step * 1e-2)
      time_to_export = time_end_expected;

    csv_writer.set_entries(
      {{"t", time_to_export}, {"u", u_0d}, {"Ca", calcium_0d}});
    for (size_t n = 0; n < n_variables; ++n)
      {
        csv_writer.set_entries({{"w" + std::to_string(n) + "", w_0d[n]}});
      }

    csv_writer.write_line();
  }

} // namespace lifex
