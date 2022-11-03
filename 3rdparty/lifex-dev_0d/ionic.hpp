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
 * @author Roberto Piersanti <roberto.piersanti@polimi.it>.
 * @author Michele Bucelli <michele.bucelli@polimi.it>.
 * @author Francesco Regazzoni <francesco.regazzoni@polimi.it>.
 * @author Matteo Salvador <matteo1.salvador@polimi.it>.
 * @author Gloria Marziali <gloria.marziali@mail.polimi.it>.
 * @author Giulia Gualtieri <giulia.gualtieri@mail.polimi.it>.
 */

#ifndef LIFEX_PHYSICS_IONIC_HPP_0D
#define LIFEX_PHYSICS_IONIC_HPP_0D

#include "source/core_model.hpp"
#include "source/generic_factory.hpp"

#include "source/io/csv_writer.hpp"

#include "source/numerics/time_handler.hpp"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace lifex;

namespace lifex0d
{
  /**
   * @brief Abstract class implementing a ionic model.
   *
   * The equation solved is:
   * @f[
   * \frac{\partial \mathbf{w}}{\partial t} + \mathbf{G}(u, \mathbf{w}) = 0.
   * @f]
   *
   * For 0D simulations, the above equation is coupled with the following one
   * for the transmembrane potential:
   * @f[
   * C_m \frac{\partial u}{\partial t} + \mathcal{I}_{\mathrm{ion}}(u,
   * \mathbf{w}) = \mathcal{I}_{\mathrm{app}}(t).
   * @f]
   * rewritten, by defining @f$ I_* = \mathcal{I}_* / C_m@f$, as
   * @f[
   * \frac{\partial u}{\partial t} + I_{\mathrm{ion}}(u,
   * \mathbf{w}) = I_{\mathrm{app}}(t).
   * @f]
   * Both equations are discretized in time with a @f$\mathrm{BDF}\sigma@f$
   * scheme (where @f$\sigma@f$ is the order of the BDF formula) as follows:
   * @f[
   * \frac{\alpha_{\mathrm{BDF}\sigma}\mathbf{w}^{n+1} -
   * \mathbf{w}_{\mathrm{BDF}\sigma}^n}{\Delta t} +
   * \mathbf{G}(u_{\mathrm{EXT}\sigma}^{n+1},
   * \mathbf{w}_{\mathrm{EXT}\sigma}^{n+1}, \mathbf{w}^{n+1}) = 0, \\
   * \frac{\alpha_{\mathrm{BDF}\sigma}u^{n+1} -
   * u_{\mathrm{BDF}\sigma}^n}{\Delta t} +
   * I_{\mathrm{ion}}(u_{\mathrm{EXT}\sigma}^{n+1}, u^{n+1},
   * \mathbf{w}^{n+1}) = I_{\mathrm{app}}(t),
   * @f]
   * where @f$\Delta t = t^{n+1} - t^n@f$ is the time step,
   * @f$\mathbf{w}^{n+1} = \mathbf{w}(t^{n+1})@f$ and the subscript
   * @f$\mathrm{EXT}\sigma@f$ denotes the extrapolation to timestep @f$n +
   * 1@f$ from previous timesteps given by the @f$\mathrm{BDF}\sigma@f$
   * formula (see @ref utils::BDFHandler). The ionic current @f$ I_\mathrm{ion} @f$ is
   * assumed to be treated semi-implicitly with respect to @f$u^{n+1}@f$. The
   * linearization of the equation for the ionic variables @f$\mathbf{w}@f$
   * is delegated to the concrete ionic model implementation.
   *
   * In order to implement a concrete ionic model, you should define a class
   * inheriting from Ionic and implementing the following virtual methods:
   * - @ref setup_initial_conditions;
   * - @ref compute_calcium_raw;
   * - @ref setup_initial_transmembrane_potential;
   * - @ref solve_time_step_0d;
   * - @ref Iion(const double &, const double &,
   * const std::vector<double>&, double &, double &) "Ionic::Iion";
   * - @ref iterations_log_string.
   *
   * A public <kbd>static inline constexpr auto label = "Model-Name"</kbd>
   * member should also be available.
   *
   * The newly implemented ionic model can be registered into a generic
   * IonicFactory using the <kbd>LIFEX_REGISTER_CHILD_INTO_FACTORY</kbd> macro.
   *
   * Moreover:
   * - if the model needs to rescale the time variable, the member
   * Ionic::dt_fact should be specified in the constructor of the derived
   * class;
   * - if the model needs the applied current to be computed for the
   * evaluation of ionic variables and/or currents, the member
   * Ionic::compute_I_app should be set to true in the constructor of the
   * derived class.
   */
  class Ionic : public CoreModel, public QuadratureEvaluationFEMScalar
  {
  public:
    /// Alias for the generic ionic model factory.
    using IonicFactory =
      GenericFactory<Ionic, const std::string &>;

    /**
     * Constructor.
     * @param[in] n_variables_ Total number of ionic and gating variables.
     * @param[in] subsection   Parameter subsection.
     */
    Ionic(const size_t &     n_variables_,
          const std::string &subsection);

    /// Virtual destructor.
    virtual ~Ionic() = default;

    /// Override of @ref CoreModel::declare_parameters.
    virtual void
    declare_parameters(ParamHandler &params) const override;

    /// Override of @ref CoreModel::parse_parameters.
    virtual void
    parse_parameters(ParamHandler &params) override;

    /// Declare parameters for the 0D electrophysiology model.
    void
    declare_parameters_0d(ParamHandler &params) const;

    /// Parse parameters for the 0D electrophysiology model.
    void
    parse_parameters_0d(ParamHandler &params);

    /// Set the initial conditions for the ionic variables.
    virtual std::vector<double>
    setup_initial_conditions() const = 0;

    /// Compute the calcium concentration (after rescaling) given the ionic
    /// variables.
    virtual double
    compute_calcium(const std::vector<double> &w) const final;

    /// Set the initial conditions for the transmembrane potential,
    /// to be used in the associated electrophysiology problem.
    virtual double
    setup_initial_transmembrane_potential() const
    {
      return 0;
    }

    /// Setup for 0D simulations. Allocates vectors for ionic variables, sets up
    /// initial conditions and initializes BDF schemes.
    void
    setup_system_0d();

    /**
     * Compute the evolution of ionic variables at a single point, given the
     * transmembrane potential, solving one timestep of the ionic model equation
     * with a BDF scheme.
     * @param[in]  u                Transmembrane potential.
     * @param[in]  alpha_bdf        @f$\alpha@f$ coefficient of the BDF scheme (from BDFHandler).
     * @param[in]  w_bdf            BDF combination of the ionic variables at previous time steps (from BDFHandler).
     * @param[in]  w_ext            BDF extrapolation of the ionic variables (from BDFHandler).
     * @param[in]  cell_type        Value between 0.0 and 1.0 that indicates the type of cardiac cell (endocardium, myocardium or epicardium).
     * @param[in]  Iapp             Applied current.
     *
     * @return The pair [state variables at current time step, number of iterations needed to solve].
     */
    virtual std::pair<std::vector<double>, unsigned int>
    solve_time_step_0d(const double &             u,
                       const double &             alpha_bdf,
                       const std::vector<double> &w_bdf,
                       const std::vector<double> &w_ext,
                       const double &             cell_type,
                       const double &             Iapp) = 0;

    /**
     * Solves a time step of the ionic model and of the electrophysiology model
     * for a 0D simulation.
     */
    void
    solve_electrophysiology_time_step_0d(const bool &verbose = true);

    /**
     * @brief Evaluates the ionic  current @f$I_\mathrm{ion}(u, \mathbf{w})@f$ at a
     * single point, given the transmembrane potential and the
     * value of ionic model variables.
     * @param[in] u               Transmembrane potential vector.
     * @param[in] u_ext           BDF-extrapolated transmembrane potential vector.
     * @param[in] w               Values of the model variables.
     * @param[in] ischemic_region Ischemic region coefficient.
     * @param[in] cell_type       Value between 0.0 and 1.0 that indicates the type of cardiac cell (endocardium, myocardium or epicardium).
     *
     * @return The pair @f$I_\mathrm{ion}(u, \mathbf{w})@f$ and its derivative
     * @f$\frac{\partial I_\mathrm{ion}}{\partial u}(u, \mathbf{w})@f$.
     */
    virtual std::pair<double, double>
    Iion(const double &             u,
         const double &             u_ext,
         const std::vector<double> &w,
         const double &             cell_type) = 0;

    /// Run a 0D stimulation, computing the evolution of both transmembrane
    /// potential and ionic variables at a single point.
    void
    run_0d(const bool &verbose = true);

    /// Output results of 0D stimulation.
    void
    output_results_0d();

    /// Time step rescaling factor, depending on the ionic model.
    double dt_fact = 1;

    /// Getter for calcium initial condition.
    double
    get_calcium_initial_condition() const
    {
      return Ca_initial_condition;
    }

    /// Get 0D CSV output filename.
    std::string
    get_output_csv_filename() const
    {
      return prm_output_basename;
    }

    /// Getter for the upper threshold to compute repolarization maps
    /// (see @ref Electrophysiology::compute_repolarization_time).
    /// @note The returned value must be provided in dimensionless form
    /// for phenomenological ionic models.
    virtual double
    get_repolarization_upper_threshold() const = 0;

    /// Getter for the lower threshold to compute repolarization maps
    /// (see @ref Electrophysiology::compute_repolarization_time).
    /// @note The returned value must be provided in dimensionless form
    /// for phenomenological ionic models.
    virtual double
    get_repolarization_lower_threshold() const = 0;

  protected:
    /// Compute the raw calcium concentration (i.e. pre-rescaling) given the
    /// ionic variables.
    virtual double
    compute_calcium_raw(const std::vector<double> &w) const = 0;

    /// Subsection path.
    std::string subsection_path;

    /// Subsection label of the actual ionic model.
    std::string subsection_label;

    size_t n_variables; ///< Total number of ionic and gating variables.

    /// @name Parameters read from file.
    /// @{

    /// Initial time @f$\left[\mathrm{s}\right]@f$.
    double prm_initial_time;
    /// Final time @f$\left[\mathrm{s}\right]@f$.
    double prm_final_time;
    /// BDF order.
    unsigned int prm_bdf_order;
    /// Initial time of applied currents for 0D simulations
    std::vector<double> prm_I_app_0d_initial_times;
    /// Duration of applied currents for 0D simulations
    std::vector<double> prm_I_app_0d_durations;
    /// Amplitude of applied currents for 0D simulations
    std::vector<double> prm_I_app_0d_amplitudes;
    /// Period of applied currents for 0D simulations
    double prm_I_app_0d_period;
    /// Toggle save results.
    bool prm_enable_output;
    /// Save every n timesteps.
    unsigned int prm_output_every_n_timesteps;
    /// First time considered in exporting 0D.
    double prm_first_time_to_export;
    /// Toggle time shift in exporting 0D.
    bool prm_shift_time;
    /// Output basename.
    std::string prm_output_basename;
    /// Cell type (Endocardium, Myocardium, Epicardium or All).
    std::string prm_cell_type;
    /// Cell type (Endocardium, Myocardium or Epicardium) associated with the
    /// current dof.
    std::string cell_type_dof;
    /// Tolerance for the epicardium.
    double prm_epi_tolerance;
    /// Tolerance for the endocardium.
    double prm_endo_tolerance;
    /// Rescaling factor for the calcium concentration variable.
    double prm_calcium_rescale;

    /// @}

    /// Return a properly formatted string for the output of solver iterations.
    virtual std::string
    iterations_log_string(const unsigned int &n_iter) = 0;

    utils::CSVWriter csv_writer;
    
    double time_step; ///< Current time step.
    double time;      ///< Current time.

    double u_0d;     ///< Potential for 0D simulations
    double u_bdf_0d; ///< BDF combination of transmembrane potential at previous
                     ///< timesteps, for 0D simulation
    double u_ext_0d; ///< BDF extrapolation of transmembrane potential for 0D
                     ///< simulations
    std::vector<double> w_0d; ///< State variables for 0D simulations
    std::vector<std::shared_ptr<const double>>
      w_bdf_0d; ///< BDF combination of state variables at
                ///< previous timesteps, for 0D simulations
    std::vector<std::shared_ptr<const double>>
           w_ext_0d; ///< BDF extrapolation of state variables, for 0D simulations
    double calcium_0d; ///< Calcium concentration for 0D simulations


    std::vector<double>
      w_initial_conditions; ///< Initial condition of ionic variables.
    double
      Ca_initial_condition; ///< Initial condition of calcium concentration.

    /// BDF handler for 0D simulations. bdf_handler_0d[0] handles the time
    /// evolution of the transmembrane potential, while the following handle the
    /// evolution of the ionic variables.
    std::vector<utils::BDFHandler<double>> bdf_handler_0d;

    /// Flag to be used to perform a 0D simulation to initialize the ionic
    /// variables.
    bool do_0d_simulation_before_initialization;

  private:
    /// Time step @f$\left[\mathrm{s}\right]@f$, used only for 0D simulations.
    double prm_time_step;
  };

} // namespace lifex0d

#endif /* LIFEX_PHYSICS_IONIC_HPP_0D */
