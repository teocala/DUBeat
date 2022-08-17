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

#ifndef LIFEX_PHYSICS_IONIC_DG_HPP_
#define LIFEX_PHYSICS_IONIC_DG_HPP_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../../DUBeat/source/DG_Volume_handler.hpp"
#include "../../DUBeat/source/DoFHandler_DG.hpp"
#include "core/source/core_model.hpp"
#include "core/source/generic_factory.hpp"
#include "core/source/geometry/mesh_handler.hpp"
#include "core/source/io/csv_writer.hpp"
#include "core/source/numerics/time_handler.hpp"
#include "source/helpers/applied_current.hpp"
#include "source/numerics/numbers.hpp"

namespace lifex
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
   * For 3D simulations, it is assumed that the ionic model uses the same
   * finite element space, quadrature formula and BDF order as the associated
   * @ref Electrophysiology model.
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

  template <class basis>
  class Ionic_DG : public CoreModel, public QuadratureEvaluationFEMScalar
  {
  public:
    /**
     * Constructor.
     * @param[in] n_variables_ Total number of ionic and gating variables.
     * @param[in] subsection   Parameter subsection.
     * @param[in] standalone_  If true, this class is run as standalone,
     *                         @a i.e. in 0D. Otherwise, it is supposed to be
     *                         coupled to a 3D electrophysiology model and
     *                         initialized through @ref initialize_3d.
     */
    Ionic_DG(const size_t      &n_variables_,
             const std::string &subsection,
             const bool        &standalone_)
      : CoreModel(subsection)
      , QuadratureEvaluationFEMScalar()
      , standalone(standalone_)
      , n_variables(n_variables_)
      , prm_bdf_order(1)
      , csv_writer(mpi_rank == 0)
      , dof_handler(std::make_unique<DoFHandler_DG<basis>>())
      , I_app_function(nullptr)
      , do_0d_simulation_before_initialization(false)
    {
      // When entering declare_parameters* methods, the current subsection is
      // the one specified in the derived class. So we go up one level w.r.to
      // the current ionic model section.
      const size_t found = prm_subsection_path.find_last_of("/");
      subsection_path    = prm_subsection_path.substr(0, found);
      subsection_label =
        "Ionic model /" + prm_subsection_path.substr(found + 1);
    }

    /// Virtual destructor.
    virtual ~Ionic_DG() = default;

    /**
     * Initializer for 3D simulations.
     *
     * @param[in] triangulation_      Triangulation of the associated @ref Electrophysiology model.
     * @param[in] fe_degree_          Degree of the finite element space of the associated @ref Electrophysiology model.
     * @param[in] quadrature_formula_ Quadrature formula of the associated @ref Electrophysiology model.
     * @param[in] I_app_function_     Pointer to the applied function object of the associated @ref Electrophysiology model.
     * @param[in] bdf_order_          BDF formula order of the associated @ref Electrophysiology model.
     * @param[in] volume_label_       String identifier of a specific volume.
     * @param[in] map_id_volume_      Associate to each material ID a volume name.
     */
    virtual void
    initialize_3d(
      const std::shared_ptr<const utils::MeshHandler> &triangulation_,
      const unsigned int                              &fe_degree_,
      const std::shared_ptr<const Quadrature<dim>>    &quadrature_formula_,
      const std::shared_ptr<const AppliedCurrent>     &I_app_function_,
      const unsigned int                              &bdf_order_,
      const std::map<types::material_id, std::string> &map_id_volume_,
      const std::string                               &volume_label_) final;

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

    /// Setup system, @a i.e. allocate matrices and vectors.
    virtual void
    setup_system(const bool &initialize_ICI);

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
     * @param[in]  ischemic_region  Ischemic region coefficient.
     * @param[in]  Iapp             Applied current.
     *
     * @return The pair [state variables at current time step, number of iterations needed to solve].
     */
    virtual std::pair<std::vector<double>, unsigned int>
    solve_time_step_0d(const double              &u,
                       const double              &alpha_bdf,
                       const std::vector<double> &w_bdf,
                       const std::vector<double> &w_ext,
                       const double              &cell_type,
                       const double              &ischemic_region,
                       const double              &Iapp) = 0;

    /**
     * Compute the evolution of ionic variables at all degrees of freedom, given
     * the transmembrane potential from the electrophysiology model.
     *
     * @tparam VectorType <kbd>LinAlg::MPI::Vector</kbd> or
     * <kbd>LinearAlgebra::distributed::Vector<double></kbd> (for use with
     * matrix-free implementations, see @ref ElectrophysiologyMF).
     */
    template <class VectorType>
    void
    solve_time_step(const VectorType &u,
                    const double     &time_step_,
                    const double     &time_);

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
    Iion(const double              &u,
         const double              &u_ext,
         const std::vector<double> &w,
         const double              &ischemic_region,
         const double              &cell_type) = 0;

    /**
     * @brief Assemble the ionic current @f$I_\mathrm{ion}(u, \mathbf{w})@f$
     * vector and its derivative at the degrees of freedom associated
     * with @ref dof_handler, using Ionic Current Interpolation (ICI).
     *
     * The result of such assembly is stored in members @ref Iion_ICI_vec
     * and @ref dIion_du_ICI_vec, for use in @ref Iion_ICI.
     */
    void
    assemble_Iion_ICI(const LinAlg::MPI::Vector &u,
                      const LinAlg::MPI::Vector &u_ext);

    /**
     * @brief Same as the function above, template version.
     *
     * @tparam VectorType <kbd>LinAlg::MPI::Vector</kbd> or
     * <kbd>LinearAlgebra::distributed::Vector<double></kbd> (for use with
     * matrix-free implementations, see @ref ElectrophysiologyMF).
     */
    template <class VectorType>
    void
    assemble_Iion_ICI_tpl(const VectorType &u,
                          const VectorType &u_ext,
                          VectorType       &Iion_ICI_vec_owned,
                          VectorType       &dIion_du_ICI_vec_owned);

    /**
     * @brief Evaluates the ionic current @f$I_\mathrm{ion}(u, \mathbf{w})@f$
     * at quadrature nodes of a given cell, using Ionic Current Interpolation
     * (ICI) in a faster way.
     *
     * @return The pair @f$I_\mathrm{ion}(u, \mathbf{w})@f$ and its derivative
     * @f$\frac{\partial I_\mathrm{ion}}{\partial u}(u, \mathbf{w})@f$ evaluated
     * at quadrature nodes.
     */
    std::pair<std::vector<double>, std::vector<double>>
    Iion_ICI();

    /**
     * @brief Evaluates the ionic current @f$I_\mathrm{ion}(u, \mathbf{w})@f$
     * at quadrature nodes of a given cell, using a hybrid between
     * Ionic Current Interpolation (ICI) and State Variable Interpolation (SVI).
     *
     * In this case @f$u@f$ and @f$w@f$ are evaluated at dofs (like in
     * ICI), whereas @f$u_\mathrm{EXT}@f$, the ischemic region and
     * @ref endo_epi are evaluated at quadrature nodes (like in SVI).
     *
     * This method has shown to be effective for 3D simulations on coarse
     * meshes.
     *
     * @param[in] u     Transmembrane potential vector.
     * @param[in] u_ext BDF-extrapolated transmembrane potential vector.
     *
     * @return The pair @f$I_\mathrm{ion}(u, \mathbf{w})@f$ and its derivative
     * @f$\frac{\partial I_\mathrm{ion}}{\partial u}(u, \mathbf{w})@f$ evaluated
     * at quadrature nodes.
     */
    std::pair<std::vector<double>, std::vector<double>>
    Iion_ICI_hybrid(const LinAlg::MPI::Vector &u,
                    const LinAlg::MPI::Vector &u_ext);
    /**
     * @brief Evaluates the ionic current @f$I_\mathrm{ion}(u, \mathbf{w})@f$
     * at quadrature nodes of a given cell, using State Variable Interpolation
     * (SVI).
     * @param[in] u_loc     Transmembrane potential evaluated at quadrature nodes.
     * @param[in] u_ext_loc BDF-extrapolated transmembrane potential evaluated at quadrature nodes.
     *
     * @return The pair @f$I_\mathrm{ion}(u, \mathbf{w})@f$ and its derivative
     * @f$\frac{\partial I_\mathrm{ion}}{\partial u}(u, \mathbf{w})@f$ evaluated
     * at quadrature nodes.
     */
    std::pair<std::vector<double>, std::vector<double>>
    Iion_SVI(const std::vector<double> &u_loc,
             const std::vector<double> &u_ext_loc);

    /**
     * @brief Evaluates the ionic current @f$I_\mathrm{ion}(u, \mathbf{w})@f$
     * solving the 0d model at quadrature nodes.
     * @param[in] u_loc     Transmembrane potential evaluated at quadrature nodes.
     * @param[in] u_ext_loc BDF-extrapolated transmembrane potential evaluated at quadrature nodes.
     *
     * @return The pair @f$I_\mathrm{ion}(u, \mathbf{w})@f$ and its derivative
     * @f$\frac{\partial I_\mathrm{ion}}{\partial u}(u, \mathbf{w})@f$ evaluated
     * at quadrature nodes.
     */
    std::pair<std::vector<double>, std::vector<double>>
    Iion_quadrature(const std::vector<double> &u_loc,
                    const std::vector<double> &u_ext_loc);

    /// Run a 0D stimulation, computing the evolution of both transmembrane
    /// potential and ionic variables at a single point.
    void
    run_0d(const bool &verbose = true);

    /// Attach ionic variables to the output handler.
    /// The output mode can be one of the following:
    /// - "Calcium": to export only calcium variable.
    /// - "All": to export all the ionic and gating variables.
    // virtual void
    // attach_output(DataOut<dim> &data_out, const std::string &output_mode)
    // const;

    /// Output results of 0D stimulation.
    void
    output_results_0d();

    /// Set ischemic region.
    void
    set_ischemic_region(const LinAlg::MPI::Vector *ischemic_region_,
                        const double              &prm_scar_tolerance_)
    {
      ischemic_region    = ischemic_region_;
      prm_scar_tolerance = prm_scar_tolerance_;
    }

    /// Set transmurally varying solution between 0.0 (endocardium) and 1.0
    /// (epicardium).
    void
    set_endo_epi(const LinAlg::MPI::Vector *endo_epi_)
    {
      endo_epi = endo_epi_;
    }

    /// Time step rescaling factor, depending on the ionic model.
    double dt_fact = 1;

    /// Getter for finite element space.
    const FiniteElement<dim> &
    get_fe() const
    {
      return *fe;
    }

    /// Getter for DoF handler.
    const DoFHandler<dim> &
    get_dof_handler() const
    {
      return *dof_handler;
    }

    /// Getter for calcium variable.
    const LinAlg::MPI::Vector &
    get_calcium() const
    {
      return calcium;
    }

    /// Getter for calcium variable, without ghost entries.
    const LinAlg::MPI::Vector &
    get_calcium_owned() const
    {
      return calcium_owned;
    }

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

    /// Declare entries for the CSV log.
    void
    declare_entries_csv(utils::CSVWriter  &csv_writer,
                        const std::string &output_mode) const;

    // /// Set entries for the CSV log.
    // void
    // set_entries_csv(utils::CSVWriter & csv_writer,
    //                 const std::string &output_mode) const;

    /// Reinit
    virtual void
    reinit_new(const DoFHandler<dim>::active_cell_iterator &cell_other);

  protected:
    /// Compute the raw calcium concentration (i.e. pre-rescaling) given the
    /// ionic variables.
    virtual double
    compute_calcium_raw(const std::vector<double> &w) const = 0;

    /// If true, this class is run as standalone, @a i.e. in 0D.
    /// Otherwise, it is supposed to be coupled to a 3D electrophysiology model
    /// and initialized through @ref initialize_3d.
    bool standalone;

    /// Subsection path.
    std::string subsection_path;

    /// Subsection label of the actual ionic model.
    std::string subsection_label;

    /// String identifier of a specific volume.
    std::string volume_label;

    /// String identifier of a specific volume (without spaces, for output).
    std::string volume_label_output;

    /// Associate to each material ID a volume name.
    std::map<types::material_id, std::string> map_id_volume;

    /// Set of dof indices belonging to current volume.
    IndexSet volume_dof_indices;

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
    /// Tolerance for the scar.
    double prm_scar_tolerance;
    /// Rescaling factor for the calcium concentration variable.
    double prm_calcium_rescale;
    /// Threshold for the ischemic region vector.
    double prm_ischemic_region_threshold;

    /// @}

    /// CSV output for 0D simulations
    utils::CSVWriter csv_writer;

    /// Return a properly formatted string for the output of solver iterations.
    virtual std::string
    iterations_log_string(const unsigned int &n_iter) = 0;

    /// Triangulation.
    std::shared_ptr<const utils::MeshHandler> triangulation;

    /// FE space.
    std::unique_ptr<basis> fe;

    /// Volume handler DG
    std::unique_ptr<DGVolumeHandler<dim>> vol_handler;

    /// DoF handler.
    std::unique_ptr<DoFHandler_DG<basis>> dof_handler;

    /// Vector of BDF solutions, for each ionic variable.
    std::vector<LinAlg::MPI::Vector> w_bdf;
    /// Vector of BDF extrapolated solutions, for each ionic
    /// variable.
    std::vector<LinAlg::MPI::Vector> w_ext;

    /// BDF time advancing handlers, for each ionic variable.
    std::vector<utils::BDFHandler<LinAlg::MPI::Vector>> bdf_handler;
    /// Backup BDF handler for VARP stimulation protocol, for each ionic
    /// variable.
    std::vector<utils::BDFHandler<LinAlg::MPI::Vector>> bdf_handler_backup;

    std::vector<LinAlg::MPI::Vector> w; ///< Solution vectors.
    std::vector<LinAlg::MPI::Vector>
      w_owned; ///< Solution, without ghost entries.

    std::vector<LinAlg::MPI::Vector>
      w_owned_backup; ///< Solution vectors backup for VARP stimulation
                      ///< protocol, without ghost entries.

    /// Flag to enable or disable the computation of applied current for 3D
    /// electrophysiology. If set to false, the applied current passed to
    /// solve_time_step_0d in 3D simulations is always 0. Should be false,
    /// unless the model needs to know the applied current. Defaults to false.
    bool compute_I_app = false;

    /// Applied current for 3D electrophysiology.
    std::shared_ptr<const AppliedCurrent> I_app_function;

    LinAlg::MPI::Vector I_app; ///< Iapp region vector.

    /// Iapp region vector, without ghost entries.
    LinAlg::MPI::Vector I_app_owned;

    std::vector<LinAlg::MPI::Vector> dw;        ///< Increment vectors.
    const LinAlg::MPI::Vector *ischemic_region; ///< Ischemic region vector.

    /// Transmurally varying vector between 0.0
    /// (endocardium) and 1.0 (epicardium).
    const LinAlg::MPI::Vector *endo_epi;

    /// Variable representing calcium concentration, without ghost entries.
    LinAlg::MPI::Vector calcium_owned;

    /// Variable representing calcium concentration.
    LinAlg::MPI::Vector calcium;

    double time_step; ///< Current time step.
    double time;      ///< Current time.

    /// Ionic current vector for @ref Iion_ICI.
    LinAlg::MPI::Vector Iion_ICI_vec;
    /// Ionic current vector for @ref Iion_ICI, without ghost entries.
    LinAlg::MPI::Vector Iion_ICI_vec_owned;

    /// Derivative of the ionic current vector w.r.t. u for @ref Iion_ICI.
    LinAlg::MPI::Vector dIion_du_ICI_vec;
    /// Derivative of the ionic current vector w.r.t. u for @ref Iion_ICI,
    /// without ghost entries.
    LinAlg::MPI::Vector dIion_du_ICI_vec_owned;

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

    /// FE degree
    double fe_degree;
  };

  template <class basis>
  void
  Ionic_DG<basis>::reinit_new(
    const DoFHandler<dim>::active_cell_iterator &cell_other)
  {
    this->cell = this->cell_next;

    vol_handler->reinit(this->cell);
    this->cell->get_dof_indices(dof_indices);

    // Get next locally owned cell.
    do
      {
        ++(this->cell_next);

        if (this->cell_next == this->endc)
          {
            break;
          }
    } while (!(this->cell_next)->is_locally_owned());

    AssertThrow(this->cell->center() == cell_other->center(),
                ExcLifexInternalError());

    post_reinit_callback(cell_other);
  }

  template <class basis>
  void
  Ionic_DG<basis>::initialize_3d(
    const std::shared_ptr<const utils::MeshHandler> &triangulation_,
    const unsigned int                              &fe_degree_,
    const std::shared_ptr<const Quadrature<dim>>    &quadrature_formula_,
    const std::shared_ptr<const AppliedCurrent>     &I_app_function_,
    const unsigned int                              &bdf_order_,
    const std::map<types::material_id, std::string> &map_id_volume_,
    const std::string                               &volume_label_)
  {
    AssertThrow(!standalone, ExcNotStandalone());

    triangulation = triangulation_;

    fe          = std::make_unique<basis>(fe_degree_);
    vol_handler = std::make_unique<DGVolumeHandler<dim>>(fe_degree_);

    fe_degree = fe_degree_;

    dof_handler->reinit(triangulation->get());
    dof_handler->distribute_dofs(*fe);
    this->QuadratureEvaluationFEMScalar::setup(*dof_handler,
                                               *quadrature_formula_,
                                               update_values);

    I_app_function = I_app_function_;
    prm_bdf_order  = bdf_order_;

    volume_label  = volume_label_;
    map_id_volume = map_id_volume_;

    // Get unique volume names.
    std::set<std::string> volumes;
    for (const auto &v : map_id_volume)
      volumes.insert(v.second);

    // Set volume label output suffix.
    // It is left empty in the case of single volume simulations.
    volume_label_output = (volumes.size() == 1) ?
                            "" :
                            (std::string("_") + utils::mangle(volume_label_));

    volume_dof_indices.set_size(dof_handler->n_dofs());
  }

  template <class basis>
  void
  Ionic_DG<basis>::declare_parameters(ParamHandler &params) const
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

      params.set_verbosity(VerbosityParam::Full);
      params.declare_entry(
        "Ischemic region threshold",
        "0.1",
        Patterns::Double(0),
        "Threshold that defines the minimum value in the "
        "ischemic region vector associated with grey zones or fibrosis [-].");
      params.reset_verbosity();

      // Only for 3D simulations.
      if (!standalone)
        {
          params.enter_subsection("Time solver 0D");
          {
            params.declare_entry(
              "Time step",
              "1e-4",
              Patterns::Double(0),
              "Time step [s]. NB: This parameter is only "
              "used when a limit cycle ionic solution is desired as initial "
              "guess for an electrophysiology simulation.");
          }
          params.leave_subsection();
        }

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

  template <class basis>
  void
  Ionic_DG<basis>::parse_parameters(ParamHandler &params)
  {
    params.enter_subsection_path(subsection_path);
    {
      // Read input parameters.
      prm_cell_type = params.get("Cell type");

      prm_calcium_rescale = params.get_double("Calcium rescale factor");

      prm_epi_tolerance  = params.get_double("Epicardium tolerance");
      prm_endo_tolerance = params.get_double("Endocardium tolerance");

      prm_ischemic_region_threshold =
        params.get_double("Ischemic region threshold");

      // Only for 3D simulations.
      if (!standalone)
        {
          params.enter_subsection("Time solver 0D");
          {
            prm_time_step = params.get_double("Time step");
          }
          params.leave_subsection();
        }

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

  template <class basis>
  void
  Ionic_DG<basis>::declare_parameters_0d(ParamHandler &params) const
  {
    AssertThrow(standalone, ExcStandaloneOnly());

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

  template <class basis>
  void
  Ionic_DG<basis>::parse_parameters_0d(ParamHandler &params)
  {
    AssertThrow(standalone, ExcStandaloneOnly());

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

  template <class basis>
  template <class VectorType>
  void
  Ionic_DG<basis>::solve_time_step(const VectorType &u,
                                   const double     &time_step_,
                                   const double     &time_)
  {
    TimerOutput::Scope timer_section(timer_output,
                                     subsection_label + " / Solve");

    time_step = time_step_;
    time      = time_;

    std::vector<double> ww_ext(n_variables);
    std::vector<double> ww_bdf(n_variables);
    std::vector<double> ww(n_variables);

    for (size_t n = 0; n < n_variables; ++n)
      {
        bdf_handler[n].time_advance(w_owned[n], true);

        // Copy into ghosted vectors for use in Iion_quadrature.
        w_ext[n] = bdf_handler[n].get_sol_extrapolation();
        w_bdf[n] = bdf_handler[n].get_sol_bdf();
      }

    const double &alpha_bdf = bdf_handler[0].get_alpha();

    const bool I_app_active = compute_I_app && I_app_function->is_active(time);

    // Get applied current at current time step.
    if (I_app_active)
      {
        VectorTools::interpolate(*dof_handler, *I_app_function, I_app_owned);
        I_app = I_app_owned;
      }

    // Maximum number of iterations needed to solve the ionic,
    // over all the dofs.
    unsigned int n_iter_max = 0;

    // Number of iterations performed to solve the ionic model
    // at current dof.
    unsigned int n_iter_dof = 0;

    for (auto idx : u.locally_owned_elements())
      {
        // Skip dofs that do not belong to current volume
        // and ischemic dofs.
        if (volume_dof_indices.is_element(idx))
          {
            const double ischemic_dof = (*ischemic_region)[idx];

            if (!utils::is_zero(ischemic_dof, prm_scar_tolerance))
              {
                for (size_t n = 0; n < n_variables; ++n)
                  {
                    ww_ext[n] = w_ext[n][idx];
                    ww_bdf[n] = w_bdf[n][idx];
                  }

                std::tie(ww, n_iter_dof) =
                  solve_time_step_0d(u[idx],
                                     alpha_bdf,
                                     ww_bdf,
                                     ww_ext,
                                     (*endo_epi)[idx],
                                     ischemic_dof,
                                     I_app_active ? I_app[idx] : 0.0);

                n_iter_max = std::max(n_iter_dof, n_iter_max);

                for (size_t n = 0; n < n_variables; ++n)
                  {
                    w_owned[n][idx] = ww[n];
                  }

                // Update calcium.
                calcium_owned[idx] = compute_calcium(ww);
              }
          }
      }

    n_iter_max = Utilities::MPI::max(n_iter_max, mpi_comm);
    pcout << "\t" << iterations_log_string(n_iter_max);

    // Compress ionic variables.
    for (size_t n = 0; n < n_variables; ++n)
      {
        w_owned[n].compress(VectorOperation::insert);
        w[n] = w_owned[n];
      }

    calcium_owned.compress(VectorOperation::insert);
    calcium = calcium_owned;
  }

  template <class basis>
  void
  Ionic_DG<basis>::solve_electrophysiology_time_step_0d(const bool &verbose)
  {
    double              dIion_du_val;
    double              Iion_val;
    const double       &alpha_bdf = bdf_handler_0d[0].get_alpha();
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
      solve_time_step_0d(u_ext_0d, alpha_bdf, ww_bdf, ww_ext, 0, 1, Iapp);

    calcium_0d = compute_calcium(w_0d);

    if (verbose)
      pcout << "\t" << iterations_log_string(n_iter) << std::endl;

    std::tie(Iion_val, dIion_du_val) = Iion(u_0d, u_ext_0d, w_0d, 1, 0.5);


    u_0d = (u_bdf_0d - prm_time_step / dt_fact * (Iion_val - Iapp)) / alpha_bdf;

    time += prm_time_step;
  }

  template <class basis>
  void
  Ionic_DG<basis>::setup_system(const bool &initialize_ICI)
  {
    AssertThrow(dof_handler != nullptr, ExcNotInitialized());

    IndexSet owned_dofs = dof_handler->locally_owned_dofs();

    IndexSet relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(*dof_handler, relevant_dofs);

    w.resize(n_variables);
    w_owned.resize(n_variables);

    bdf_handler.resize(n_variables);
    if (I_app_function->VARP_protocol_enabled())
      {
        bdf_handler_backup.resize(n_variables);
      }

    w_bdf.resize(n_variables);
    w_ext.resize(n_variables);

    dw.resize(n_variables);
    w_owned_backup.resize(n_variables);

    if (compute_I_app)
      {
        I_app.reinit(owned_dofs, relevant_dofs, mpi_comm);
        I_app_owned.reinit(owned_dofs, mpi_comm);
      }

    if (do_0d_simulation_before_initialization)
      {
        pcout << "Running ionic model 0D for initialization..." << std::endl;
        run_0d(false);
        pcout << "Finished running ionic model 0D for initialization."
              << std::endl;
        w_initial_conditions = w_0d;
        Ca_initial_condition = calcium_0d;
      }
    else
      {
        w_initial_conditions = setup_initial_conditions();
        Ca_initial_condition = compute_calcium(w_initial_conditions);
      }

    for (size_t n = 0; n < n_variables; ++n)
      {
        w[n].reinit(owned_dofs, relevant_dofs, mpi_comm);
        w_owned[n].reinit(owned_dofs, mpi_comm);

        w_bdf[n].reinit(owned_dofs, relevant_dofs, mpi_comm);
        w_ext[n].reinit(owned_dofs, relevant_dofs, mpi_comm);

        dw[n].reinit(owned_dofs, mpi_comm);
        w_owned_backup[n].reinit(owned_dofs, mpi_comm);

        w[n] = w_owned[n] = w_initial_conditions[n];

        // Initialize BDF handler.
        std::vector<LinAlg::MPI::Vector> w_init(prm_bdf_order, w_owned[n]);
        bdf_handler[n].initialize(prm_bdf_order, w_init);

        // Initialize backup BDF handler in case of restart.
        if (I_app_function->VARP_protocol_enabled())
          {
            bdf_handler_backup[n].initialize(prm_bdf_order, w_init);
          }
      }

    calcium_owned.reinit(owned_dofs, mpi_comm);
    calcium.reinit(owned_dofs, relevant_dofs, mpi_comm);

    calcium = calcium_owned = Ca_initial_condition;

    if (initialize_ICI)
      {
        IndexSet owned_dofs = dof_handler->locally_owned_dofs();

        IndexSet relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(*dof_handler, relevant_dofs);

        Iion_ICI_vec.reinit(owned_dofs, relevant_dofs, mpi_comm);
        Iion_ICI_vec_owned.reinit(owned_dofs, mpi_comm);
        dIion_du_ICI_vec.reinit(owned_dofs, relevant_dofs, mpi_comm);
        dIion_du_ICI_vec_owned.reinit(owned_dofs, mpi_comm);
      }

    // Determine global dof indices of current volume.
    const unsigned int                   dofs_per_cell = fe->dofs_per_cell;
    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
    for (const auto &cell : dof_handler->active_cell_iterators())
      {
        if (cell->is_locally_owned() || cell->is_ghost())
          {
            cell->get_dof_indices(dof_indices);

            if (map_id_volume.at(cell->material_id()) == volume_label)
              {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    volume_dof_indices.add_index(dof_indices[i]);
                  }
              }
          }
      }
  }

  template <class basis>
  void
  Ionic_DG<basis>::setup_system_0d()
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

  template <class basis>
  void
  Ionic_DG<basis>::run_0d(const bool &verbose)
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

  template <class basis>
  void
  Ionic_DG<basis>::assemble_Iion_ICI(const LinAlg::MPI::Vector &u,
                                     const LinAlg::MPI::Vector &u_ext)

  {
    assemble_Iion_ICI_tpl<LinAlg::MPI::Vector>(u,
                                               u_ext,
                                               Iion_ICI_vec_owned,
                                               dIion_du_ICI_vec_owned);

    Iion_ICI_vec_owned.compress(VectorOperation::insert);
    Iion_ICI_vec = Iion_ICI_vec_owned;

    dIion_du_ICI_vec_owned.compress(VectorOperation::insert);
    dIion_du_ICI_vec = dIion_du_ICI_vec_owned;
  }

  template <class basis>
  template <class VectorType>
  void
  Ionic_DG<basis>::assemble_Iion_ICI_tpl(const VectorType &u,
                                         const VectorType &u_ext,
                                         VectorType       &Iion_ICI_vec_owned,
                                         VectorType &dIion_du_ICI_vec_owned)
  {
    std::vector<double> w_loc(n_variables);

    double Iion_tmp, dIion_du_tmp;

    for (auto idx : Iion_ICI_vec_owned.locally_owned_elements())
      {
        // Skip dofs that do not belong to current volume.
        if (volume_dof_indices.is_element(idx))
          {
            for (size_t n = 0; n < n_variables; ++n)
              {
                w_loc[n] = w_owned[n][idx];
              }

            std::tie(Iion_tmp, dIion_du_tmp) = Iion(u[idx],
                                                    u_ext[idx],
                                                    w_loc,
                                                    (*ischemic_region)[idx],
                                                    (*endo_epi)[idx]);

            Iion_ICI_vec_owned[idx]     = Iion_tmp;
            dIion_du_ICI_vec_owned[idx] = dIion_du_tmp;
          }
      }
  }

  template <class basis>
  std::pair<std::vector<double>, std::vector<double>>
  Ionic_DG<basis>::Iion_ICI()
  {
    AssertThrow(dof_handler != nullptr, ExcNotInitialized());

    const unsigned int n_q_points = std::pow(fe_degree + 2, dim);

    std::vector<double> Iion_loc(n_q_points, 0.0);
    std::vector<double> dIion_du_loc(n_q_points, 0.0);
    std::vector<double> ischemic_region_loc(n_q_points, 0.0);

    // Determine if cell is in current volume.
    const bool cell_in_volume =
      std::any_of(dof_indices.begin(),
                  dof_indices.end(),
                  [this](const types::global_dof_index &dof_indices_i) {
                    return volume_dof_indices.is_element(dof_indices_i);
                  });

    if (cell_in_volume)
      {
        const Tensor<2, dim> BJinv = vol_handler->get_jacobian_inverse();
        const double         det   = 1 / determinant(BJinv);

        for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
          {
            const double ischemic_dof = (*ischemic_region)(dof_indices[i]);

            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                ischemic_region_loc[q] +=
                  ischemic_dof *
                  fe->shape_value(i, vol_handler->quadrature_ref(q)) *
                  vol_handler->quadrature_weight(q) * det;
              }
          }

        for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
          {
            const double Iion_ICI_vec_dof = Iion_ICI_vec[dof_indices[i]];
            const double dIion_du_ICI_vec_dof =
              dIion_du_ICI_vec[dof_indices[i]];

            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                // If not ischemic quadrature node.
                if (!utils::is_zero(ischemic_region_loc[q], prm_scar_tolerance))
                  {
                    Iion_loc[q] +=
                      Iion_ICI_vec_dof *
                      fe->shape_value(i, vol_handler->quadrature_ref(q)) *
                      vol_handler->quadrature_weight(q) * det;

                    dIion_du_loc[q] +=
                      dIion_du_ICI_vec_dof *
                      fe->shape_value(i, vol_handler->quadrature_ref(q)) *
                      vol_handler->quadrature_weight(q) * det;
                  }
              }
          }
      }

    return std::make_pair(Iion_loc, dIion_du_loc);
  }

  template <class basis>
  std::pair<std::vector<double>, std::vector<double>>
  Ionic_DG<basis>::Iion_ICI_hybrid(const LinAlg::MPI::Vector &u,
                                   const LinAlg::MPI::Vector &u_ext)
  {
    AssertThrow(dof_handler != nullptr, ExcNotInitialized());

    const unsigned int n_q_points = std::pow(fe_degree + 2, dim);

    std::vector<double> Iion_loc(n_q_points, 0.0);
    std::vector<double> dIion_du_loc(n_q_points, 0.0);

    std::vector<double> u_ext_loc(n_q_points, 0.0);
    std::vector<double> ischemic_region_loc(n_q_points, 0.0);
    std::vector<double> endo_epi_loc(n_q_points, 0.0);

    // Determine if cell is in current volume.
    const bool cell_in_volume =
      std::any_of(dof_indices.begin(),
                  dof_indices.end(),
                  [this](const types::global_dof_index &dof_indices_i) {
                    return volume_dof_indices.is_element(dof_indices_i);
                  });

    if (cell_in_volume)
      {
        const Tensor<2, dim> BJinv = vol_handler->get_jacobian_inverse();
        const double         det   = 1 / determinant(BJinv);

        for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
          {
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                u_ext_loc[q] +=
                  u_ext(dof_indices[i]) *
                  fe->shape_value(i, vol_handler->quadrature_ref(q)) *
                  vol_handler->quadrature_weight(q) * det;

                ischemic_region_loc[q] +=
                  (*ischemic_region)(dof_indices[i]) *
                  fe->shape_value(i, vol_handler->quadrature_ref(q)) *
                  vol_handler->quadrature_weight(q) * det;

                endo_epi_loc[q] +=
                  (*endo_epi)(dof_indices[i]) *
                  fe->shape_value(i, vol_handler->quadrature_ref(q)) *
                  vol_handler->quadrature_weight(q) * det;
              }
          }

        std::vector<double> w_i(n_variables);

        double Iion_tmp;
        double dIion_du_tmp;

        for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
          {
            for (size_t n = 0; n < n_variables; ++n)
              {
                w_i[n] = w[n](dof_indices[i]);
              }

            // If not ischemic dof.
            if (!utils::is_zero((*ischemic_region)(dof_indices[i]),
                                prm_scar_tolerance))
              {
                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    std::tie(Iion_tmp, dIion_du_tmp) =
                      Iion(u[dof_indices[i]],
                           u_ext_loc[q],
                           w_i,
                           ischemic_region_loc[q],
                           endo_epi_loc[q]);
                    Iion_loc[q] +=
                      Iion_tmp *
                      fe->shape_value(i, vol_handler->quadrature_ref(q)) *
                      vol_handler->quadrature_weight(q) * det;

                    dIion_du_loc[q] +=
                      dIion_du_tmp *
                      fe->shape_value(i, vol_handler->quadrature_ref(q)) *
                      vol_handler->quadrature_weight(q) * det;
                  }
              }
          }
      }

    return std::make_pair(Iion_loc, dIion_du_loc);
  }

  template <class basis>
  std::pair<std::vector<double>, std::vector<double>>
  Ionic_DG<basis>::Iion_SVI(const std::vector<double> &u_loc,
                            const std::vector<double> &u_ext_loc)
  {
    AssertThrow(dof_handler != nullptr, ExcNotInitialized());

    const unsigned int n_q_points = std::pow(fe_degree + 2, dim);

    std::vector<double> Iion_loc(n_q_points, 0.0);
    std::vector<double> dIion_du_loc(n_q_points, 0.0);

    std::vector<std::vector<double>> w_loc(n_q_points,
                                           std::vector<double>(n_variables,
                                                               0.0));

    std::vector<double> ischemic_region_loc(n_q_points, 0.0);
    std::vector<double> endo_epi_loc(n_q_points, 0.0);

    // Determine if cell is in current volume.
    const bool cell_in_volume =
      std::any_of(dof_indices.begin(),
                  dof_indices.end(),
                  [this](const types::global_dof_index &dof_indices_i) {
                    return volume_dof_indices.is_element(dof_indices_i);
                  });

    if (cell_in_volume)
      {
        const Tensor<2, dim> BJinv = vol_handler->get_jacobian_inverse();
        const double         det   = 1 / determinant(BJinv);

        for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
          {
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                for (size_t n = 0; n < n_variables; ++n)
                  {
                    w_loc[q][n] +=
                      w[n](dof_indices[i]) *
                      fe->shape_value(i, vol_handler->quadrature_ref(q)) *
                      vol_handler->quadrature_weight(q) * det;
                  }

                ischemic_region_loc[q] +=
                  (*ischemic_region)(dof_indices[i]) *
                  fe->shape_value(i, vol_handler->quadrature_ref(q)) *
                  vol_handler->quadrature_weight(q) * det;

                endo_epi_loc[q] +=
                  (*endo_epi)(dof_indices[i]) *
                  fe->shape_value(i, vol_handler->quadrature_ref(q)) *
                  vol_handler->quadrature_weight(q) * det;
              }
          }

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            // If not ischemic quadrature node.
            if (!utils::is_zero(ischemic_region_loc[q], prm_scar_tolerance))
              {
                std::tie(Iion_loc[q], dIion_du_loc[q]) =
                  Iion(u_loc[q],
                       u_ext_loc[q],
                       w_loc[q],
                       ischemic_region_loc[q],
                       endo_epi_loc[q]);
              }
          }
      }

    return std::make_pair(Iion_loc, dIion_du_loc);
  }

  template <class basis>
  std::pair<std::vector<double>, std::vector<double>>
  Ionic_DG<basis>::Iion_quadrature(const std::vector<double> &u_loc,
                                   const std::vector<double> &u_ext_loc)
  {
    AssertThrow(dof_handler != nullptr, ExcNotInitialized());

    const unsigned int n_q_points = std::pow(fe_degree + 2, dim);

    std::vector<double> Iion_loc(n_q_points, 0.0);
    std::vector<double> dIion_du_loc(n_q_points, 0.0);

    std::vector<double> ww_ext(n_variables, 0.0);
    std::vector<double> ww_bdf(n_variables, 0.0);
    std::vector<double> ww(n_variables, 0.0);

    double ischemic_region_q;
    double endo_epi_q;
    double I_app_q;

    unsigned int n_iter_q;

    const double &alpha_bdf = bdf_handler[0].get_alpha();

    // Determine if cell is in current volume.
    const bool cell_in_volume =
      std::any_of(dof_indices.begin(),
                  dof_indices.end(),
                  [this](const types::global_dof_index &dof_indices_i) {
                    return volume_dof_indices.is_element(dof_indices_i);
                  });

    if (cell_in_volume)
      {
        const Tensor<2, dim> BJinv = vol_handler->get_jacobian_inverse();
        const double         det   = 1 / determinant(BJinv);

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (size_t n = 0; n < n_variables; ++n)
              {
                ww_ext[n] = 0;
                ww_bdf[n] = 0;
              }

            ischemic_region_q = endo_epi_q = I_app_q = 0;

            for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
              {
                for (size_t n = 0; n < n_variables; ++n)
                  {
                    ww_ext[n] +=
                      w_ext[n][dof_indices[i]] *
                      fe->shape_value(i, vol_handler->quadrature_ref(q)) *
                      vol_handler->quadrature_weight(q) * det;

                    ww_bdf[n] +=
                      w_bdf[n][dof_indices[i]] *
                      fe->shape_value(i, vol_handler->quadrature_ref(q)) *
                      vol_handler->quadrature_weight(q) * det;
                  }

                ischemic_region_q +=
                  (*ischemic_region)[dof_indices[i]] *
                  fe->shape_value(i, vol_handler->quadrature_ref(q)) *
                  vol_handler->quadrature_weight(q) * det;

                endo_epi_q +=
                  (*endo_epi)[dof_indices[i]] *
                  fe->shape_value(i, vol_handler->quadrature_ref(q)) *
                  vol_handler->quadrature_weight(q) * det;

                if (compute_I_app)
                  {
                    I_app_q +=
                      I_app[dof_indices[i]] *
                      fe->shape_value(i, vol_handler->quadrature_ref(q)) *
                      vol_handler->quadrature_weight(q) * det;
                  }
              }

            // If not ischemic quadrature node.
            if (!utils::is_zero(ischemic_region_q, prm_scar_tolerance))
              {
                std::tie(ww, n_iter_q) = solve_time_step_0d(u_loc[q],
                                                            alpha_bdf,
                                                            ww_bdf,
                                                            ww_ext,
                                                            endo_epi_q,
                                                            ischemic_region_q,
                                                            I_app_q);

                std::tie(Iion_loc[q], dIion_du_loc[q]) = Iion(
                  u_loc[q], u_ext_loc[q], ww, ischemic_region_q, endo_epi_q);
              }
          }
      }

    return std::make_pair(Iion_loc, dIion_du_loc);
  }

  template <class basis>
  double
  Ionic_DG<basis>::compute_calcium(const std::vector<double> &w) const
  {
    return prm_calcium_rescale * compute_calcium_raw(w);
  }
  //
  // template <class basis>
  // void
  // Ionic_DG<basis>::attach_output(DataOut<dim> &     data_out,
  //                      const std::string &output_mode) const
  // {
  //   data_out.add_data_vector(*dof_handler,
  //                            calcium,
  //                            "calcium" + volume_label_output);
  //
  //   if (output_mode == "All")
  //     {
  //       for (size_t n = 0; n < n_variables; ++n)
  //         {
  //           data_out.add_data_vector(*dof_handler,
  //                                    w[n],
  //                                    "w" + std::to_string(n) +
  //                                      volume_label_output);
  //         }
  //     }
  // }

  template <class basis>
  void
  Ionic_DG<basis>::output_results_0d()
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

  template <class basis>
  void
  Ionic_DG<basis>::declare_entries_csv(utils::CSVWriter  &csv_writer,
                                       const std::string &output_mode) const
  {
    csv_writer.declare_entries({"calcium_min" + volume_label_output,
                                "calcium_avg" + volume_label_output,
                                "calcium_max" + volume_label_output,
                                "calcium_point" + volume_label_output});

    if (output_mode == "All")
      {
        for (size_t n = 0; n < n_variables; ++n)
          {
            csv_writer.declare_entries(
              {"w" + std::to_string(n) + "_min" + volume_label_output,
               "w" + std::to_string(n) + "_avg" + volume_label_output,
               "w" + std::to_string(n) + "_max" + volume_label_output,
               "w" + std::to_string(n) + "_point" + volume_label_output});
          }
      }
  }

  // template <class basis>
  // void
  // Ionic_DG<basis>::set_entries_csv(utils::CSVWriter & csv_writer,
  //                        const std::string &output_mode) const
  // {
  //   csv_writer.set_entries(
  //     {{"calcium_min" + volume_label_output,
  //       utils::vec_min(calcium_owned, volume_dof_indices)},
  //      {"calcium_avg" + volume_label_output,
  //       utils::vec_avg(calcium_owned, volume_dof_indices)},
  //      {"calcium_max" + volume_label_output,
  //       utils::vec_max(calcium_owned, volume_dof_indices)}});
  //
  //   if (output_mode == "All")
  //     {
  //       for (size_t n = 0; n < n_variables; ++n)
  //         {
  //           csv_writer.set_entries(
  //             {{"w" + std::to_string(n) + "_min" + volume_label_output,
  //               utils::vec_min(w_owned[n], volume_dof_indices)},
  //              {"w" + std::to_string(n) + "_avg" + volume_label_output,
  //               utils::vec_avg(w_owned[n], volume_dof_indices)},
  //              {"w" + std::to_string(n) + "_max" + volume_label_output,
  //               utils::vec_max(w_owned[n], volume_dof_indices)}});
  //         }
  //     }
  //
  //   // The purpose here is to visualize the time transient of some variables
  //   of
  //   // interest at a specific dof: sometimes global information such as
  //   // min, max and mean do not provide enough insight, thus it is useful to
  //   // plot the time transient at a specific dof. The rationale is simply to
  //   // take an arbitrary dof (but always the same dof across different
  //   // timesteps).
  //   //
  //   // Here, since the aim is to store a random point per volume (and every
  //   // volume may be shared among multiple processes), the value retrieved
  //   // is simply the first owned dof in the current volume belonging to the
  //   // process with the lowest rank, which is then communicated for writing.
  //
  //   // Determine the lowest rank process owning a dof in the current volume.
  //   const bool is_volume_owned = !volume_dof_indices.is_empty();
  //
  //   const std::vector<bool> is_volume_owned_by_rank =
  //     Utilities::MPI::all_gather(mpi_comm, is_volume_owned);
  //
  //   const size_t mpi_lowest_rank_owning =
  //     std::distance(is_volume_owned_by_rank.begin(),
  //                   std::find_if(is_volume_owned_by_rank.begin(),
  //                                is_volume_owned_by_rank.end(),
  //                                [](const bool owned) {
  //                                  return owned == true;
  //                                }));
  //
  //   // Set variables to lowest double value on all processes.
  //   double              calcium_volume =
  //   std::numeric_limits<double>::lowest(); std::vector<double>
  //   w_volume(n_variables, calcium_volume);
  //
  //   // Set actual values on lowest rank owning process.
  //   if (mpi_rank == mpi_lowest_rank_owning)
  //     {
  //       const types::global_dof_index idx_first_owned_dof =
  //         *(volume_dof_indices.begin());
  //
  //       calcium_volume = calcium_owned[idx_first_owned_dof];
  //
  //       for (size_t n = 0; n < n_variables; ++n)
  //         w_volume[n] = w_owned[n][idx_first_owned_dof];
  //     }
  //
  //   // Reduce values among processes.
  //   calcium_volume = Utilities::MPI::max(calcium_volume, mpi_comm);
  //   for (size_t n = 0; n < n_variables; ++n)
  //     w_volume[n] = Utilities::MPI::max(w_volume[n], mpi_comm);
  //
  //   if (csv_writer.is_active())
  //     {
  //       csv_writer.set_entries(
  //         {{"calcium_point" + volume_label_output, calcium_volume}});
  //
  //       if (output_mode == "All")
  //         {
  //           for (size_t n = 0; n < n_variables; ++n)
  //             {
  //               csv_writer.set_entries(
  //                 {{"w" + std::to_string(n) + "_point" + volume_label_output,
  //                   w_volume[n]}});
  //             }
  //         }
  //     }
  // }
} // namespace lifex

#endif /* LIFEX_PHYSICS_IONIC__DG_HPP_ */
