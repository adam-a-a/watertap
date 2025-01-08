#################################################################################
# WaterTAP Copyright (c) 2020-2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################
from pyomo.environ import (
    ConcreteModel,
    value,
    TransformationFactory,
    units as pyunits,
    Block,
    Constraint,
    assert_optimal_termination,
    check_optimal_termination,
    Objective,    
)
from pyomo.network import Arc, Port
from pyomo.util.check_units import assert_units_consistent
from idaes.core import FlowsheetBlock, UnitModelBlockData
from idaes.core.solvers import get_solver
from idaes.core.util.initialization import propagate_state
from idaes.core.util import DiagnosticsToolbox

from idaes.core.util.exceptions import ConfigurationError
from idaes.models.unit_models.translator import Translator
from idaes.models.unit_models import Mixer, Separator, Product, Feed
from idaes.models.unit_models.mixer import MomentumMixingType
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
from idaes.core.util.tables import arcs_to_stream_dict, stream_states_dict, _get_state_from_port

# TODO: bring costing in subsequent PR
# from idaes.core import UnitModelCostingBlock

from watertap.property_models.seawater_prop_pack import SeawaterParameterBlock
from watertap.property_models.NaCl_T_dep_prop_pack import NaClParameterBlock

from watertap.unit_models.reverse_osmosis_0D import (
    ReverseOsmosis0D,
    ConcentrationPolarizationType,
    MassTransferCoefficient,
    PressureChangeType,
)
from watertap.core import ModuleType

from watertap.unit_models.reverse_osmosis_1D import ReverseOsmosis1D
from watertap.unit_models.pressure_exchanger import PressureExchanger
from watertap.unit_models.pressure_changer import Pump, EnergyRecoveryDevice
from watertap.flowsheets.RO_with_energy_recovery.RO_with_energy_recovery import (
    ERDtype,
    erd_type_not_found,
)
from watertap.unit_models.uv_aop import Ultraviolet0D
from watertap.core.util.initialization import assert_degrees_of_freedom, check_solve
from watertap.core.wt_database import Database
from watertap.property_models.multicomp_aq_sol_prop_pack import (
    MCASParameterBlock,
    MaterialFlowBasis,
)
from watertap.unit_models.zero_order import (
    MicroFiltrationZO,
    CartridgeFiltrationZO,
    UVZO,
    UVAOPZO,
    # TODO: consider addition of some of the following units in subsequent PR
    # ChemicalAdditionZO,
    # StaticMixerZO,
    # StorageTankZO,
    # MediaFiltrationZO,
    # BackwashSolidsHandlingZO,
)

# TODO: handle costing in subsequent PR
# from watertap.costing.zero_order_costing import ZeroOrderCosting
# from watertap.costing import WaterTAPCosting
from idaes.core.util.misc import StrEnum

__author__ = "Adam Atia"
# Set up logger
_log = idaeslog.getLogger(__name__)

"""
This flowsheet represents a typical configuration for advanced treatment of treated wastewater effluent; i.e., MF/UF-RO-UV.
The flowsheet currently includes microfiltration to cartridge filtration, followed by reverse osmosis and an ultraviolet reactor.
The flowsheet is setup to be configured in a flexible manner:
- choose between NaCl and seawater (TDS) property models
- choose level of modeling detail for RO model
- choose whether to include UV or UV/AOP after RO, and if so, choose level of modeling detail
"""


class rodimension(StrEnum):
    """
    Options for dimensionality of RO unit model: 0d, 1d
    """

    zero_d = "0d"
    one_d = "1d"


class uvdimension(StrEnum):
    """
    Options for dimensionality of UV unit model: none, zo, 0d
    """

    none = "none"
    zo = "zo"
    zero_d = "0d"


def main(
    ro_props="seawater",
    ro_dimension="0d",
    erd_config=ERDtype.no_ERD,
    uvdimension=uvdimension.zo,
    has_aop=False,
    has_measured_vars=True,
    diagnostics_active=False,
):
    m = build(
        ro_props=ro_props,
        ro_dimension=ro_dimension,
        erd_config=erd_config,
        uvdimension=uvdimension,
        has_aop=has_aop,
        has_measured_vars=has_measured_vars
    )

    set_operating_conditions(m)

    dt = DiagnosticsToolbox(m)
    dt.report_structural_issues()

    if diagnostics_active:
        results = {}

        try:
            assert_degrees_of_freedom(m, 0)
            assert_units_consistent(m)
            initialize_system(m)
            assert_degrees_of_freedom(m, 0)
            if diagnostics_active:
                dt.report_numerical_issues()
            results = solve(m, tee=True, checkpoint="solve flowsheet after initializing system")
            assert_optimal_termination(results)
            display_results(m)
            m.fs.stream_table()
            setup_optimization(m)

        except:
            pass


        # TODO: handle costing in next PR
        # add_costing(m)
        # assert_degrees_of_freedom(m, 0)
        # m.fs.costing.initialize()

        # results = solve(m, checkpoint="solve flowsheet after costing")

        # display_results(m)
        return m, results, dt

        # except:
            # return _, _, dt
    else:
        assert_degrees_of_freedom(m, 0)
        assert_units_consistent(m)
        initialize_system(m)
        assert_degrees_of_freedom(m, 0)

        results = solve(m, checkpoint="solve flowsheet after initializing system")
        assert_optimal_termination(results)
        display_results(m)
        m.fs.stream_table()

        # TODO:
        # add_costing(m)
        # assert_degrees_of_freedom(m, 0)
        # m.fs.costing.initialize()

        # results = solve(m, checkpoint="solve flowsheet after costing")

    return m, results


def build(ro_props, ro_dimension, erd_config, uvdimension, has_aop, has_measured_vars):
    """
    ro_props: choose between "NaCl" and "Seawater" prop models for RO
    ro_dimension: choose between 0d, 1d
    """
    # flowsheet set up
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    # Property Models: MCAS and either NaCl/Seawater

    # Choose between NaCl or Seawater prop models for RO
    m.fs.ro_props, m.fs.ro_ion = get_ro_props(ro_props)

    # Use MCAS for whole flowsheet, except RO
    m.fs.mcas_props = MCASParameterBlock(
        solute_list=[m.fs.ro_ion, "tss"],
        diffusivity_data={("Liq", m.fs.ro_ion): 1e-9, ("Liq", "tss"): 1e-9},
        mw_data={m.fs.ro_ion: None, "tss": None},
        material_flow_basis=MaterialFlowBasis.mass,
        ignore_neutral_charge=True,
    )
    # m.fs.mcas_props._metadata.add_properties({"pH": {"method": None}})
    # Add Database for initially parameterization of ZO models
    m.db = Database()

    # Feed block
    m.fs.feed = Feed(property_package=m.fs.mcas_props)

    # Microfiltration Pump
    m.fs.mf_pump = Pump(property_package=m.fs.mcas_props)

    # Microfiltration
    m.fs.mf = MicroFiltrationZO(property_package=m.fs.mcas_props, database=m.db)

    # Cartridge Filtration Pump
    m.fs.cf_pump = Pump(property_package=m.fs.mcas_props)

    # CF
    m.fs.cf = CartridgeFiltrationZO(property_package=m.fs.mcas_props, database=m.db)

    # Translate MCAS to RO property model
    m.fs.mcas_to_ro_translator = Translator(
        inlet_property_package=m.fs.mcas_props, outlet_property_package=m.fs.ro_props
    )

    @m.fs.mcas_to_ro_translator.Constraint(m.fs.ro_props.component_list)
    def eq_flow_mass_comp(blk, j):
        return (
            blk.properties_in[0].flow_mass_phase_comp["Liq", j]
            == blk.properties_out[0].flow_mass_phase_comp["Liq", j]
        )

    # Add Mixer for concentrate recirculation
    m.fs.feed_mixer = Mixer(property_package=m.fs.ro_props,
                            # momentum_mixing_type=MomentumMixingType.equality,
                            inlet_list=["cf_effluent", "recirculated_concentrate"]) 
    # RO Train =====================================================================
    # High-pressure RO pump
    m.fs.hp_pump = Pump(property_package=m.fs.ro_props)

    # --- Reverse Osmosis Block ---
    m.fs.RO = get_ro_model(dimension=ro_dimension, ro_props=m.fs.ro_props)

    # --- ERD blocks ---
    if erd_config == ERDtype.pressure_exchanger:
        m.fs.feed_to_hp_and_erd_splitter = Separator(
            property_package=m.fs.ro_props, outlet_list=["hp_pump", "erd"]
        )

        m.fs.erd = PressureExchanger(property_package=m.fs.ro_props)
        m.fs.booster_pump = Pump(property_package=m.fs.ro_props)
        m.fs.hp_and_booster_mixer = Mixer(
            property_package=m.fs.ro_props,
            momentum_mixing_type=MomentumMixingType.equality,
            inlet_list=["hp_pump", "booster_pump"],
        )

    elif erd_config == ERDtype.pump_as_turbine:
        # add energy recovery turbine block
        m.fs.erd = EnergyRecoveryDevice(property_package=m.fs.ro_props)

    elif erd_config == ERDtype.no_ERD:
        pass
    else:
        erd_type_not_found(erd_config)
    # ===============================================================================
    # Add splitter for recirculated concentrate and waste concentrate
    m.fs.concentrate_splitter = Separator(property_package=m.fs.ro_props, outlet_list=["recirculated_concentrate", "waste_brine"])


    # Translate RO to MCAS property model
    m.fs.ro_to_mcas_translator = Translator(
        inlet_property_package=m.fs.ro_props, outlet_property_package=m.fs.mcas_props
    )

    #TODO: this seems incorrect since translator connects to permeate and TSS shouldn't end up there
    @m.fs.ro_to_mcas_translator.Constraint(m.fs.mcas_props.component_list)
    def eq_flow_mass_comp(blk, j):
        if j.lower() == "tss":
            blk.properties_out[0].flow_mass_phase_comp["Liq", j].fix(0)
            return Constraint.Skip
        else:
            return (
                blk.properties_in[0].flow_mass_phase_comp["Liq", j]
                == blk.properties_out[0].flow_mass_phase_comp["Liq", j]
            )

    # UV
    # TODO: Add UV as an option: None, UV, UV-AOP, UVZO, UVAOPZO?
    m.fs.uv = get_uv_model(
        m, uv_props=m.fs.mcas_props, dimension=uvdimension, has_aop=has_aop
    )

    # Product blocks for permeate and disposal
    m.fs.product_water = Product(property_package=m.fs.mcas_props)
    m.fs.waste_brine = Product(property_package=m.fs.ro_props)
    # TODO: consider adding mixer to collect MF, CF, and RO waste streams and send to drain (Product block)

    # connections
    m.fs.feed_to_mf_pump = Arc(source=m.fs.feed.outlet, destination=m.fs.mf_pump.inlet)
    m.fs.mf_pump_to_mf = Arc(source=m.fs.mf_pump.outlet, destination=m.fs.mf.inlet)
    m.fs.mf_to_cf_pump = Arc(source=m.fs.mf.treated, destination=m.fs.cf_pump.inlet)
    m.fs.cf_pump_to_cf = Arc(source=m.fs.cf_pump.outlet, destination=m.fs.cf.inlet)
    m.fs.cf_to_translator = Arc(
        source=m.fs.cf.treated, destination=m.fs.mcas_to_ro_translator.inlet
    )

    if erd_config == ERDtype.pressure_exchanger:
        raise NotImplementedError(
            f"While arc connections are ready, setting up the square problem (setting DOF=0) has not been completed for erd_config={erd_config}"
        )
        # TODO: finalize arc connections and square prob setup with this erd_config
        m.fs.translator_to_hp_erd_splitter = Arc(
            source=m.fs.mcas_to_ro_translator.outlet,
            destination=m.fs.feed_to_hp_and_erd_splitter.inlet,
        )
        m.fs.splitter_to_hp_pump = Arc(
            source=m.fs.feed_to_hp_and_erd_splitter.hp_pump,
            destination=m.fs.hp_pump.inlet,
        )
        m.fs.splitter_to_erd = Arc(
            source=m.fs.feed_to_hp_and_erd_splitter.erd,
            destination=m.fs.erd.low_pressure_inlet,
        )
        m.fs.erd_to_booster_pump = Arc(
            source=m.fs.erd.low_pressure_outlet, destination=m.fs.booster_pump.inlet
        )
        m.fs.booster_pump_to_mixer = Arc(
            source=m.fs.booster_pump.outlet,
            destination=m.fs.hp_and_booster_mixer.booster_pump,
        )
        m.fs.hp_pump_to_mixer = Arc(
            source=m.fs.hp_pump.outlet, destination=m.fs.hp_and_booster_mixer.hp_pump
        )
        m.fs.mixer_to_RO = Arc(
            source=m.fs.hp_and_booster_mixer.outlet, destination=m.fs.RO.inlet
        )
        m.fs.RO_brine_to_erd = Arc(
            source=m.fs.RO.retentate, destination=m.fs.erd.high_pressure_inlet
        )
        m.fs.erd_to_waste = Arc(
            source=m.fs.erd.high_pressure_outlet, destination=m.fs.waste_brine.inlet
        )
    elif erd_config == ERDtype.pump_as_turbine:
        raise NotImplementedError(
            f"While arc connections are ready, setting up the square problem (setting DOF=0) has not been completed for erd_config={erd_config}"
        )
        # TODO: finalize arc connections and square prob setup with this erd_config

        m.fs.translator_to_hp_pump = Arc(
            source=m.fs.mcas_to_ro_translator.outlet, destination=m.fs.hp_pump.inlet
        )
        m.fs.hp_pump_to_RO = Arc(source=m.fs.hp_pump.outlet, destination=m.fs.RO.inlet)
        m.fs.RO_brine_to_erd = Arc(source=m.fs.RO.retentate, destination=m.fs.erd.inlet)
        m.fs.erd_to_waste = Arc(
            source=m.fs.erd.outlet, destination=m.fs.waste_brine.inlet
        )
    elif erd_config == ERDtype.no_ERD:

        m.fs.translator_to_mixer = Arc(
            source=m.fs.mcas_to_ro_translator.outlet, destination=m.fs.feed_mixer.cf_effluent)
        m.fs.mixer_to_hp_pump = Arc(
            source=m.fs.feed_mixer.outlet, destination=m.fs.hp_pump.inlet)
        m.fs.hp_pump_to_RO = Arc(source=m.fs.hp_pump.outlet, destination=m.fs.RO.inlet)
        m.fs.RO_brine_to_splitter = Arc(
            source=m.fs.RO.retentate, destination=m.fs.concentrate_splitter.inlet
        )
        m.fs.concentrate_to_mixer = Arc(source=m.fs.concentrate_splitter.recirculated_concentrate, destination=m.fs.feed_mixer.recirculated_concentrate)
        m.fs.concentrate_to_waste = Arc(source=m.fs.concentrate_splitter.waste_brine, destination=m.fs.waste_brine.inlet)
 
    else:
        # this case should be caught in the previous conditional
        erd_type_not_found(erd_config)

    m.fs.ro_permeate_to_translator = Arc(
        source=m.fs.RO.permeate, destination=m.fs.ro_to_mcas_translator.inlet
    )
    # TODO: UV conditionals to determine RO permeate connections
    if uvdimension == uvdimension.none:
        m.fs.translator_to_product = Arc(
            source=m.fs.ro_to_mcas_translator.outlet,
            destination=m.fs.product_water.inlet,
        )
    elif uvdimension == uvdimension.zo:
        m.fs.translator_to_uv = Arc(
            source=m.fs.ro_to_mcas_translator.outlet, destination=m.fs.uv.inlet
        )
        m.fs.uv_to_product = Arc(
            source=m.fs.uv.treated, destination=m.fs.product_water.inlet
        )
    elif uvdimension == uvdimension.zero_d:
        m.fs.translator_to_uv = Arc(
            source=m.fs.ro_to_mcas_translator.outlet, destination=m.fs.uv.inlet
        )
        m.fs.uv_to_product = Arc(
            source=m.fs.uv.outlet, destination=m.fs.product_water.inlet
        )
    else:
        raise NotImplementedError("This config isn't available.")
    # Apply connections
    TransformationFactory("network.expand_arcs").apply_to(m)

    if has_measured_vars:
        touch_measurable_vars(m)
    # scaling
    # set default property values
    m.fs.ro_props.set_default_scaling(
        "flow_mass_phase_comp", 1e1, index=("Liq", "H2O")
    )
    m.fs.ro_props.set_default_scaling(
        "flow_mass_phase_comp", 1e4, index=("Liq", m.fs.ro_ion)
    )
    # if m.fs.uv is not None:
    #     iscale.set_scaling_factor(m.fs.uv.control_volume.properties_in[0].flow_mass_phase_comp['Liq', 'tss'], 1e3)
    # set unit model values
    # iscale.set_scaling_factor(m.fs.hp_pump.control_volume.work, 1e-5)
    # iscale.set_scaling_factor(m.fs.RO.area, 1e-4)
    # calculate and propagate scaling factors
    iscale.calculate_scaling_factors(m)
    return m


def set_operating_conditions(m):
    # ---specifications---
    # feed
    flow_vol = 3.703e-4 #* pyunits.gallon / pyunits.min
    conc_mass_tds = 1.402 * pyunits.kg / pyunits.m**3
    conc_mass_tss = 0.03 * pyunits.kg / pyunits.m**3
    temperature = 298 * pyunits.K
    pressure = 1e5 * pyunits.Pa
    mf_and_cf_pump_discharge_pressures= 2.55e5* pyunits.Pa
    hp_discharge_pressure = 12.76e5* pyunits.Pa
    pressure_atm =101325* pyunits.Pa
    m.fs.feed.temperature[0].fix(temperature)
    m.fs.feed.pressure[0].fix(pressure)
    m.fs.feed.properties.calculate_state(
        var_args={
            ("conc_mass_phase_comp", ("Liq", m.fs.ro_ion)): conc_mass_tds,
            ("conc_mass_phase_comp", ("Liq", "tss")): conc_mass_tss,
            ("flow_vol_phase", "Liq"): flow_vol,
        },
        hold_state=True,
    )
    # TODO: add scaling at least for feed props before solve
    # solve(m.fs.feed, checkpoint="solve feed block")

    # ---pretreatment---
    # TODO: add option to eliminate underlying fixed energy calculations and shift to pump
    # mf pump
    m.fs.mf_pump.efficiency_pump.fix(0.8)
    mf_dP=mf_and_cf_pump_discharge_pressures-pressure
    m.fs.mf_pump.control_volume.deltaP[0].fix(mf_dP)
    # m.fs.mf_pump.control_volume.properties_out[0].pressure.fix(2e5)

    # microfiltration
    m.db.get_unit_operation_parameters("microfiltration")
    m.fs.mf.load_parameters_from_database(use_default_removal=True)
    # Negate energy accounting for MF since modeling pump separately
    m.fs.mf.energy_electric_flow_vol_inlet.fix(0)
    
    # cf pump
    m.fs.cf_pump.efficiency_pump.fix(0.8)
    # m.fs.cf_pump.control_volume.properties_out[0].pressure.fix(2e5)
    m.fs.cf_pump.control_volume.deltaP[0].fix(0.0)


    # cartridge filtration
    m.db.get_unit_operation_parameters("cartridge_filtration")
    m.fs.cf.load_parameters_from_database(use_default_removal=True)
    # Negate energy accounting for CF since modeling pump separately
    m.fs.cf.energy_electric_flow_vol_inlet.fix(0)

    # MCAS to RO prop translator
    m.fs.mcas_to_ro_translator.outlet.pressure[0].fix(mf_and_cf_pump_discharge_pressures)
    m.fs.mcas_to_ro_translator.outlet.temperature[0].fix(temperature)

    # hp pump
    m.fs.hp_pump.efficiency_pump.fix(0.8)
    hp_dP = hp_discharge_pressure - pressure_atm

    # m.fs.hp_pump.control_volume.deltaP[0].fix(hp_dP+2e5*pyunits.Pa)
    # m.fs.hp_pump.control_volume.properties_out[0].pressure = hp_discharge_pressure
    
    @m.fs.hp_pump.control_volume.Constraint([0])
    def eq_fix_hp_pressure(blk, t):
        return blk.properties_out[t].pressure == hp_discharge_pressure


    
    # RO
    #TODO: update A, B, channel height, porosity, area, width (others?)
    m.fs.RO.A_comp.fix(1.2e-11)  # membrane water permeability coefficient [m/s-Pa]
    m.fs.RO.B_comp.fix(6.7e-8)  # membrane salt permeability coefficient [m/s]
    if hasattr(m.fs.RO.feed_side, 'channel_height'):
        m.fs.RO.feed_side.channel_height.fix(34e-3*pyunits.inch)  # channel height in membrane stage [m]
    if hasattr(m.fs.RO.feed_side,'spacer_porosity'):
        m.fs.RO.feed_side.spacer_porosity.fix(0.75)  # spacer porosity in membrane stage [-]
    # m.fs.RO.width.fix(1000)  # stage width [m]
    m.fs.RO.area.fix(7.2*4)
    m.fs.RO.deltaP.fix(-0.75e5)
    # m.fs.RO.length(4*(1.016-2*26.7e-3))
    @m.fs.RO.Constraint([0])
    def eq_fix_RO_perm_pressure(blk, t):
        return blk.permeate.pressure[t] == pressure_atm
    
    # m.fs.RO.permeate.pressure[0].fix(101325)  # atmospheric pressure [Pa]

    # Concentrate Separator
    m.fs.concentrate_splitter.split_fraction[0, "recirculated_concentrate"].fix(1e-8)

    # RO props to MCAS translator
    m.fs.ro_to_mcas_translator.outlet.pressure[0].fix(pressure_atm)
    m.fs.ro_to_mcas_translator.outlet.temperature[0].fix(temperature)

    if hasattr(m.fs.uv, "_tech_type"):
        m.fs.uv.load_parameters_from_database(use_default_removal=True)
    elif isinstance(m.fs.uv, UnitModelBlockData):
        m.fs.uv.uv_intensity.fix(1 * pyunits.mW / pyunits.cm**2)
        m.fs.uv.exposure_time.fix(500 * pyunits.s)
        m.fs.uv.inactivation_rate["Liq", "tss"].fix(2.3 * pyunits.cm**2 / pyunits.J)
        # m.fs.uv.outlet.temperature[0].fix(temperature)
        # @m.fs.uv.Constraint([0])
        # def eq_fix_outlet_temp(blk, t):
        #     return blk.control_volume.properties_out[t].temperature == temperature

        m.fs.uv.electrical_efficiency_phase_comp[0, "Liq", "tss"].fix(
            0.1 * pyunits.kWh / pyunits.m**3
        )
        m.fs.uv.lamp_efficiency.fix(0.8)
        if m.fs.uv.config.has_aop:
            m.fs.uv.second_order_reaction_rate_constant["Liq", "tss"].fix(
                3.3e8 * pyunits.M**-1 * pyunits.s**-1
            )
            # TODO: hydrogen_peroxide_conc should be generalized to oxidant_conc or something of the like
            m.fs.uv.hydrogen_peroxide_conc.fix(5.05e-13 * pyunits.M)


def initialize_system(m):

    m.fs.feed.initialize()
    propagate_state(m.fs.feed_to_mf_pump)
    m.fs.mf_pump.initialize()
    propagate_state(m.fs.mf_pump_to_mf)
    m.fs.mf.initialize()

    propagate_state(m.fs.mf_to_cf_pump)
    m.fs.cf_pump.initialize()

    propagate_state(m.fs.cf_pump_to_cf)
    m.fs.cf.initialize()

    propagate_state(m.fs.cf_to_translator)
    m.fs.mcas_to_ro_translator.initialize()

    propagate_state(m.fs.translator_to_mixer)
    initialize_with_recirculation(m)
    # master_initialize_with_recirculation(m, count=3)
    
def master_initialize_with_recirculation(m, count):
    solved = 0 
    counter = 0
    while not solved:
        try:
            initialize_with_recirculation(m)
            res= solve(m, tee=True, fail_flag=False)
        except:
            pass
        counter = counter + 1
        if counter == count:
            break 
        if check_optimal_termination(res):
            solved = 1
        

def initialize_with_recirculation(m):
        propagate_state(m.fs.concentrate_to_mixer)
        m.fs.feed_mixer.initialize()
        
        propagate_state(m.fs.mixer_to_hp_pump)
        m.fs.hp_pump.initialize()

        propagate_state(m.fs.hp_pump_to_RO)

        # try:
        m.fs.RO.initialize(outlvl=idaeslog.DEBUG)
        # except:
        #     pass
        
        propagate_state(m.fs.RO_brine_to_splitter)
        m.fs.concentrate_splitter.initialize()
        
        propagate_state(m.fs.ro_permeate_to_translator)
        m.fs.ro_to_mcas_translator.initialize()

        if hasattr(m.fs, "translator_to_uv"):
            propagate_state(m.fs.translator_to_uv)
            m.fs.uv.initialize(outlvl=idaeslog.DEBUG)
            propagate_state(m.fs.uv_to_product)
        else:
            propagate_state(m.fs.translator_to_product)


def solve(blk, solver=None, checkpoint=None, tee=False, fail_flag=True):
    if solver is None:
        solver = get_solver()
    results = solver.solve(blk, tee=tee)
    check_solve(results, checkpoint=checkpoint, logger=_log, fail_flag=fail_flag)
    return results

def setup_optimization(m):
    # Concentrate Separator
    m.fs.concentrate_splitter.split_fraction[0, "recirculated_concentrate"].fix(0.2)
    
    master_initialize_with_recirculation(m, 3)
    # res=solve(m,tee=True)
    # assert_optimal_termination(res)

def get_uv_model(m, uv_props, dimension, has_aop):
    if dimension == "zo" and not has_aop:
        return UVZO(
            property_package=uv_props,
            database=m.db,
        )
    if dimension == "zo" and has_aop:
        return UVAOPZO(
            property_package=uv_props,
            database=m.db,
        )
    elif dimension == "0d" and not has_aop:
        return Ultraviolet0D(property_package=uv_props, target_species=["tss"])
    elif dimension == "0d" and has_aop:
        return Ultraviolet0D(
            property_package=uv_props, target_species=["tss"], has_aop=True
        )
    elif dimension == "none":
        return None
    else:
        if dimension not in uvdimension:
            raise ConfigurationError(
                f"Either 'none', 'zo', or '0d' should be provided for uvdimension instead of {dimension}"
            )
        elif has_aop is not bool:
            raise ConfigurationError(f"has_aop should be True or False, not {has_aop}")
        else:
            raise ConfigurationError("There's something wrong...but what?")


def get_ro_props(ro_props):
    if ro_props.lower() == "nacl":
        return NaClParameterBlock(), "NaCl"
    elif ro_props.lower() == "seawater":
        return SeawaterParameterBlock(), "TDS"
    else:
        raise ConfigurationError(
            f"Either 'nacl' or 'seawater' should be provided for ro_props. Instead, {ro_props} was provided."
        )


def get_ro_model(dimension, ro_props):
    if dimension == rodimension.zero_d:
        return ReverseOsmosis0D(
            property_package=ro_props,
            has_pressure_change=True,
            pressure_change_type=PressureChangeType.fixed_per_stage,
            mass_transfer_coefficient=MassTransferCoefficient.none,
            concentration_polarization_type=ConcentrationPolarizationType.none,
            module_type=ModuleType.spiral_wound,
            has_full_reporting=True,
        )
    elif dimension == rodimension.one_d:
        return ReverseOsmosis1D(
            property_package=ro_props,
            has_pressure_change=True,
            pressure_change_type=PressureChangeType.fixed_per_stage,
            mass_transfer_coefficient=MassTransferCoefficient.none,
            concentration_polarization_type=ConcentrationPolarizationType.none,
            module_type=ModuleType.spiral_wound,
            has_full_reporting=True,
        )
    else:
        raise ConfigurationError(
            f"Either '0d' or '1d' should be provided for dimension instead of {dimension}"
        )


def display_results(m):
    m.fs.report()
    # call report() on every unit in the flowsheet:
    for block in m.fs.component_objects(Block, descend_into=True):
        if isinstance(block, UnitModelBlockData):
            block.report()


def add_costing(m):
    # TODO: add costing
    # process costing and add system level metrics
    # m.fs.costing.cost_process()
    # m.fs.costing.add_annual_water_production(m.fs.product.properties[0].flow_vol)
    # m.fs.costing.add_LCOW(m.fs.product.properties[0].flow_vol)
    # m.fs.costing.add_specific_energy_consumption(m.fs.product.properties[0].flow_vol)
    # m.fs.costing.add_specific_electrical_carbon_intensity(
    #     m.fs.product.properties[0].flow_vol
    # )
    pass

def touch_measurable_vars(m, target_vars=None):
    if target_vars is None:
        target_vars = ["flow_vol_phase", "mass_frac_phase_comp", "pressure"]
    if target_vars is not None and not isinstance(target_vars, list):
        raise ValueError("target_vars argument must be a list of strings representing valid property names.")
        
    stream_state_list=list_stateblocks_on_active_ports(m)
    for sb in stream_state_list:
        for var in target_vars:
            getattr(sb,var)
    
    return

def list_stateblocks_on_active_ports(blk):
    sb_list=[]
    active_port_list=[]
    for p in blk.fs.component_objects(Port, descend_into=True):
        if p.active:
            active_port_list.append(p.name)
            sb_temp = _get_state_from_port(p,0)
            sb_list.append(sb_temp)
    return sb_list

# def fix_by_constraint(variable, index=None):
#     @variable.Constraint(index)
#     def eq_fix_value(blk, j):
#         return (
#             blk.properties_in[0].flow_mass_phase_comp["Liq", j]
#             == blk.properties_out[0].flow_mass_phase_comp["Liq", j]
#         )

def get_metadata(object_list):
    """
    Get doc strings for variables or constraints
    object_list: var list or constraint list from pyomo NLP

    Return
    dict with object names to doc string descriptions
    """
    obj_docs={}

    for o_string in object_list:
        
        # print(o_string)
        if ']' in o_string:
            for oi in range(-1,-len(o_string),-1):
                # print(o_string[oi])
                if o_string[oi] == '[':
                    break
            new_o_string = o_string[:oi]        
        else:
            new_o_string = o_string        


        print(new_o_string)

        obj = eval('m.'+new_o_string)
        print(obj.doc)

        obj_dict = {'m.'+new_o_string:obj.doc}
        obj_docs.update(obj_dict)

    return obj_docs

def write_metadata(object_list, object_descriptor=None):
    if object_descriptor is None:
        object_descriptor = "Variable"
    obj_docs = get_metadata(object_list)
    import pandas as pd
    data_dict = {f'{object_descriptor}': [k for k in obj_docs.keys()], f'{object_descriptor} Descriptions': [v for v in obj_docs.values()]}

    df = pd.DataFrame(data_dict)
    df.to_csv(f'{object_descriptor.lower()}_descriptions.csv', index=False)

if __name__ == "__main__":
    diagnostics_flag = True
    has_touched_vars=True
    has_sub_jac=False
    get_jacobian=True
    RO_dim="0d"
    if diagnostics_flag is True:
        m, results, dt = main(
            ro_props="seawater",
            ro_dimension=RO_dim,
            erd_config=ERDtype.no_ERD,
            uvdimension=uvdimension.zero_d,
            has_aop=True,
            has_measured_vars=has_touched_vars,
            diagnostics_active=diagnostics_flag,
        )
    elif diagnostics_flag is False:
        m, results = main(
            ro_props="seawater",
            ro_dimension=RO_dim,
            erd_config=ERDtype.no_ERD,
            uvdimension=uvdimension.zero_d,
            has_aop=True,
            has_measured_vars=has_touched_vars,
            diagnostics_active=diagnostics_flag,
        )
    else:
        raise TypeError("diagnostics_flag should be set to True or False.")

    if get_jacobian:
        from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP

        m.fs.mf_pump.control_volume
        m.fs.obj = Objective(expr=0)
        nlp = PyomoNLP(m)
        jac = nlp.evaluate_jacobian()
        var_list = nlp.primals_names()
        con_list = nlp.constraint_names()
        jac_array = jac.toarray()
        import pandas as pd
        df = pd.DataFrame(jac_array, index = con_list, columns=var_list)
        if has_touched_vars:
            msg= "_with_touched_vars"
        else:
            msg = ""    
        df.to_csv(f'ro{RO_dim}_pilot_FULL_jacobian{msg}.csv')

        #%% Write csvs for variable and constraint metadata (doc strings)
        write_metadata(con_list, "Constraint")
        # var_docs={}
        # con_docs={}
        # for v_string in var_list:
            
        #     # print(v_string)
        #     if ']' in v_string:
        #         for vi in range(-1,-len(v_string),-1):
        #             # print(v_string[vi])
        #             if v_string[vi] == '[':
        #                 break
        #         new_v_string = v_string[:vi]        
        #     else:
        #         new_v_string = v_string        


        #     print(new_v_string)

        #     var = eval('m.'+new_v_string)
        #     print(var.doc)


        #     var_dict = {'m.'+new_v_string:var.doc}
        #     var_docs.update(var_dict)



        if has_sub_jac:

            # get submatrix
            pyomo_var_list = nlp.get_pyomo_variables()
            pyomo_con_list = nlp.get_pyomo_constraints()

            def get_measurable_vars(m, target_vars=None):
                if target_vars is None:
                    target_vars = ["flow_vol_phase", "mass_frac_phase_comp", "pressure"]
                if target_vars is not None and not isinstance(target_vars, list):
                    raise ValueError("target_vars argument must be a list of strings representing valid property names.")

                # Commenting out the code below and trying another approach to access all Stateblocks on active ports:                    
                # arc_stream_pairs = arcs_to_stream_dict(m)
                # stream_state_pairs = stream_states_dict(arc_stream_pairs)
                sub_var_list = []
                # for sb in stream_state_pairs.values():
                stream_state_list = list_stateblocks_on_active_ports(m)
                for sb in stream_state_list:
                    for var in target_vars:
                        sub_var_list.append(getattr(sb,var))
                
                return sub_var_list
            
            subvarlist=get_measurable_vars(m)
            sub_jac = nlp.extract_submatrix_jacobian(subvarlist, pyomo_con_list)
            sub_jac_array = sub_jac.toarray()
            sub_var_names = []
            for v in subvarlist:
                if not v.is_indexed():
                    sub_var_names.append(v.name)
                else:
                    for val in v.values():
                        sub_var_names.append(val.name)
            
            subdf = pd.DataFrame(sub_jac_array, index = con_list, columns=sub_var_names)
            subdf.to_csv(f'ro{RO_dim}_pilot_SUB_jacobian{msg}.csv')

        




