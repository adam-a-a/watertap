###############################################################################
# ProteusLib Copyright (c) 2021, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National
# Laboratory, National Renewable Energy Laboratory, and National Energy
# Technology Laboratory (subject to receipt of any required approvals from
# the U.S. Dept. of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/nawi-hub/proteuslib/"
#
###############################################################################
from pyomo.environ import ConcreteModel, TransformationFactory,\
                          Expression, SolverFactory, Constraint, \
                          Objective, Var, NonNegativeReals
from pyomo.network import Arc, SequentialDecomposition
from pyomo.environ import units as pyunits
from idaes.core import FlowsheetBlock
from idaes.generic_models.unit_models import Mixer, Separator
from idaes.generic_models.unit_models.separator import SplittingType, EnergySplittingType
from idaes.generic_models.unit_models.mixer import MixingType
from idaes.core.util.model_statistics import degrees_of_freedom, unfixed_variables_in_activated_equalities_set, activated_equalities_set
import idaes.core.util.scaling as iscale
import proteuslib.property_models.NaCl_CaSO4_prop_pack as nacl_caso4
from proteuslib.util.initialization import check_solve, check_dof
import copy

# from pyomo.environ import (TerminationCondition,
#                            value,
#                            Constraint,
#                            Expression,
#                            Objective,
#                            Param,
#                            TransformationFactory,
#                            units as pyunits)
# import pyomo.util.infeasible as infeas
from idaes.core.util import get_solver
# from idaes.core.util.model_statistics import degrees_of_freedom
# from idaes.core.util.initialization import (solve_indexed_blocks,
#                                             propagate_state,
#                                             fix_state_vars,
#                                             revert_state_vars)
# from idaes.generic_models.unit_models import Mixer, Separator, Product, Feed
# from idaes.generic_models.unit_models.mixer import MomentumMixingType
# import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
#
# import proteuslib.property_models.NaCl_prop_pack as nacl
# from proteuslib.unit_models.reverse_osmosis_0D import (ReverseOsmosis0D,
#                                                        ConcentrationPolarizationType,
#                                                        MassTransferCoefficient,
#                                                        PressureChangeType)
# from proteuslib.unit_models.pressure_exchanger import PressureExchanger
# from proteuslib.unit_models.pump_isothermal import Pump
# import proteuslib.flowsheets.RO_with_energy_recovery.financials as financials

# Set up logger
_log = idaeslog.getLogger(__name__)

# set up solver
solver = get_solver(options={'nlp_scaling_method': 'user-scaling'})

def build_flowsheet():
    ### Build the flowsheet model object ###
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.properties = nacl_caso4.PropParameterBlock()

    ### Add the mixer block ###
    m.fs.pre_mixer = Mixer(default={
            "property_package": m.fs.properties,
            "inlet_list": ['recycle', 'bypass'],
            "energy_mixing_type": MixingType.none
            })
    ### Add the mixer block ###
    m.fs.post_mixer = Mixer(default={
            "property_package": m.fs.properties,
            "inlet_list": ['pretreatment', 'bypass'],
            "energy_mixing_type": MixingType.none
            })

    ### Add the splitter block ###
    m.fs.bypass = Separator(default={
            "property_package": m.fs.properties,
            "outlet_list": ['pretreatment', 'bypass'],
            "ideal_separation": False,
            "split_basis": SplittingType.totalFlow,
            "energy_split_basis": EnergySplittingType.equal_temperature
            })

    ### Add the separator block ###
    m.fs.NF = Separator(default={
            "property_package": m.fs.properties,
            "outlet_list": ['retentate', 'permeate'],
            "ideal_separation": False,
            "split_basis": SplittingType.componentFlow,
            "energy_split_basis": EnergySplittingType.equal_temperature
            })

    ### Add the splitter block ###
    m.fs.split = Separator(default={
            "property_package": m.fs.properties,
            "outlet_list": ['disposal', 'recycle'],
            "ideal_separation": False,
            "split_basis": SplittingType.totalFlow,
            "energy_split_basis": EnergySplittingType.equal_temperature
            })

    ### Add connections ###
    #
    #            ------------------------------------|
    #            |                                   v
    #  feed -> bypass  -> pre_mixer ----> NF --> post_mixer --> to RO model
    #                         ^           |
    #                         |           V
    #                      recycle ---  split
    #                                     |
    #                                     V
    #                                  disposal
    #
    m.fs.S1 = Arc(source=m.fs.bypass.pretreatment, destination=m.fs.pre_mixer.bypass)
    m.fs.S2 = Arc(source=m.fs.bypass.bypass,       destination=m.fs.post_mixer.bypass)
    m.fs.S3 = Arc(source=m.fs.pre_mixer.outlet,    destination=m.fs.NF.inlet)
    m.fs.S4 = Arc(source=m.fs.NF.retentate,        destination=m.fs.split.inlet)
    m.fs.S5 = Arc(source=m.fs.NF.permeate,         destination=m.fs.post_mixer.pretreatment)
    m.fs.S6 = Arc(source=m.fs.split.recycle,       destination=m.fs.pre_mixer.recycle)
    TransformationFactory("network.expand_arcs").apply_to(m)

    # Create a water recover vars
    m.fs.RO_water_recovery = Var(
            initialize=0.5,
            bounds=(0, 1),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc='RO Water Recovery')

    m.fs.system_water_recovery = Var(
            initialize=0.5,
            bounds=(0, 1),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc='System Water Recovery')

    ### Add water recovery constraints ###
    m.fs.eq_system_recovery = Constraint(
        expr=m.fs.system_water_recovery==(m.fs.post_mixer.outlet.flow_mass_phase_comp[0, 'Liq', 'H2O']*m.fs.RO_water_recovery)
                                         /m.fs.bypass.inlet.flow_mass_phase_comp[0, 'Liq', 'H2O'])

    ### Add QOI ###
    # m.fs.system_water_recovery = Expression(
    # expr=(sum(m.fs.NF.permeate.flow_mass_phase_comp[0, 'Liq', j] for j in ['H2O'])
    #       / sum(m.fs.mixer.feed.flow_mass_phase_comp[0, 'Liq', j] for j in ['H2O'])))

    return m

def set_dof(m):
    ### initialize feed mixer ###
    mass_flow = 1
    x_NaCl = 0.005
    x_CaSO4 = 0.0015
    room_temp = 273.15 + 25
    m.fs.bypass.inlet.pressure.fix(101325)
    m.fs.bypass.inlet.temperature.fix(room_temp)
    m.fs.bypass.inlet.flow_mass_phase_comp[0.0, 'Liq', 'NaCl'].fix(x_NaCl * mass_flow)
    m.fs.bypass.inlet.flow_mass_phase_comp[0.0, 'Liq', 'CaSO4'].fix(x_CaSO4 * mass_flow)
    # m.fs.bypass.inlet.flow_mass_phase_comp[0.0, 'Liq', 'H2O'].fix((1 - x_NaCl - x_CaSO4) * mass_flow)

    m.fs.feed_H2O = Constraint(
        expr=m.fs.bypass.inlet.flow_mass_phase_comp[0.0, 'Liq', 'H2O'] ==
             (
             1.0
             - m.fs.bypass.inlet.flow_mass_phase_comp[0.0, 'Liq', 'NaCl']
             - m.fs.bypass.inlet.flow_mass_phase_comp[0.0, 'Liq', 'CaSO4']
             )
             * mass_flow)


    ### Fix mixer temp out ###
    m.fs.pre_mixer.outlet.temperature.fix(room_temp)
    m.fs.post_mixer.outlet.temperature.fix(room_temp)

    ### set NF recover ratio ###
    m.fs.NF.split_fraction[0, "permeate", "H2O"].fix(0.85)
    m.fs.NF.split_fraction[0, "permeate", "NaCl"].fix(0.8)
    m.fs.NF.split_fraction[0, "permeate", "CaSO4"].fix(0.2) # Parameter of Interest

    ### set waste ratio
    m.fs.split.split_fraction[0, "recycle"].fix(0.8)

    ### set bypass ratio
    m.fs.bypass.split_fraction[0, "bypass"].fix(0.5)

    ### fix RO recovery ###
    m.fs.RO_water_recovery.fix(0.6)

    m.fs.properties.set_default_scaling('flow_mass_phase_comp', 1/mass_flow, index=('Liq', 'H2O'))
    m.fs.properties.set_default_scaling('flow_mass_phase_comp', 1/mass_flow * 1e2, index=('Liq', 'NaCl'))
    m.fs.properties.set_default_scaling('flow_mass_phase_comp', 1/mass_flow * 1e4, index=('Liq', 'CaSO4'))
    iscale.calculate_scaling_factors(m)

    check_dof(m, fail_flag=True)

def simulate(m):
    # m.fs.pre_mixer.initialize(hold_state=True)
    # m.fs.post_mixer.initialize((hold_state=True)
    # m.fs.bypass.initialize()
    # m.fs.NF.initialize()
    # m.fs.split.initialize()

    # ---solving---
    results = solver.solve(m, tee=False)
    check_solve(results, checkpoint="Simulation", fail_flag=True)

def minimize_pretreatment(original_m, system_water_recovery):
    ### Copy over the model ###
    m = copy.deepcopy(original_m)

    ### Unfix the split disposal ###
    m.fs.split.split_fraction[0,  "recycle"].unfix()
    m.fs.split.split_fraction[0,  "recycle"].setlb(0.0)
    m.fs.split.split_fraction[0,  "recycle"].setub(1.0)
    m.fs.bypass.split_fraction[0, "bypass"].unfix()
    m.fs.bypass.split_fraction[0, "bypass"].setlb(0.0)
    m.fs.bypass.split_fraction[0, "bypass"].setub(1.0)
    m.fs.RO_water_recovery.unfix()
    m.fs.system_water_recovery.fix(system_water_recovery)

    ### Create a constraint to prevent scaling ###
    scaling_mass_fac = 0.002

    m.fs.eq_no_scaling = Constraint(
        expr=m.fs.post_mixer.outlet.flow_mass_phase_comp[0, 'Liq', 'CaSO4']/(1-m.fs.RO_water_recovery) <= scaling_mass_fac)

    ### Create objective for
    # m.fs.objective = Objective(expr=m.fs.split.split_fraction[0, "recycle"])
    m.fs.objective = Objective(expr=m.fs.NF.inlet.flow_mass_phase_comp[0, 'Liq', 'H2O'])
    # m.fs.objective = Objective(expr=-m.fs.system_water_recovery)

    # ---solving---
    results = solver.solve(m, tee=False)
    check_solve(results, checkpoint="Solve for minimal pretreatment")

    return m

def maximize_system_recovery(original_m, RO_water_recovery):
    ### Copy over the model ###
    m = copy.deepcopy(original_m)

    ### Unfix the split disposal ###
    m.fs.split.split_fraction[0,  "recycle"].unfix()
    m.fs.split.split_fraction[0,  "recycle"].setlb(0.0)
    m.fs.split.split_fraction[0,  "recycle"].setub(1.0)
    m.fs.bypass.split_fraction[0, "bypass"].unfix()
    m.fs.bypass.split_fraction[0, "bypass"].setlb(0.0)
    m.fs.bypass.split_fraction[0, "bypass"].setub(1.0)
    m.fs.RO_water_recovery.fix(RO_water_recovery)
    m.fs.system_water_recovery.unfix()

    ### Create a constraint to prevent scaling ###
    scaling_mass_fac = 0.002
    m.fs.eq_no_scaling = Constraint(
        expr=m.fs.post_mixer.outlet.flow_mass_phase_comp[0, 'Liq', 'CaSO4']/(1-m.fs.RO_water_recovery)<=scaling_mass_fac)

    ### Create objective for
    m.fs.objective = Objective(expr=-m.fs.system_water_recovery)

    # ---solving---
    results = solver.solve(m, tee=False)
    check_solve(results, checkpoint="Solve for max system recovery")
    return m

def main():
    m = build_flowsheet()
    set_dof(m)
    simulate(m)

    print("-------------------------------------")
    print("|       Minimize Pretreatment       |")
    print("-------------------------------------")
    m1 = minimize_pretreatment(m,0.99)
    min_recycle_fraction, RO_water_recovery, System_water_recovery, NF_inflow, bypass_fraction = [
        m1.fs.split.split_fraction[0, "recycle"].value,
        m1.fs.RO_water_recovery.value,
        m1.fs.system_water_recovery.value,
        m1.fs.NF.inlet.flow_mass_phase_comp[0, 'Liq', 'H2O'].value,
        m1.fs.bypass.split_fraction[0, "bypass"].value,
    ]
    print("recycle fraction:      {:>5.2f} %".format(min_recycle_fraction*100))
    print("bypassed fraction:     {:>5.2f} %".format(bypass_fraction*100))
    print("RO water recovery:     {:>5.2f} %".format(RO_water_recovery*100))
    print("System water recovery: {:>5.2f} %".format(System_water_recovery*100))
    print("NF inflow:             {:>5.2f} kg/s".format(NF_inflow))
    print("-------------------------------------")
    print()
    print()
    print()
    print("-------------------------------------")
    print("|     Maximize System Recovery      |")
    print("-------------------------------------")
    m2 = maximize_system_recovery(m,0.70)
    min_recycle_fraction, RO_water_recovery, System_water_recovery, NF_inflow, bypass_fraction = [
        m2.fs.split.split_fraction[0, "recycle"].value,
        m2.fs.RO_water_recovery.value,
        m2.fs.system_water_recovery.value,
        m2.fs.NF.inlet.flow_mass_phase_comp[0, 'Liq', 'H2O'].value,
        m2.fs.bypass.split_fraction[0, "bypass"].value,
    ]
    print("recycle fraction:      {:>5.2f} %".format(min_recycle_fraction*100))
    print("bypassed fraction:     {:>5.2f} %".format(bypass_fraction*100))
    print("RO water recovery:     {:>5.2f} %".format(RO_water_recovery*100))
    print("System water recovery: {:>5.2f} %".format(System_water_recovery*100))
    print("NF inflow:             {:>5.2f} kg/s".format(NF_inflow))
    print("-------------------------------------")

if __name__ == "__main__":
    main()