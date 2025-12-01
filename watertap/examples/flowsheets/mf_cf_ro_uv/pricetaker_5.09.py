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
from watertap.core.util.model_diagnostics.infeasible import *
from idaes.core.scaling import report_scaling_factors
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
    Expression,
    SolverFactory,
    Param,
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
    UVAOPZO,)

from idaes.apps.grid_integration import PriceTakerModel
import numpy as np
from pyomo.environ import Var, value
import watertap.examples.flowsheets.mf_cf_ro_uv.mf_cf_ro_uv as uci

if __name__== "__main__":
    m = PriceTakerModel()
    test_24hr_LMP = [
        0.12584, 0.12584, 0.12584, 0.12584,
        0.12584, 0.12584, 0.12584, 0.12584,
        0.12584, 0.12584, 0.12584, 0.12584,
        0.12584, 0.12584, 0.12584, 0.17709,
        0.17709, 0.17709, 0.17709, 0.17709,
        0.12584, 0.12584, 0.12584, 0.12584,
    ]
    new_test= [i*100 for i in test_24hr_LMP]
    
    temp_test = [0.12584, 0.17709,0.12584, ]

    lmp_run = temp_test

    m.append_lmp_data(lmp_data=lmp_run)
 
    

    ro_props = "Seawater"
    diagnostics_flag = False
    has_touched_vars=True
    has_sub_jac=False
    get_jacobian=False
    RO_dim="0d"
    uv_dim= "none"
    ERD_conf= "no_ERD"
    has_aop=False

    # build multiperiod model off of initialized flowsheet
    m.build_multiperiod_model(flowsheet_func=uci.build_flowsheet,
                              flowsheet_options= {
                                                #   "ro_props": ro_props,
                                                  "ro_dimension": RO_dim,
                                                #   "erd_config": "no_ERD",
                                                #   "uvdimension": uv_dim,
                                                #   "has_aop": has_aop,
                                                #   "has_measured_vars": has_touched_vars
                                                  }
                                                  )
    

    
    m.total_energy_cost = Expression(expr=sum(m.period[:, :].fs.energy_cost))
    m.total_water_production = Expression(expr=sum(m.period[1,t].fs.product_water.properties[0].flow_vol_phase["Liq"]*3600 for t in range(1,m.horizon_length+1)))
    m.target_water_production = Constraint(expr=m.total_water_production>=0.61*len(lmp_run)) #25.75)
    
    m.obj = Objective(
        expr=m.total_energy_cost,
        sense="minimize",
    )
    
    # fixed vars
    '''
    feed mass flowrates
    feed temperature 
    feed pressure
    UF Feed pump work_mechanical (power consumption)
    UF Feed pump discharge pressure
    MF/UF embedded energy intensity --> fixed to 0 since pump explicitly accounted for
    CF/Booster pump efficiency
    CF/Booster pump deltaP
    CF embedded energy intensity --> fixed to 0 since pump explicitly accounted for
    MCAS to RO translator outlet pressure
    MCAS to RO translator outlet temperature
    HP RO pump efficiency
    HP RO outlet pressure (equality constraint to deactivate)
    RO A Value <-----
    RO B Value <-----
    RO Channel height
    RO spacer porosity
    RO area
    RO delta P  <-----
    RO length
    RO permeate pressure (equality constraint)
    Concentrate splitter split fraction for recirculated brine
    RO to MCAS translator outlet pressure
    RO to MCAS translator outlet temperature
    Ignoring UV vars

    Can also set isobaric=False for MF(UF) and CF if we want to ingest pressure drop data for them 
    '''


    # solver_name = "gurobi"
    # solver_name = "baron"
    # solver_name = "scip"
    # solver_name = "cplex"
    # solver_name = "conopt"
    # solver_name = "minos"
    solver_name = "ipopt"
    mip_gap = 0.1

    if solver_name == "gurobi":
        # solver = utils.get_gurobi_solver_model(m, mip_gap=0.005)
        solver = SolverFactory("gurobi")
        solver.options["MIPGap"] = mip_gap
        solver.solve(m, tee=True)

    elif solver_name == "scip":
        solver = SolverFactory("scip", validate=False)
        solver.solve(m, tee=True)

    elif solver_name in ["baron", "cplex", "conopt", "minos"]:
        solver = SolverFactory("gams")
        solver.solve(
            m,
            tee=True,
            solver=solver_name,
            add_options=[
            f"options optcr={mip_gap};",
            # "option nlp=ipopt;",  # tell GAMS to use IPOPT as the continuous subsolver
            "option threads=24",
            # "option ipopt_linear_solver=ma27;"
            "option reslim=600",
            ],
        )
    else:
        ipopt = get_solver()
        ipopt.options["max_iter"] = 5000
        res = ipopt.solve(m, tee=True)
    #%%
    # plotting stacked area of pump mechanical work and baseline power over time
    import matplotlib.pyplot as plt
    import numpy as np

    h = int(m.horizon_length)
    time = np.arange(1, h + 1)

    # collect pump work_mechanical variables per period
    pumps = {}
    baseline = np.zeros(h)

    for t in range(1, h + 1):
        fs_block = m.period[1, t].fs
        # collect pump work variables and classify into hp_pump, cf_pump, mf_pump
        for v in fs_block.component_data_objects((Var, Expression), descend_into=True):
            name = v.name.lower()
            # consider any "work" variable (work, work_mechanical, control_volume.work, etc.)
            if "control_volume.work" in name:
                parts = v.name.split(".")
                print(parts)
                # lbl = parts[-2] if len(parts) >= 2 else parts[-1]
                lbl = parts[2]
                lbl_low = lbl.lower()
                # map unit label to desired pump keys
                if "hp_pump" in lbl_low or ("hp" in lbl_low and "pump" in lbl_low) or "high_pressure" in lbl_low:
                    key = "hp_pump"
                elif "cf_pump" in lbl_low or ("cf" in lbl_low and "pump" in lbl_low) or "booster" in lbl_low:
                    key = "cf_pump"
                elif "mf_pump" in lbl_low or ("mf" in lbl_low and "pump" in lbl_low) or "feed" in lbl_low or "uf" in lbl_low:
                    key = "mf_pump"
                else:
                    # fallback to unit label if it doesn't match one of the known pumps
                    key = lbl_low

                if key not in pumps:
                    print(key, "NOT IN PUMPS")
                    pumps[key] = np.zeros(h)
                try:
                    print(value(v))
                    pumps[key][t - 1] = float(value(v))
                except Exception:
                    pumps[key][t - 1] = 0.0

        # also capture any base_power_consumption inside the same flowsheet block
        for v in fs_block.component_data_objects(Param, descend_into=True):
            if "base_line_power_consumption" in v.name.lower():
                print(v.name)
                if "baseline_power_consumption" not in pumps:
                    pumps["baseline_power_consumption"] = np.zeros(h)
                try:
                    pumps["baseline_power_consumption"][t - 1] = float(value(v))
                except Exception:
                    pumps["baseline_power_consumption"][t - 1] = 0.0
            # if "base_power_consumption" in name:
            #     try:
            #         baseline[t - 1] = float(value(v))
            #     except Exception:
            #         baseline[t - 1] = 0.0

    # # if baseline not found inside period blocks, search whole model
    # if not baseline.any():
    #     for v in m.component_data_objects(Var, descend_into=True):
    #         if "base_power_consumption" in v.name.lower():
    #             try:
    #                 val = float(value(v))
    #             except Exception:
    #                 val = 0.0
    #             baseline[:] = val
    #             break

    # prepare data for stacking
    pump_names = sorted(pumps.keys())
    stack_vals = [pumps[name] for name in pump_names]
    stack_vals.append(baseline)
    labels = pump_names #+ ["baseline_power_consumption"]

    # plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(time, stack_vals, labels=labels)
    ax.set_xlabel("Time period")
    ax.set_ylabel("Pump Power")
    ax.set_title("Stacked area: pump work_mechanical and baseline_power_consumption over time")
    ax.legend(loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.show()
# %%
