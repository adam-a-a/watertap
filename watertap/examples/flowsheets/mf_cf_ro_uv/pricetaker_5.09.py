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
    SolverFactory
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
    new_test= [0.12584, 0.17709,
        0.17709, 0.17709, 0.17709, 0.17709,
        0.12584,] 

    m.append_lmp_data(lmp_data=test_24hr_LMP)
 
    

    ro_props = "Seawater"
    diagnostics_flag = False
    has_touched_vars=True
    has_sub_jac=False
    get_jacobian=False
    RO_dim="1d"
    uv_dim= "none"
    ERD_conf= "no_ERD"
    has_aop=False

    # build multiperiod model off of initialized flowsheet
    m.build_multiperiod_model(flowsheet_func=uci.build_flowsheet,
                            #   flowsheet_options= {"ro_props": ro_props,
                                                #   "ro_dimension": RO_dim,
                                                #   "erd_config": "no_ERD",
                                                #   "uvdimension": uv_dim,
                                                #   "has_aop": has_aop,
                                                #   "has_measured_vars": has_touched_vars
                                                #   }
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
    m.total_energy_cost = Expression(expr=sum(m.period[:, :].fs.energy_cost))
    m.total_water_production = Expression(expr=sum(m.period[1,t].fs.product_water.properties[0].flow_vol_phase["Liq"]*3600 for t in range(1,m.horizon_length+1)))
    m.target_water_production = Constraint(expr=m.total_water_production>=4)
    
    m.obj = Objective(
        expr=m.total_energy_cost,
        sense="minimize",
    )

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
        ipopt.options["max_iter"] = 500
        res = ipopt.solve(m, tee=True)
