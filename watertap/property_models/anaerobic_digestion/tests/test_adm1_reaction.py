#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES), and is copyright (c) 2018-2021
# by the software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia University
# Research Corporation, et al.  All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
#################################################################################
"""
Tests for ADM1 reaction package.
Authors: Adam Atia
"""
import pytest

from pyomo.environ import (
    check_optimal_termination,
    ConcreteModel,
    Constraint,
    units,
    value,
    Var,
)
from pyomo.util.check_units import assert_units_consistent

from idaes.core import FlowsheetBlock
from idaes.models.unit_models import CSTR
from idaes.core import MaterialFlowBasis
from idaes.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom

from watertap.property_models.anaerobic_digestion.adm1_properties import (
    ADM1ParameterBlock,
)
from watertap.property_models.anaerobic_digestion.adm1_reactions import (
    ADM1ReactionParameterBlock,
    ADM1ReactionBlock,
)

# -----------------------------------------------------------------------------
# Get default solver for testing
solver = get_solver()


class TestParamBlock(object):
    @pytest.fixture(scope="class")
    def model(self):
        model = ConcreteModel()
        model.pparams = ADM1ParameterBlock()
        model.rparams = ADM1ReactionParameterBlock(
            default={"property_package": model.pparams}
        )

        return model

    @pytest.mark.unit
    def test_build(self, model):
        assert model.rparams.reaction_block_class is ADM1ReactionBlock

        assert len(model.rparams.rate_reaction_idx) == 19
        for i in model.rparams.rate_reaction_idx:
            assert i in [
                "R1",
                "R2",
                "R3",
                "R4",
                "R5",
                "R6",
                "R7",
                "R8",
                "R9",
                "R10",
                "R11",
                "R12",
                "R13",
                "R14",
                "R15",
                "R16",
                "R17",
                "R18",
                "R19",
            ]

        assert len(model.rparams.rate_reaction_stoichiometry) == 19 * 25
        for i in model.rparams.rate_reaction_stoichiometry:
            assert i[0] in [
                "R1",
                "R2",
                "R3",
                "R4",
                "R5",
                "R6",
                "R7",
                "R8",
                "R9",
                "R10",
                "R11",
                "R12",
                "R13",
                "R14",
                "R15",
                "R16",
                "R17",
                "R18",
                "R19",
            ]
            assert i[1] == "Liq"
            assert i[2] in [
                "H2O",
                "S_su",
                "S_aa",
                "S_fa",
                "S_va",
                "S_bu",
                "S_pro",
                "S_ac",
                "S_h2",
                "S_ch4",
                "S_IC",
                "S_IN",
                "S_I",
                "X_c",
                "X_ch",
                "X_pr",
                "X_li",
                "X_su",
                "X_aa",
                "X_fa",
                "X_c4",
                "X_pro",
                "X_ac",
                "X_h2",
                "X_I",
            ]

        assert isinstance(model.rparams.Y_A, Var)
        assert value(model.rparams.Y_A) == 0.24
        assert isinstance(model.rparams.Y_H, Var)
        assert value(model.rparams.Y_H) == 0.67
        assert isinstance(model.rparams.f_p, Var)
        assert value(model.rparams.f_p) == 0.08
        assert isinstance(model.rparams.i_xb, Var)
        assert value(model.rparams.i_xb) == 0.08
        assert isinstance(model.rparams.i_xp, Var)
        assert value(model.rparams.i_xp) == 0.06

        assert isinstance(model.rparams.mu_A, Var)
        assert value(model.rparams.mu_A) == 0.5
        assert isinstance(model.rparams.mu_H, Var)
        assert value(model.rparams.mu_H) == 4
        assert isinstance(model.rparams.K_S, Var)
        assert value(model.rparams.K_S) == 10e-3
        assert isinstance(model.rparams.K_OH, Var)
        assert value(model.rparams.K_OH) == 0.2e-3
        assert isinstance(model.rparams.K_OA, Var)
        assert value(model.rparams.K_OA) == 0.4e-3
        assert isinstance(model.rparams.K_NO, Var)
        assert value(model.rparams.K_NO) == 0.5e-3
        assert isinstance(model.rparams.b_H, Var)
        assert value(model.rparams.b_H) == 0.3
        assert isinstance(model.rparams.b_A, Var)
        assert value(model.rparams.b_A) == 0.05
        assert isinstance(model.rparams.eta_g, Var)
        assert value(model.rparams.eta_g) == 0.8
        assert isinstance(model.rparams.eta_h, Var)
        assert value(model.rparams.eta_h) == 0.8
        assert isinstance(model.rparams.k_h, Var)
        assert value(model.rparams.k_h) == 3
        assert isinstance(model.rparams.K_X, Var)
        assert value(model.rparams.K_X) == 0.1
        assert isinstance(model.rparams.K_NH, Var)
        assert value(model.rparams.K_NH) == 1e-3
        assert isinstance(model.rparams.k_a, Var)
        assert value(model.rparams.k_a) == 50


class TestReactionBlock(object):
    @pytest.fixture(scope="class")
    def model(self):
        model = ConcreteModel()
        model.pparams = ADM1ParameterBlock()
        model.rparams = ADM1ReactionParameterBlock(
            default={"property_package": model.pparams}
        )

        model.props = model.pparams.build_state_block([1])

        model.rxns = model.rparams.build_reaction_block(
            [1], default={"state_block": model.props}
        )

        return model

    @pytest.mark.unit
    def test_build(self, model):
        assert model.rxns[1].conc_mass_comp_ref is model.props[1].conc_mass_comp

    @pytest.mark.unit
    def test_rxn_rate(self, model):
        assert isinstance(model.rxns[1].reaction_rate, Var)
        assert len(model.rxns[1].reaction_rate) == 8
        assert isinstance(model.rxns[1].rate_expression, Constraint)
        assert len(model.rxns[1].rate_expression) == 8

    @pytest.mark.unit
    def test_get_reaction_rate_basis(self, model):
        assert model.rxns[1].get_reaction_rate_basis() == MaterialFlowBasis.mass

    @pytest.mark.component
    def test_initialize(self, model):
        assert model.rxns.initialize() is None

    @pytest.mark.component
    def check_units(self, model):
        assert_units_consistent(model)


class TestReactor:
    @pytest.fixture(scope="class")
    def model(self):
        m = ConcreteModel()

        m.fs = FlowsheetBlock(default={"dynamic": False})

        m.fs.props = ADM1ParameterBlock()
        m.fs.rxn_props = ADM1ReactionParameterBlock(
            default={"property_package": m.fs.props}
        )

        m.fs.R1 = CSTR(
            default={
                "property_package": m.fs.props,
                "reaction_package": m.fs.rxn_props,
            }
        )

        # Feed conditions based on manual mass balance of inlet and recycle streams
        m.fs.R1.inlet.flow_vol.fix(92230 * units.m**3 / units.day)
        m.fs.R1.inlet.temperature.fix(298.15 * units.K)
        m.fs.R1.inlet.pressure.fix(1 * units.atm)
        m.fs.R1.inlet.conc_mass_comp[0, "S_I"].fix(30 * units.g / units.m**3)
        m.fs.R1.inlet.conc_mass_comp[0, "S_S"].fix(14.6112 * units.g / units.m**3)
        m.fs.R1.inlet.conc_mass_comp[0, "X_I"].fix(1149 * units.g / units.m**3)
        m.fs.R1.inlet.conc_mass_comp[0, "X_S"].fix(89.324 * units.g / units.m**3)
        m.fs.R1.inlet.conc_mass_comp[0, "X_BH"].fix(2542.03 * units.g / units.m**3)
        m.fs.R1.inlet.conc_mass_comp[0, "X_BA"].fix(148.6 * units.g / units.m**3)
        m.fs.R1.inlet.conc_mass_comp[0, "X_P"].fix(448 * units.g / units.m**3)
        m.fs.R1.inlet.conc_mass_comp[0, "S_O"].fix(0.3928 * units.g / units.m**3)
        m.fs.R1.inlet.conc_mass_comp[0, "S_NO"].fix(8.32 * units.g / units.m**3)
        m.fs.R1.inlet.conc_mass_comp[0, "S_NH"].fix(7.696 * units.g / units.m**3)
        m.fs.R1.inlet.conc_mass_comp[0, "S_ND"].fix(1.9404 * units.g / units.m**3)
        m.fs.R1.inlet.conc_mass_comp[0, "X_ND"].fix(5.616 * units.g / units.m**3)
        m.fs.R1.inlet.alkalinity.fix(4.704 * units.mol / units.m**3)

        m.fs.R1.volume.fix(1000 * units.m**3)

        return m

    @pytest.mark.component
    def test_dof(self, model):
        assert degrees_of_freedom(model) == 0

    @pytest.mark.component
    def test_unit_consistency(self, model):
        assert_units_consistent(model) == 0

    @pytest.mark.component
    def test_solve(self, model):
        model.fs.R1.initialize()

        solver = get_solver()
        results = solver.solve(model, tee=True)
        assert check_optimal_termination(results)

    @pytest.mark.component
    def test_solution(self, model):
        assert value(model.fs.R1.outlet.flow_vol[0]) == pytest.approx(1.0675, rel=1e-4)
        assert value(model.fs.R1.outlet.temperature[0]) == pytest.approx(
            298.15, rel=1e-4
        )
        assert value(model.fs.R1.outlet.pressure[0]) == pytest.approx(101325, rel=1e-4)
        assert value(model.fs.R1.outlet.conc_mass_comp[0, "S_I"]) == pytest.approx(
            30e-3, rel=1e-5
        )
        assert value(model.fs.R1.outlet.conc_mass_comp[0, "S_S"]) == pytest.approx(
            2.81e-3, rel=1e-2
        )
        assert value(model.fs.R1.outlet.conc_mass_comp[0, "X_I"]) == pytest.approx(
            1149e-3, rel=1e-3
        )
        assert value(model.fs.R1.outlet.conc_mass_comp[0, "X_S"]) == pytest.approx(
            82.1e-3, rel=1e-2
        )
        assert value(model.fs.R1.outlet.conc_mass_comp[0, "X_BH"]) == pytest.approx(
            2552e-3, rel=1e-3
        )
        assert value(model.fs.R1.outlet.conc_mass_comp[0, "X_BA"]) == pytest.approx(
            149e-3, rel=1e-2
        )
        assert value(model.fs.R1.outlet.conc_mass_comp[0, "X_P"]) == pytest.approx(
            449e-3, rel=1e-2
        )
        assert value(model.fs.R1.outlet.conc_mass_comp[0, "S_O"]) == pytest.approx(
            4.3e-6, rel=1e-2
        )
        assert value(model.fs.R1.outlet.conc_mass_comp[0, "S_NO"]) == pytest.approx(
            5.36e-3, rel=1e-2
        )
        assert value(model.fs.R1.outlet.conc_mass_comp[0, "S_NH"]) == pytest.approx(
            7.92e-3, rel=1e-2
        )
        assert value(model.fs.R1.outlet.conc_mass_comp[0, "S_ND"]) == pytest.approx(
            1.22e-3, rel=1e-2
        )
        assert value(model.fs.R1.outlet.conc_mass_comp[0, "X_ND"]) == pytest.approx(
            5.29e-3, rel=1e-2
        )
        assert value(model.fs.R1.outlet.alkalinity[0]) == pytest.approx(
            4.93e-3, rel=1e-2
        )
