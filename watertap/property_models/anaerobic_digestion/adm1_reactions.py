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
ADM1 reaction package.

References:

[1] D.J. Batstone, J. Keller, I. Angelidaki, S.V. Kalyuzhnyi, S.G. Pavlostathis,
A. Rozzi, W.T.M. Sanders, H. Siegrist, V.A. Vavilin,
The IWA Anaerobic Digestion Model No 1 (ADM1), Water Science and Technology.
45 (2002) 65–73. https://doi.org/10.2166/wst.2002.0292.

[2] C. Rosén, U. Jeppsson, Aspects on ADM1 Implementation within the BSM2 Framework,
Department of Industrial Electrical Engineering and Automation, Lund University, Lund, Sweden. (2006) 1–35.

"""

# Import Pyomo libraries
import pyomo.environ as pyo

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    MaterialFlowBasis,
    ReactionParameterBlock,
    ReactionBlockDataBase,
    ReactionBlockBase,
)
from idaes.core.util.misc import add_object_reference
from idaes.core.util.exceptions import BurntToast
import idaes.logger as idaeslog


# Some more information about this module
__author__ = "Adam Atia"
# Using Andrew Lee's formulation of ASM1 as a template


# Set up logger
_log = idaeslog.getLogger(__name__)


@declare_process_block_class("ADM1ReactionParameterBlock")
class ADM1ReactionParameterData(ReactionParameterBlock):
    """
    Property Parameter Block Class
    """

    def build(self):
        """
        Callable method for Block construction.
        """
        super().build()

        self._reaction_block_class = ADM1ReactionBlock

        # Reaction Index
        # Reaction names based on standard numbering in ADM1 paper
        # R1:  Disintegration
        # R2:  Hydrolysis of carbohydrates
        # R3:  Hydrolysis of proteins
        # R4:  Hydrolysis of lipids
        # R5:  Uptake of sugars
        # R6:  Uptake of amino acids
        # R7:  Uptake of long chain fatty acids (LCFAs)
        # R8:  Uptake of valerate
        # R9:  Uptake of butyrate
        # R10: Uptake of propionate
        # R11: Uptake of acetate
        # R12: Uptake of hydrogen
        # R13: Decay of X_su
        # R14: Decay of X_aa
        # R15: Decay of X_fa
        # R16: Decay of X_c4
        # R17: Decay of X_pro
        # R18: Decay of X_ac
        # R19: Decay of X_h2

        self.rate_reaction_idx = pyo.Set(initialize=[f"R{i}" for i in range(1, 20)])

        # TODO: for all carbon and nitrogen content values with "varies" need to revisited; putting 0s for now
        # Carbon content
        Ci_dict = {
            "S_su": 0.0313,
            "S_aa": 0,  # varies
            "S_fa": 0.0217,
            "S_va": 0.0240,
            "S_bu": 0.0250,
            "S_pro": 0.0268,
            "S_ac": 0.0313,
            "S_h2": 0,
            "S_ch4": 0.0156,
            "S_IC": 1,
            "S_I": 0,  # varies
            "X_c": 0,  # varies
            "X_ch": 0.0313,
            "X_pr": 0,  # varies
            "X_li": 0.0220,
            "X_su": 0.0313,
            "X_aa": 0.0313,
            "X_fa": 0.0313,
            "X_c4": 0.0313,
            "X_pro": 0.0313,
            "X_ac": 0.0313,
            "X_h2": 0.0313,
            "X_I": 0,  # varies
            "S_an": 0,
            "S_cat": 0,
        }
        self.Ci = pyo.Var(
            self.state_ref.solute_set,
            initialize=Ci_dict,
            units=pyo.units.kmol / pyo.units.kg,
            domain=pyo.PositiveReals,
            doc="Carbon content of component, kmole C/kg COD",
        )

        # Nitrogen content
        Ni_dict = {
            "S_su": 0,
            "S_aa": 0,  # varies
            "S_fa": 0,
            "S_va": 0,
            "S_bu": 0,
            "S_pro": 0,
            "S_ac": 0,
            "S_h2": 0,
            "S_ch4": 0,
            "S_IC": 0,
            "S_I": 1,  # varies
            "X_c": 0,  # varies
            "X_ch": 0,
            "X_pr": 0,  # varies
            "X_li": 0,
            "X_su": 0.00625,
            "X_aa": 0.00625,
            "X_fa": 0.00625,
            "X_c4": 0.00625,
            "X_pro": 0.00625,
            "X_ac": 0.00625,
            "X_h2": 0.00625,
            "X_I": 0,  # varies
            "S_an": 0,
            "S_cat": 0,
        }
        self.Ni = pyo.Var(
            self.state_ref.solute_set,
            initialize=Ni_dict,
            units=pyo.units.kmol / pyo.units.kg,
            domain=pyo.PositiveReals,
            doc="Nitrogen content of component, kmole N/kg COD",
        )
        # Stoichiometric Parameters (Table 6.1 in Batstone et al., 2002)
        self.f_sI_xc = pyo.Var(
            initialize=0.1,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Soluble inerts from composites",
        )
        self.f_xI_xc = pyo.Var(
            initialize=0.25,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Particulate inerts from composites",
        )
        self.f_ch_xc = pyo.Var(
            initialize=0.2,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Carbohydrates from composites",
        )
        self.f_pr_xc = pyo.Var(
            initialize=0.2,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Proteins from composites",
        )
        self.f_li_xc = pyo.Var(
            initialize=0.25,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Lipids from composites",
        )
        self.N_xc = pyo.Var(
            initialize=0.002,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Nitrogen content of composites",
        )
        self.N_I = pyo.Var(
            initialize=0.002,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Nitrogen content of inerts",
        )
        self.f_fa_li = pyo.Var(
            initialize=0.95,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Fatty acids from lipids",
        )
        self.f_h2_su = pyo.Var(
            initialize=0.19,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Hydrogen from sugars",
        )
        self.f_bu_su = pyo.Var(
            initialize=0.13,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Butyrate from sugars",
        )
        self.f_pro_su = pyo.Var(
            initialize=0.27,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Propionate from sugars",
        )
        self.f_ac_su = pyo.Var(
            initialize=0.41,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Acetate from sugars",
        )
        self.f_h2_aa = pyo.Var(
            initialize=0.06,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Hydrogen from amino acids",
        )
        self.N_aa = pyo.Var(
            initialize=0.007,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Nitrogen in amino acids and proteins",
        )
        self.f_va_aa = pyo.Var(
            initialize=0.23,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Valerate from amino acids",
        )
        self.f_bu_aa = pyo.Var(
            initialize=0.26,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Butyrate from amino acids",
        )
        self.f_pro_aa = pyo.Var(
            initialize=0.05,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Propionate from amino acids",
        )
        self.f_ac_aa = pyo.Var(
            initialize=0.40,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Acetate from amino acids",
        )
        # Kinetic Parameters

        # Reaction Stoichiometry
        # This is the stoichiometric part of the Peterson matrix in dict form.
        # Note that reaction stoichiometry is on a mass basis.
        # See Table 3.1 in Batstone et al., 2002.
        mw_n = 14 * pyo.units.kg / pyo.units.kmol

        # Exclude non-zero stoichiometric coefficients for S_IC initially since they depend on other stoichiometric coefficients.
        self.rate_reaction_stoichiometry = {
            # R1: Disintegration
            ("R1", "Liq", "H2O"): 0,
            ("R1", "Liq", "S_su"): 0,
            ("R1", "Liq", "S_aa"): 0,
            ("R1", "Liq", "S_fa"): 0,
            ("R1", "Liq", "S_va"): 0,
            ("R1", "Liq", "S_bu"): 0,
            ("R1", "Liq", "S_pro"): 0,
            ("R1", "Liq", "S_ac"): 0,
            ("R1", "Liq", "S_h2"): 0,
            ("R1", "Liq", "S_ch4"): 0,
            ("R1", "Liq", "S_IC"): 0,
            ("R1", "Liq", "S_IN"): 0,
            ("R1", "Liq", "S_I"): self.f_sI_xc,
            ("R1", "Liq", "X_c"): -1,
            ("R1", "Liq", "X_ch"): self.f_ch_xc,
            ("R1", "Liq", "X_pr"): self.f_pr_xc,
            ("R1", "Liq", "X_li"): self.f_li_xc,
            ("R1", "Liq", "X_su"): 0,
            ("R1", "Liq", "X_aa"): 0,
            ("R1", "Liq", "X_fa"): 0,
            ("R1", "Liq", "X_c4"): 0,
            ("R1", "Liq", "X_pro"): 0,
            ("R1", "Liq", "X_ac"): 0,
            ("R1", "Liq", "X_h2"): 0,
            ("R1", "Liq", "X_I"): self.f_xI_xc,
            # R2: Hydrolysis of carbohydrates
            ("R2", "Liq", "H2O"): 0,
            ("R2", "Liq", "S_su"): 1,
            ("R2", "Liq", "S_aa"): 0,
            ("R2", "Liq", "S_fa"): 0,
            ("R2", "Liq", "S_va"): 0,
            ("R2", "Liq", "S_bu"): 0,
            ("R2", "Liq", "S_pro"): 0,
            ("R2", "Liq", "S_ac"): 0,
            ("R2", "Liq", "S_h2"): 0,
            ("R2", "Liq", "S_ch4"): 0,
            ("R2", "Liq", "S_IC"): 0,
            ("R2", "Liq", "S_IN"): 0,
            ("R2", "Liq", "S_I"): 0,
            ("R2", "Liq", "X_c"): 0,
            ("R2", "Liq", "X_ch"): -1,
            ("R2", "Liq", "X_pr"): 0,
            ("R2", "Liq", "X_li"): 0,
            ("R2", "Liq", "X_su"): 0,
            ("R2", "Liq", "X_aa"): 0,
            ("R2", "Liq", "X_fa"): 0,
            ("R2", "Liq", "X_c4"): 0,
            ("R2", "Liq", "X_pro"): 0,
            ("R2", "Liq", "X_ac"): 0,
            ("R2", "Liq", "X_h2"): 0,
            ("R2", "Liq", "X_I"): 0,
            # R3:  Hydrolysis of proteins
            ("R3", "Liq", "H2O"): 0,
            ("R3", "Liq", "S_su"): 0,
            ("R3", "Liq", "S_aa"): 1,
            ("R3", "Liq", "S_fa"): 0,
            ("R3", "Liq", "S_va"): 0,
            ("R3", "Liq", "S_bu"): 0,
            ("R3", "Liq", "S_pro"): 0,
            ("R3", "Liq", "S_ac"): 0,
            ("R3", "Liq", "S_h2"): 0,
            ("R3", "Liq", "S_ch4"): 0,
            ("R3", "Liq", "S_IC"): 0,
            ("R3", "Liq", "S_IN"): 0,
            ("R3", "Liq", "S_I"): 0,
            ("R3", "Liq", "X_c"): 0,
            ("R3", "Liq", "X_ch"): 0,
            ("R3", "Liq", "X_pr"): -1,
            ("R3", "Liq", "X_li"): 0,
            ("R3", "Liq", "X_su"): 0,
            ("R3", "Liq", "X_aa"): 0,
            ("R3", "Liq", "X_fa"): 0,
            ("R3", "Liq", "X_c4"): 0,
            ("R3", "Liq", "X_pro"): 0,
            ("R3", "Liq", "X_ac"): 0,
            ("R3", "Liq", "X_h2"): 0,
            ("R3", "Liq", "X_I"): 0,
            # R4:  Hydrolysis of lipids
            ("R4", "Liq", "H2O"): 0,
            ("R4", "Liq", "S_su"): 1 - self.f_fa_li,
            ("R4", "Liq", "S_aa"): 0,
            ("R4", "Liq", "S_fa"): self.f_fa_li,
            ("R4", "Liq", "S_va"): 0,
            ("R4", "Liq", "S_bu"): 0,
            ("R4", "Liq", "S_pro"): 0,
            ("R4", "Liq", "S_ac"): 0,
            ("R4", "Liq", "S_h2"): 0,
            ("R4", "Liq", "S_ch4"): 0,
            ("R4", "Liq", "S_IC"): 0,
            ("R4", "Liq", "S_IN"): 0,
            ("R4", "Liq", "S_I"): 0,
            ("R4", "Liq", "X_c"): 0,
            ("R4", "Liq", "X_ch"): 0,
            ("R4", "Liq", "X_pr"): 0,
            ("R4", "Liq", "X_li"): -1,
            ("R4", "Liq", "X_su"): 0,
            ("R4", "Liq", "X_aa"): 0,
            ("R4", "Liq", "X_fa"): 0,
            ("R4", "Liq", "X_c4"): 0,
            ("R4", "Liq", "X_pro"): 0,
            ("R4", "Liq", "X_ac"): 0,
            ("R4", "Liq", "X_h2"): 0,
            ("R4", "Liq", "X_I"): 0,
            # R5:  Uptake of sugars
            ("R5", "Liq", "H2O"): 0,
            ("R5", "Liq", "S_su"): -1,
            ("R5", "Liq", "S_aa"): 0,
            ("R5", "Liq", "S_fa"): 0,
            ("R5", "Liq", "S_va"): 0,
            ("R5", "Liq", "S_bu"): (1 - self.Y_su) * self.f_bu_su,
            ("R5", "Liq", "S_pro"): (1 - self.Y_su) * self.f_pro_su,
            ("R5", "Liq", "S_ac"): (1 - self.Y_su) * self.f_ac_su,
            ("R5", "Liq", "S_h2"): (1 - self.Y_su) * self.f_h2_su,
            ("R5", "Liq", "S_ch4"): 0,
            # ("R5", "Liq", "S_IC"): -sum(self.conc_mass_comp_ref[S] * self.),
            ("R5", "Liq", "S_IN"): -self.Y_su * self.N_bac,
            ("R5", "Liq", "S_I"): 0,
            ("R5", "Liq", "X_c"): 0,
            ("R5", "Liq", "X_ch"): 0,
            ("R5", "Liq", "X_pr"): 0,
            ("R5", "Liq", "X_li"): 0,
            ("R5", "Liq", "X_su"): self.Y_su,
            ("R5", "Liq", "X_aa"): 0,
            ("R5", "Liq", "X_fa"): 0,
            ("R5", "Liq", "X_c4"): 0,
            ("R5", "Liq", "X_pro"): 0,
            ("R5", "Liq", "X_ac"): 0,
            ("R5", "Liq", "X_h2"): 0,
            ("R5", "Liq", "X_I"): 0,
            # R6:  Uptake of amino acids
            ("R6", "Liq", "H2O"): 0,
            ("R6", "Liq", "S_su"): 0,
            ("R6", "Liq", "S_aa"): -1,
            ("R6", "Liq", "S_fa"): 0,
            ("R6", "Liq", "S_va"): (1 - self.Y_aa) * self.f_va_aa,
            ("R6", "Liq", "S_bu"): (1 - self.Y_aa) * self.f_bu_aa,
            ("R6", "Liq", "S_pro"): (1 - self.Y_aa) * self.f_pro_aa,
            ("R6", "Liq", "S_ac"): (1 - self.Y_aa) * self.f_ac_aa,
            ("R6", "Liq", "S_h2"): (1 - self.Y_aa) * self.f_h2_aa,
            ("R6", "Liq", "S_ch4"): 0,
            # ("R6", "Liq", "S_IC"): 0,
            ("R6", "Liq", "S_IN"): self.N_aa - self.Y_aa * self.N_bac,
            ("R6", "Liq", "S_I"): 0,
            ("R6", "Liq", "X_c"): 0,
            ("R6", "Liq", "X_ch"): 0,
            ("R6", "Liq", "X_pr"): 0,
            ("R6", "Liq", "X_li"): 0,
            ("R6", "Liq", "X_su"): 0,
            ("R6", "Liq", "X_aa"): self.Y_aa,
            ("R6", "Liq", "X_fa"): 0,
            ("R6", "Liq", "X_c4"): 0,
            ("R6", "Liq", "X_pro"): 0,
            ("R6", "Liq", "X_ac"): 0,
            ("R6", "Liq", "X_h2"): 0,
            ("R6", "Liq", "X_I"): 0,
            # R7:  Uptake of long chain fatty acids (LCFAs)
            ("R7", "Liq", "H2O"): 0,
            ("R7", "Liq", "S_su"): 0,
            ("R7", "Liq", "S_aa"): 0,
            ("R7", "Liq", "S_fa"): -1,
            ("R7", "Liq", "S_va"): 0,
            ("R7", "Liq", "S_bu"): 0,
            ("R7", "Liq", "S_pro"): 0,
            ("R7", "Liq", "S_ac"): (1 - self.Y_fa) * 0.7,
            ("R7", "Liq", "S_h2"): (1 - self.Y_fa) * 0.3,
            ("R7", "Liq", "S_ch4"): 0,
            ("R7", "Liq", "S_IC"): 0,
            ("R7", "Liq", "S_IN"): -self.Y_fa * self.N_bac,
            ("R7", "Liq", "S_I"): 0,
            ("R7", "Liq", "X_c"): 0,
            ("R7", "Liq", "X_ch"): 0,
            ("R7", "Liq", "X_pr"): 0,
            ("R7", "Liq", "X_li"): 0,
            ("R7", "Liq", "X_su"): 0,
            ("R7", "Liq", "X_aa"): 0,
            ("R7", "Liq", "X_fa"): self.Y_fa,
            ("R7", "Liq", "X_c4"): 0,
            ("R7", "Liq", "X_pro"): 0,
            ("R7", "Liq", "X_ac"): 0,
            ("R7", "Liq", "X_h2"): 0,
            ("R7", "Liq", "X_I"): 0,
            # R8:  Uptake of valerate
            ("R8", "Liq", "H2O"): 0,
            ("R8", "Liq", "S_su"): 0,
            ("R8", "Liq", "S_aa"): 0,
            ("R8", "Liq", "S_fa"): 0,
            ("R8", "Liq", "S_va"): -1,
            ("R8", "Liq", "S_bu"): 0,
            ("R8", "Liq", "S_pro"): (1 - self.Y_c4) * 0.54,
            ("R8", "Liq", "S_ac"): (1 - self.Y_c4) * 0.31,
            ("R8", "Liq", "S_h2"): (1 - self.Y_c4) * 0.15,
            ("R8", "Liq", "S_ch4"): 0,
            ("R8", "Liq", "S_IC"): 0,
            ("R8", "Liq", "S_IN"): -self.Y_c4 * self.N_bac,
            ("R8", "Liq", "S_I"): 0,
            ("R8", "Liq", "X_c"): 0,
            ("R8", "Liq", "X_ch"): 0,
            ("R8", "Liq", "X_pr"): 0,
            ("R8", "Liq", "X_li"): 0,
            ("R8", "Liq", "X_su"): 0,
            ("R8", "Liq", "X_aa"): 0,
            ("R8", "Liq", "X_fa"): 0,
            ("R8", "Liq", "X_c4"): self.Y_c4,
            ("R8", "Liq", "X_pro"): 0,
            ("R8", "Liq", "X_ac"): 0,
            ("R8", "Liq", "X_h2"): 0,
            ("R8", "Liq", "X_I"): 0,
            # R9:  Uptake of butyrate
            ("R9", "Liq", "H2O"): 0,
            ("R9", "Liq", "S_su"): 0,
            ("R9", "Liq", "S_aa"): 0,
            ("R9", "Liq", "S_fa"): 0,
            ("R9", "Liq", "S_va"): 0,
            ("R9", "Liq", "S_bu"): -1,
            ("R9", "Liq", "S_pro"): 0,
            ("R9", "Liq", "S_ac"): (1 - self.Y_c4) * 0.8,
            ("R9", "Liq", "S_h2"): (1 - self.Y_c4) * 0.2,
            ("R9", "Liq", "S_ch4"): 0,
            ("R9", "Liq", "S_IC"): 0,
            ("R9", "Liq", "S_IN"): -self.Y_c4 * self.N_bac,
            ("R9", "Liq", "S_I"): 0,
            ("R9", "Liq", "X_c"): 0,
            ("R9", "Liq", "X_ch"): 0,
            ("R9", "Liq", "X_pr"): 0,
            ("R9", "Liq", "X_li"): 0,
            ("R9", "Liq", "X_su"): 0,
            ("R9", "Liq", "X_aa"): 0,
            ("R9", "Liq", "X_fa"): 0,
            ("R9", "Liq", "X_c4"): self.Y_c4,
            ("R9", "Liq", "X_pro"): 0,
            ("R9", "Liq", "X_ac"): 0,
            ("R9", "Liq", "X_h2"): 0,
            ("R9", "Liq", "X_I"): 0,
            # R10: Uptake of propionate
            ("R10", "Liq", "H2O"): 0,
            ("R10", "Liq", "S_su"): 0,
            ("R10", "Liq", "S_aa"): 0,
            ("R10", "Liq", "S_fa"): 0,
            ("R10", "Liq", "S_va"): 0,
            ("R10", "Liq", "S_bu"): 0,
            ("R10", "Liq", "S_pro"): -1,
            ("R10", "Liq", "S_ac"): (1 - self.Y_pro) * 0.57,
            ("R10", "Liq", "S_h2"): (1 - self.Y_pro) * 0.43,
            ("R10", "Liq", "S_ch4"): 0,
            # ("R10", "Liq", "S_IC"): 0,
            ("R10", "Liq", "S_IN"): -self.Y_pro * self.N_bac,
            ("R10", "Liq", "S_I"): 0,
            ("R10", "Liq", "X_c"): 0,
            ("R10", "Liq", "X_ch"): 0,
            ("R10", "Liq", "X_pr"): 0,
            ("R10", "Liq", "X_li"): 0,
            ("R10", "Liq", "X_su"): 0,
            ("R10", "Liq", "X_aa"): 0,
            ("R10", "Liq", "X_fa"): 0,
            ("R10", "Liq", "X_c4"): 0,
            ("R10", "Liq", "X_pro"): self.Y_pro,
            ("R10", "Liq", "X_ac"): 0,
            ("R10", "Liq", "X_h2"): 0,
            ("R10", "Liq", "X_I"): 0,
            # R11: Uptake of acetate
            ("R11", "Liq", "H2O"): 0,
            ("R11", "Liq", "S_su"): 0,
            ("R11", "Liq", "S_aa"): 0,
            ("R11", "Liq", "S_fa"): 0,
            ("R11", "Liq", "S_va"): 0,
            ("R11", "Liq", "S_bu"): 0,
            ("R11", "Liq", "S_pro"): 0,
            ("R11", "Liq", "S_ac"): -1,
            ("R11", "Liq", "S_h2"): 0,
            ("R11", "Liq", "S_ch4"): 1 - self.Y_ac,
            # ("R11", "Liq", "S_IC"): 0,
            ("R11", "Liq", "S_IN"): -self.Y_ac * self.N_bac,
            ("R11", "Liq", "S_I"): 0,
            ("R11", "Liq", "X_c"): 0,
            ("R11", "Liq", "X_ch"): 0,
            ("R11", "Liq", "X_pr"): 0,
            ("R11", "Liq", "X_li"): 0,
            ("R11", "Liq", "X_su"): 0,
            ("R11", "Liq", "X_aa"): 0,
            ("R11", "Liq", "X_fa"): 0,
            ("R11", "Liq", "X_c4"): 0,
            ("R11", "Liq", "X_pro"): 0,
            ("R11", "Liq", "X_ac"): self.Y_ac,
            ("R11", "Liq", "X_h2"): 0,
            ("R11", "Liq", "X_I"): 0,
            # R12: Uptake of hydrogen
            ("R12", "Liq", "H2O"): 0,
            ("R12", "Liq", "S_su"): 0,
            ("R12", "Liq", "S_aa"): 0,
            ("R12", "Liq", "S_fa"): 0,
            ("R12", "Liq", "S_va"): 0,
            ("R12", "Liq", "S_bu"): 0,
            ("R12", "Liq", "S_pro"): 0,
            ("R12", "Liq", "S_ac"): 0,
            ("R12", "Liq", "S_h2"): -1,
            ("R12", "Liq", "S_ch4"): (1 - self.Y_h2),
            # ("R12", "Liq", "S_IC"): 0,
            ("R12", "Liq", "S_IN"): -self.Y_h2 * self.N_bac,
            ("R12", "Liq", "S_I"): 0,
            ("R12", "Liq", "X_c"): 0,
            ("R12", "Liq", "X_ch"): 0,
            ("R12", "Liq", "X_pr"): 0,
            ("R12", "Liq", "X_li"): 0,
            ("R12", "Liq", "X_su"): 0,
            ("R12", "Liq", "X_aa"): 0,
            ("R12", "Liq", "X_fa"): 0,
            ("R12", "Liq", "X_c4"): 0,
            ("R12", "Liq", "X_pro"): 0,
            ("R12", "Liq", "X_ac"): 0,
            ("R12", "Liq", "X_h2"): self.Y_h2,
            ("R12", "Liq", "X_I"): 0,
            # R13: Decay of X_su
            ("R13", "Liq", "H2O"): 0,
            ("R13", "Liq", "S_su"): 0,
            ("R13", "Liq", "S_aa"): 0,
            ("R13", "Liq", "S_fa"): 0,
            ("R13", "Liq", "S_va"): 0,
            ("R13", "Liq", "S_bu"): 0,
            ("R13", "Liq", "S_pro"): 0,
            ("R13", "Liq", "S_ac"): 0,
            ("R13", "Liq", "S_h2"): 0,
            ("R13", "Liq", "S_ch4"): 0,
            ("R13", "Liq", "S_IC"): 0,
            ("R13", "Liq", "S_IN"): 0,
            ("R13", "Liq", "S_I"): 0,
            ("R13", "Liq", "X_c"): 1,
            ("R13", "Liq", "X_ch"): 0,
            ("R13", "Liq", "X_pr"): 0,
            ("R13", "Liq", "X_li"): 0,
            ("R13", "Liq", "X_su"): -1,
            ("R13", "Liq", "X_aa"): 0,
            ("R13", "Liq", "X_fa"): 0,
            ("R13", "Liq", "X_c4"): 0,
            ("R13", "Liq", "X_pro"): 0,
            ("R13", "Liq", "X_ac"): 0,
            ("R13", "Liq", "X_h2"): 0,
            ("R13", "Liq", "X_I"): 0,
            # R14: Decay of X_aa
            ("R14", "Liq", "H2O"): 0,
            ("R14", "Liq", "S_su"): 0,
            ("R14", "Liq", "S_aa"): 0,
            ("R14", "Liq", "S_fa"): 0,
            ("R14", "Liq", "S_va"): 0,
            ("R14", "Liq", "S_bu"): 0,
            ("R14", "Liq", "S_pro"): 0,
            ("R14", "Liq", "S_ac"): 0,
            ("R14", "Liq", "S_h2"): 0,
            ("R14", "Liq", "S_ch4"): 0,
            ("R14", "Liq", "S_IC"): 0,
            ("R14", "Liq", "S_IN"): 0,
            ("R14", "Liq", "S_I"): 0,
            ("R14", "Liq", "X_c"): 1,
            ("R14", "Liq", "X_ch"): 0,
            ("R14", "Liq", "X_pr"): 0,
            ("R14", "Liq", "X_li"): 0,
            ("R14", "Liq", "X_su"): 0,
            ("R14", "Liq", "X_aa"): -1,
            ("R14", "Liq", "X_fa"): 0,
            ("R14", "Liq", "X_c4"): 0,
            ("R14", "Liq", "X_pro"): 0,
            ("R14", "Liq", "X_ac"): 0,
            ("R14", "Liq", "X_h2"): 0,
            ("R14", "Liq", "X_I"): 0,
            # R15: Decay of X_fa
            ("R15", "Liq", "H2O"): 0,
            ("R15", "Liq", "S_su"): 0,
            ("R15", "Liq", "S_aa"): 0,
            ("R15", "Liq", "S_fa"): 0,
            ("R15", "Liq", "S_va"): 0,
            ("R15", "Liq", "S_bu"): 0,
            ("R15", "Liq", "S_pro"): 0,
            ("R15", "Liq", "S_ac"): 0,
            ("R15", "Liq", "S_h2"): 0,
            ("R15", "Liq", "S_ch4"): 0,
            ("R15", "Liq", "S_IC"): 0,
            ("R15", "Liq", "S_IN"): 0,
            ("R15", "Liq", "S_I"): 0,
            ("R15", "Liq", "X_c"): 1,
            ("R15", "Liq", "X_ch"): 0,
            ("R15", "Liq", "X_pr"): 0,
            ("R15", "Liq", "X_li"): 0,
            ("R15", "Liq", "X_su"): 0,
            ("R15", "Liq", "X_aa"): 0,
            ("R15", "Liq", "X_fa"): -1,
            ("R15", "Liq", "X_c4"): 0,
            ("R15", "Liq", "X_pro"): 0,
            ("R15", "Liq", "X_ac"): 0,
            ("R15", "Liq", "X_h2"): 0,
            ("R15", "Liq", "X_I"): 0,
            # R16: Decay of X_c4
            ("R16", "Liq", "H2O"): 0,
            ("R16", "Liq", "S_su"): 0,
            ("R16", "Liq", "S_aa"): 0,
            ("R16", "Liq", "S_fa"): 0,
            ("R16", "Liq", "S_va"): 0,
            ("R16", "Liq", "S_bu"): 0,
            ("R16", "Liq", "S_pro"): 0,
            ("R16", "Liq", "S_ac"): 0,
            ("R16", "Liq", "S_h2"): 0,
            ("R16", "Liq", "S_ch4"): 0,
            ("R16", "Liq", "S_IC"): 0,
            ("R16", "Liq", "S_IN"): 0,
            ("R16", "Liq", "S_I"): 0,
            ("R16", "Liq", "X_c"): 1,
            ("R16", "Liq", "X_ch"): 0,
            ("R16", "Liq", "X_pr"): 0,
            ("R16", "Liq", "X_li"): 0,
            ("R16", "Liq", "X_su"): 0,
            ("R16", "Liq", "X_aa"): 0,
            ("R16", "Liq", "X_fa"): 0,
            ("R16", "Liq", "X_c4"): -1,
            ("R16", "Liq", "X_pro"): 0,
            ("R16", "Liq", "X_ac"): 0,
            ("R16", "Liq", "X_h2"): 0,
            ("R16", "Liq", "X_I"): 0,
            # R17: Decay of X_pro
            ("R17", "Liq", "H2O"): 0,
            ("R17", "Liq", "S_su"): 0,
            ("R17", "Liq", "S_aa"): 0,
            ("R17", "Liq", "S_fa"): 0,
            ("R17", "Liq", "S_va"): 0,
            ("R17", "Liq", "S_bu"): 0,
            ("R17", "Liq", "S_pro"): 0,
            ("R17", "Liq", "S_ac"): 0,
            ("R17", "Liq", "S_h2"): 0,
            ("R17", "Liq", "S_ch4"): 0,
            ("R17", "Liq", "S_IC"): 0,
            ("R17", "Liq", "S_IN"): 0,
            ("R17", "Liq", "S_I"): 0,
            ("R17", "Liq", "X_c"): 1,
            ("R17", "Liq", "X_ch"): 0,
            ("R17", "Liq", "X_pr"): 0,
            ("R17", "Liq", "X_li"): 0,
            ("R17", "Liq", "X_su"): 0,
            ("R17", "Liq", "X_aa"): 0,
            ("R17", "Liq", "X_fa"): 0,
            ("R17", "Liq", "X_c4"): 0,
            ("R17", "Liq", "X_pro"): -1,
            ("R17", "Liq", "X_ac"): 0,
            ("R17", "Liq", "X_h2"): 0,
            ("R17", "Liq", "X_I"): 0,
            # R18: Decay of X_ac
            ("R18", "Liq", "H2O"): 0,
            ("R18", "Liq", "S_su"): 0,
            ("R18", "Liq", "S_aa"): 0,
            ("R18", "Liq", "S_fa"): 0,
            ("R18", "Liq", "S_va"): 0,
            ("R18", "Liq", "S_bu"): 0,
            ("R18", "Liq", "S_pro"): 0,
            ("R18", "Liq", "S_ac"): 0,
            ("R18", "Liq", "S_h2"): 0,
            ("R18", "Liq", "S_ch4"): 0,
            ("R18", "Liq", "S_IC"): 0,
            ("R18", "Liq", "S_IN"): 0,
            ("R18", "Liq", "S_I"): 0,
            ("R18", "Liq", "X_c"): 1,
            ("R18", "Liq", "X_ch"): 0,
            ("R18", "Liq", "X_pr"): 0,
            ("R18", "Liq", "X_li"): 0,
            ("R18", "Liq", "X_su"): 0,
            ("R18", "Liq", "X_aa"): 0,
            ("R18", "Liq", "X_fa"): 0,
            ("R18", "Liq", "X_c4"): 0,
            ("R18", "Liq", "X_pro"): 0,
            ("R18", "Liq", "X_ac"): -1,
            ("R18", "Liq", "X_h2"): 0,
            ("R18", "Liq", "X_I"): 0,
            # R19: Decay of X_h2
            ("R19", "Liq", "H2O"): 0,
            ("R19", "Liq", "S_su"): 0,
            ("R19", "Liq", "S_aa"): 0,
            ("R19", "Liq", "S_fa"): 0,
            ("R19", "Liq", "S_va"): 0,
            ("R19", "Liq", "S_bu"): 0,
            ("R19", "Liq", "S_pro"): 0,
            ("R19", "Liq", "S_ac"): 0,
            ("R19", "Liq", "S_h2"): 0,
            ("R19", "Liq", "S_ch4"): 0,
            ("R19", "Liq", "S_IC"): 0,
            ("R19", "Liq", "S_IN"): 0,
            ("R19", "Liq", "S_I"): 0,
            ("R19", "Liq", "X_c"): 1,
            ("R19", "Liq", "X_ch"): 0,
            ("R19", "Liq", "X_pr"): 0,
            ("R19", "Liq", "X_li"): 0,
            ("R19", "Liq", "X_su"): 0,
            ("R19", "Liq", "X_aa"): 0,
            ("R19", "Liq", "X_fa"): 0,
            ("R19", "Liq", "X_c4"): 0,
            ("R19", "Liq", "X_pro"): 0,
            ("R19", "Liq", "X_ac"): 0,
            ("R19", "Liq", "X_h2"): -1,
            ("R19", "Liq", "X_I"): 0,
        }

        # TODO: Add S_IC stoichiometric coefficients that depend on other stoich coefficients (R5, 6, 10, 11, 12)
        #  - determine conversion factors for S_IC stoichiometry as/if needed
        s_ic_rxns = ["R5", "R6", "R10", "R11", "R12"]
        for R in s_ic_rxns:
            self.rate_reaction_stoichiometry[R, "Liq", "S_IC"] = -sum(
                self.Ci[S] * self.rate_reaction_stoichiometry[R, "Liq", S]
                for S in self.state_ref.solute_set
                if S != "S_IC"
            )

        # Fix all the variables we just created
        for v in self.component_objects(pyo.Var, descend_into=False):
            v.fix()

    @classmethod
    def define_metadata(cls, obj):
        obj.add_properties(
            {
                "reaction_rate": {"method": "_rxn_rate"},
            }
        )
        obj.add_default_units(
            {
                "time": pyo.units.s,
                "length": pyo.units.m,
                "mass": pyo.units.kg,
                "amount": pyo.units.kmol,
                "temperature": pyo.units.K,
            }
        )


class _ADM1ReactionBlock(ReactionBlockBase):
    """
    This Class contains methods which should be applied to Reaction Blocks as a
    whole, rather than individual elements of indexed Reaction Blocks.
    """

    def initialize(blk, outlvl=idaeslog.NOTSET, **kwargs):
        """
        Initialization routine for reaction package.

        Keyword Arguments:
            outlvl : sets output level of initialization routine

        Returns:
            None
        """
        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="properties")
        init_log.info("Initialization Complete.")


@declare_process_block_class("ADM1ReactionBlock", block_class=_ADM1ReactionBlock)
class ADM1ReactionBlockData(ReactionBlockDataBase):
    """
    ReactionBlcok for ADM1.
    """

    def build(self):
        """
        Callable method for Block construction
        """
        super().build()

        # Create references to state vars
        # Concentration
        add_object_reference(self, "conc_mass_comp_ref", self.state_ref.conc_mass_comp)

    # Rate of reaction method
    def _rxn_rate(self):
        self.reaction_rate = pyo.Var(
            self.params.rate_reaction_idx,
            initialize=0,
            doc="Rate of reaction",
            units=pyo.units.kg / pyo.units.m**3 / pyo.units.s,
        )

        # try:
        #
        #     def rate_expression_rule(b, r):
        # if r == "R1":
        #     # R1: Aerobic growth of heterotrophs
        #     return b.reaction_rate[r] == pyo.units.convert(
        #         b.params.mu_H
        #         * (
        #             b.conc_mass_comp_ref["S_S"]
        #             / (b.params.K_S + b.conc_mass_comp_ref["S_S"])
        #         )
        #         * (
        #             b.conc_mass_comp_ref["S_O"]
        #             / (b.params.K_OH + b.conc_mass_comp_ref["S_O"])
        #         )
        #         * b.conc_mass_comp_ref["X_BH"],
        #         to_units=pyo.units.kg / pyo.units.m**3 / pyo.units.s,
        #     )
        # elif r == "R2":
        #     # R2: Anoxic growth of heterotrophs
        #     return b.reaction_rate[r] == pyo.units.convert(
        #         b.params.mu_H
        #         * (
        #             b.conc_mass_comp_ref["S_S"]
        #             / (b.params.K_S + b.conc_mass_comp_ref["S_S"])
        #         )
        #         * (
        #             b.params.K_OH
        #             / (b.params.K_OH + b.conc_mass_comp_ref["S_O"])
        #         )
        #         * (
        #             b.conc_mass_comp_ref["S_NO"]
        #             / (b.params.K_NO + b.conc_mass_comp_ref["S_NO"])
        #         )
        #         * b.params.eta_g
        #         * b.conc_mass_comp_ref["X_BH"],
        #         to_units=pyo.units.kg / pyo.units.m**3 / pyo.units.s,
        #     )
        # elif r == "R3":
        #     # R3: Aerobic growth of autotrophs
        #     return b.reaction_rate[r] == pyo.units.convert(
        #         b.params.mu_A
        #         * (
        #             b.conc_mass_comp_ref["S_NH"]
        #             / (b.params.K_NH + b.conc_mass_comp_ref["S_NH"])
        #         )
        #         * (
        #             b.conc_mass_comp_ref["S_O"]
        #             / (b.params.K_OA + b.conc_mass_comp_ref["S_O"])
        #         )
        #         * b.conc_mass_comp_ref["X_BA"],
        #         to_units=pyo.units.kg / pyo.units.m**3 / pyo.units.s,
        #     )
        # elif r == "R4":
        #     # R4: Decay of heterotrophs
        #     return b.reaction_rate[r] == pyo.units.convert(
        #         b.params.b_H * b.conc_mass_comp_ref["X_BH"],
        #         to_units=pyo.units.kg / pyo.units.m**3 / pyo.units.s,
        #     )
        # elif r == "R5":
        #     # R5: Decay of autotrophs
        #     return b.reaction_rate[r] == pyo.units.convert(
        #         b.params.b_A * b.conc_mass_comp_ref["X_BA"],
        #         to_units=pyo.units.kg / pyo.units.m**3 / pyo.units.s,
        #     )
        # elif r == "R6":
        #     # R6: Ammonification of soluble organic nitrogen
        #     return b.reaction_rate[r] == pyo.units.convert(
        #         b.params.k_a
        #         * b.conc_mass_comp_ref["S_ND"]
        #         * b.conc_mass_comp_ref["X_BH"],
        #         to_units=pyo.units.kg / pyo.units.m**3 / pyo.units.s,
        #     )
        # elif r == "R7":
        #     # R7: Hydrolysis of entrapped organics
        #     return b.reaction_rate[r] == pyo.units.convert(
        #         b.params.k_h
        #         * (b.conc_mass_comp_ref["X_S"] / b.conc_mass_comp_ref["X_BH"])
        #         / (
        #             b.params.K_X
        #             + (
        #                 b.conc_mass_comp_ref["X_S"]
        #                 / b.conc_mass_comp_ref["X_BH"]
        #             )
        #         )
        #         * (
        #             (
        #                 b.conc_mass_comp_ref["S_O"]
        #                 / (b.params.K_OH + b.conc_mass_comp_ref["S_O"])
        #             )
        #             + b.params.eta_h
        #             * b.params.K_OH
        #             / (b.params.K_OH + b.conc_mass_comp_ref["S_O"])
        #             * (
        #                 b.conc_mass_comp_ref["S_NO"]
        #                 / (b.params.K_NO + b.conc_mass_comp_ref["S_NO"])
        #             )
        #         )
        #         * b.conc_mass_comp_ref["X_BH"],
        #         to_units=pyo.units.kg / pyo.units.m**3 / pyo.units.s,
        #     )
        # elif r == "R8":
        #     # R8: Hydrolysis of entrapped organic nitrogen
        #     return b.reaction_rate[r] == (
        #         b.reaction_rate["R7"]
        #         * (b.conc_mass_comp_ref["X_ND"] / b.conc_mass_comp_ref["X_S"])
        #     )
        # else:
        #     raise BurntToast()

        # self.rate_expression = pyo.Constraint(
        #     self.params.rate_reaction_idx,
        #     rule=rate_expression_rule,
        #     doc="ADM1 rate expressions",
        # )

        # except AttributeError:
        #     # If constraint fails, clean up so that DAE can try again later
        #     self.del_component(self.reaction_rate)
        #     self.del_component(self.rate_expression)
        #     raise

    def get_reaction_rate_basis(b):
        return MaterialFlowBasis.mass
