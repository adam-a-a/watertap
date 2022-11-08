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

        self.rate_reaction_idx = pyo.Set(
            initialize=[
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
        )

        # TODO: for all carbon and nitrogen content values with "varies" need to revisited; putting 0s for now
        # Carbon content
        Ci_dict = {
            "S_su": 0.0313,
            "S_aa": 0.03,  # varies
            "S_fa": 0.0217,
            "S_va": 0.0240,
            "S_bu": 0.0250,
            "S_pro": 0.0268,
            "S_ac": 0.0313,
            "S_ch4": 0.0156,
            "S_IC": 1,
            "S_I": 0.03,  # varies
            "X_c": 0.02786,  # varies
            "X_ch": 0.0313,
            "X_pr": 0.03,  # varies
            "X_li": 0.0220,
            "X_su": 0.0313,
            "X_aa": 0.0313,
            "X_fa": 0.0313,
            "X_c4": 0.0313,
            "X_pro": 0.0313,
            "X_ac": 0.0313,
            "X_I": 0.03,  # varies
        }

        self.Ci = pyo.Var(
            Ci_dict.keys(),
            initialize=Ci_dict,
            units=pyo.units.kmol / pyo.units.kg,
            domain=pyo.PositiveReals,
            doc="Carbon content of component [kmole C/kg COD]",
        )

        mw_n = 14 * pyo.units.kg / pyo.units.kmol
        mw_c = 12 * pyo.units.kg / pyo.units.kmol

        # Nitrogen content
        # N_bac is nitrogen content
        N_bac = 0.08 * pyo.units.kg * pyo.units.m**-3 / mw_n

        # Stoichiometric Parameters (Table 6.1 in Batstone et al., 2002)
        self.f_sI_xc = pyo.Var(
            initialize=0.1,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Soluble inerts from composites",
        )
        self.f_xI_xc = pyo.Var(
            initialize=0.20,  # replacing 0.25 with 0.2 in accordance with Rosen & Jeppsson, 2006
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
            initialize=0.30,  # replacing 0.25 with 0.3 in accordance with Rosen & Jeppsson, 2006
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Lipids from composites",
        )
        self.N_xc = pyo.Var(
            initialize=0.0376
            / 14,  # change from 0.002 to 0.0376/14 based on Rosen & Jeppsson, 2006
            units=pyo.units.kmol * pyo.units.kg**-1,
            domain=pyo.PositiveReals,
            doc="Nitrogen content of composites [kmole N/kg COD]",
        )
        self.N_I = pyo.Var(
            initialize=0.06
            / 14,  # change from 0.002 to 0.06/14 based on Rosen & Jeppsson, 2006
            units=pyo.units.kmol * pyo.units.kg**-1,
            domain=pyo.PositiveReals,
            doc="Nitrogen content of inerts [kmole N/kg COD]",
        )
        self.N_aa = pyo.Var(
            initialize=0.007,
            units=pyo.units.kmol * pyo.units.kg**-1,
            domain=pyo.PositiveReals,
            doc="Nitrogen in amino acids and proteins [kmole N/kg COD]",
        )
        self.N_bac = pyo.Var(
            initialize=0.08 / 14,
            units=pyo.units.kmol * pyo.units.kg**-1,
            # units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Nitrogen content in bacteria [kmole N/kg COD]",
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
        self.Y_su = pyo.Var(
            initialize=0.10,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Yield of biomass on sugar substrate [kg COD X/ kg COD S]",
        )
        self.Y_aa = pyo.Var(
            initialize=0.08,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Yield of biomass on amino acid substrate [kg COD X/ kg COD S]",
        )
        self.Y_fa = pyo.Var(
            initialize=0.06,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Yield of biomass on fatty acid substrate [kg COD X/ kg COD S]",
        )
        self.Y_c4 = pyo.Var(
            initialize=0.06,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Yield of biomass on valerate and butyrate substrate [kg COD X/ kg COD S]",
        )
        self.Y_pro = pyo.Var(
            initialize=0.04,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Yield of biomass on propionate substrate [kg COD X/ kg COD S]",
        )
        self.Y_ac = pyo.Var(
            initialize=0.05,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Yield of biomass on acetate substrate [kg COD X/ kg COD S]",
        )
        self.Y_h2 = pyo.Var(
            initialize=0.06,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Yield of hydrogen per biomass [kg COD S/ kg COD X]",  # TODO: unsure if typo in Rosen & Jeppsson, 2006- revisit
        )
        # Biochemical Parameters
        self.k_dis = pyo.Var(
            initialize=0.5,
            units=pyo.units.day**-1,
            domain=pyo.PositiveReals,
            doc="First-order kinetic parameter for disintegration",
        )
        self.k_hyd_ch = pyo.Var(
            initialize=10,
            units=pyo.units.day**-1,
            domain=pyo.PositiveReals,
            doc="First-order kinetic parameter for hydrolysis of carbohydrates",
        )
        self.k_hyd_pr = pyo.Var(
            initialize=10,
            units=pyo.units.day**-1,
            domain=pyo.PositiveReals,
            doc="First-order kinetic parameter for hydrolysis of proteins",
        )
        self.k_hyd_li = pyo.Var(
            initialize=10,
            units=pyo.units.day**-1,
            domain=pyo.PositiveReals,
            doc="First-order kinetic parameter for hydrolysis of lipids",
        )
        self.K_S_IN = pyo.Var(
            initialize=1e-4,
            units=pyo.units.kmol * pyo.units.m**-3,
            domain=pyo.PositiveReals,
            doc="Inhibition parameter for inorganic nitrogen",
        )
        self.k_m_su = pyo.Var(
            initialize=30,
            units=pyo.units.day**-1,
            domain=pyo.PositiveReals,
            doc="Monod maximum specific uptake rate of sugars",
        )
        self.K_S_su = pyo.Var(
            initialize=0.5,
            units=pyo.units.kg * pyo.units.m**-3,
            domain=pyo.PositiveReals,
            doc="Half saturation value for uptake of sugars",
        )
        self.pH_UL_aa = pyo.Var(
            initialize=5.5,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Upper limit of pH for uptake rate of amino acids",
        )
        self.pH_LL_aa = pyo.Var(
            initialize=4,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Lower limit of pH for uptake rate of amino acids",
        )
        self.k_m_aa = pyo.Var(
            initialize=50,
            units=pyo.units.day**-1,
            domain=pyo.PositiveReals,
            doc="Monod maximum specific uptake rate of amino acids",
        )

        self.K_S_aa = pyo.Var(
            initialize=0.3,
            units=pyo.units.kg * pyo.units.m**-3,
            domain=pyo.PositiveReals,
            doc="Half saturation value for uptake of amino acids",
        )
        self.k_m_fa = pyo.Var(
            initialize=6,
            units=pyo.units.day**-1,
            domain=pyo.PositiveReals,
            doc="Monod maximum specific uptake rate of fatty acids",
        )
        self.K_S_fa = pyo.Var(
            initialize=0.4,
            units=pyo.units.kg * pyo.units.m**-3,
            domain=pyo.PositiveReals,
            doc="Half saturation value for uptake of fatty acids",
        )
        self.K_I_h2_fa = pyo.Var(
            initialize=5e-6,
            units=pyo.units.kg * pyo.units.m**-3,
            domain=pyo.PositiveReals,
            doc="Inhibition parameter for hydrogen during uptake of fatty acids",
        )
        self.k_m_c4 = pyo.Var(
            initialize=20,
            units=pyo.units.day**-1,
            domain=pyo.PositiveReals,
            doc="Monod maximum specific uptake rate of valerate and butyrate",
        )
        self.K_S_c4 = pyo.Var(
            initialize=0.2,
            units=pyo.units.kg * pyo.units.m**-3,
            domain=pyo.PositiveReals,
            doc="Half saturation value for uptake of valerate and butyrate",
        )
        self.K_I_h2_c4 = pyo.Var(
            initialize=1e-5,
            units=pyo.units.kg * pyo.units.m**-3,
            domain=pyo.PositiveReals,
            doc="Inhibition parameter for hydrogen during uptake of valerate and butyrate",
        )
        self.k_m_pro = pyo.Var(
            initialize=13,
            units=pyo.units.day**-1,
            domain=pyo.PositiveReals,
            doc="Monod maximum specific uptake rate of propionate",
        )
        self.K_S_pro = pyo.Var(
            initialize=0.1,
            units=pyo.units.kg * pyo.units.m**-3,
            domain=pyo.PositiveReals,
            doc="Half saturation value for uptake of propionate",
        )
        self.K_I_h2_pro = pyo.Var(
            initialize=3.5e-6,
            units=pyo.units.kg * pyo.units.m**-3,
            domain=pyo.PositiveReals,
            doc="Inhibition parameter for hydrogen during uptake of propionate",
        )
        self.k_m_ac = pyo.Var(
            initialize=8,
            units=pyo.units.day**-1,
            domain=pyo.PositiveReals,
            doc="Monod maximum specific uptake rate of acetate",
        )
        self.K_S_ac = pyo.Var(
            initialize=0.15,
            units=pyo.units.kg * pyo.units.m**-3,
            domain=pyo.PositiveReals,
            doc="Half saturation value for uptake of acetate",
        )
        self.K_I_nh3 = pyo.Var(
            initialize=0.0018,
            units=pyo.units.kmol * pyo.units.m**-3,
            domain=pyo.PositiveReals,
            doc="Inhibition parameter for ammonia during uptake of acetate",
        )
        self.pH_UL_ac = pyo.Var(
            initialize=7,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Upper limit of pH for uptake rate of acetate",
        )
        self.pH_LL_ac = pyo.Var(
            initialize=6,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Lower limit of pH for uptake rate of acetate",
        )
        self.k_m_h2 = pyo.Var(
            initialize=35,
            units=pyo.units.day**-1,
            domain=pyo.PositiveReals,
            doc="Monod maximum specific uptake rate of hydrogen",
        )
        self.K_S_h2 = pyo.Var(
            initialize=7e-6,
            units=pyo.units.kg * pyo.units.m**-3,
            domain=pyo.PositiveReals,
            doc="Half saturation value for uptake of hydrogen",
        )
        self.pH_UL_h2 = pyo.Var(
            initialize=6,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Upper limit of pH for uptake rate of hydrogen",
        )
        self.pH_LL_h2 = pyo.Var(
            initialize=5,
            units=pyo.units.dimensionless,
            domain=pyo.PositiveReals,
            doc="Lower limit of pH for uptake rate of hydrogen",
        )
        self.k_dec_X_su = pyo.Var(
            initialize=0.02,
            units=pyo.units.day**-1,
            domain=pyo.PositiveReals,
            doc="First-order decay rate for X_su",
        )
        self.k_dec_X_aa = pyo.Var(
            initialize=0.02,
            units=pyo.units.day**-1,
            domain=pyo.PositiveReals,
            doc="First-order decay rate for X_aa",
        )
        self.k_dec_X_fa = pyo.Var(
            initialize=0.02,
            units=pyo.units.day**-1,
            domain=pyo.PositiveReals,
            doc="First-order decay rate for X_fa",
        )
        self.k_dec_X_c4 = pyo.Var(
            initialize=0.02,
            units=pyo.units.day**-1,
            domain=pyo.PositiveReals,
            doc="First-order decay rate for X_c4",
        )
        self.k_dec_X_pro = pyo.Var(
            initialize=0.02,
            units=pyo.units.day**-1,
            domain=pyo.PositiveReals,
            doc="First-order decay rate for X_pro",
        )
        self.k_dec_X_ac = pyo.Var(
            initialize=0.02,
            units=pyo.units.day**-1,
            domain=pyo.PositiveReals,
            doc="First-order decay rate for X_ac",
        )
        self.k_dec_X_h2 = pyo.Var(
            initialize=0.02,
            units=pyo.units.day**-1,
            domain=pyo.PositiveReals,
            doc="First-order decay rate for X_h2",
        )
        # Todo: calculate this from expression
        # Todo: complete doc
        self.KW = pyo.Var(
            initialize=2.08e-14,
            units=pyo.units.kmol,
            domain=pyo.PositiveReals,
            doc="KW",
        )
        self.Ka_va = pyo.Var(
            initialize=1.38e-5,
            units=pyo.units.kmol,
            domain=pyo.PositiveReals,
            doc="Ka_va",
        )
        self.Ka_bu = pyo.Var(
            initialize=1.5e-5,
            units=pyo.units.kmol,
            domain=pyo.PositiveReals,
            doc="Ka_bu",
        )
        self.Ka_pro = pyo.Var(
            initialize=1.32e-5,
            units=pyo.units.kmol,
            domain=pyo.PositiveReals,
            doc="Ka_pro",
        )
        self.Ka_ac = pyo.Var(
            initialize=1.74e-5,
            units=pyo.units.kmol,
            domain=pyo.PositiveReals,
            doc="Ka_ac",
        )
        self.Ka_co2 = pyo.Var(
            initialize=4.94e-7,
            units=pyo.units.kmol,
            domain=pyo.PositiveReals,
            doc="Ka_co2",
        )
        self.Ka_IN = pyo.Var(
            initialize=1.11e-9,
            units=pyo.units.kmol,
            domain=pyo.PositiveReals,
            doc="Ka_IN",
        )
        self.KA_Bva = pyo.Var(
            initialize=1.1**8,
            units=pyo.units.kmol**-1 * pyo.units.day**-1,
            domain=pyo.PositiveReals,
            doc="Ka_IN",
        )
        self.KA_Bbu = pyo.Var(
            initialize=1.1**8,
            units=pyo.units.kmol**-1 * pyo.units.day**-1,
            domain=pyo.PositiveReals,
            doc="KA_Bbu",
        )
        self.KA_Bpro = pyo.Var(
            initialize=1.1**8,
            units=pyo.units.kmol**-1 * pyo.units.day**-1,
            domain=pyo.PositiveReals,
            doc="KA_Bpro",
        )
        self.KA_Bac = pyo.Var(
            initialize=1.1**8,
            units=pyo.units.kmol**-1 * pyo.units.day**-1,
            domain=pyo.PositiveReals,
            doc="KA_Bac",
        )
        self.KA_Bco2 = pyo.Var(
            initialize=1.1**8,
            units=pyo.units.kmol**-1 * pyo.units.day**-1,
            domain=pyo.PositiveReals,
            doc="KA_Bco2",
        )
        self.KA_BIN = pyo.Var(
            initialize=1.1**8,
            units=pyo.units.kmol**-1 * pyo.units.day**-1,
            domain=pyo.PositiveReals,
            doc="KA_BIN",
        )

        # Reaction Stoichiometry
        # This is the stoichiometric part of the Peterson matrix in dict form.
        # See Table 3.1 in Batstone et al., 2002.

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
            # ("R1", "Liq", "S_IN"): self.N_xc
            # - self.f_xI_xc * self.N_I
            # - self.f_sI_xc * self.N_I
            # - self.f_pr_xc * self.N_aa,
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
            ("R5", "Liq", "S_IC"): 0,  ## updated later
            ("R5", "Liq", "S_IN"): (-self.Y_su * self.N_bac) * mw_n,
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
            ("R6", "Liq", "S_IC"): 0,
            ("R6", "Liq", "S_IN"): (self.N_aa - self.Y_aa * self.N_bac) * mw_n,
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
            ("R7", "Liq", "S_IN"): (-self.Y_fa * self.N_bac) * mw_n,
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
            ("R8", "Liq", "S_IN"): (-self.Y_c4 * self.N_bac) * mw_n,
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
            ("R9", "Liq", "S_IN"): (-self.Y_c4 * self.N_bac) * mw_n,
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
            ("R10", "Liq", "S_IC"): 0,
            ("R10", "Liq", "S_IN"): (-self.Y_pro * self.N_bac) * mw_n,
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
            ("R11", "Liq", "S_IC"): 0,
            ("R11", "Liq", "S_IN"): (-self.Y_ac * self.N_bac) * mw_n,
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
            ("R12", "Liq", "S_IC"): 0,
            ("R12", "Liq", "S_IN"): (-self.Y_h2 * self.N_bac) * mw_n,
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
            ("R13", "Liq", "S_IN"): (self.N_bac - self.N_xc) * mw_n,
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
            ("R14", "Liq", "S_IN"): (self.N_bac - self.N_xc) * mw_n,
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
            ("R15", "Liq", "S_IN"): (self.N_bac - self.N_xc) * mw_n,
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
            ("R16", "Liq", "S_IN"): (self.N_bac - self.N_xc) * mw_n,
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
            ("R17", "Liq", "S_IN"): (self.N_bac - self.N_xc) * mw_n,
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
            ("R18", "Liq", "S_IN"): (self.N_bac - self.N_xc) * mw_n,
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
            ("R19", "Liq", "S_IN"): (self.N_bac - self.N_xc) * mw_n,
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

        s_ic_rxns = ["R5", "R6", "R10", "R11", "R12"]

        for R in s_ic_rxns:
            self.rate_reaction_stoichiometry[R, "Liq", "S_IC"] = -sum(
                self.Ci[S] * self.rate_reaction_stoichiometry[R, "Liq", S] * mw_c
                for S in Ci_dict.keys()
                if S != "S_IC"
            )

        for R in self.rate_reaction_idx:
            self.rate_reaction_stoichiometry[R, "Liq", "S_cat"] = 0
            self.rate_reaction_stoichiometry[R, "Liq", "S_an"] = 0

        # Fix all the variables we just created
        for v in self.component_objects(pyo.Var, descend_into=False):
            v.fix()

    @classmethod
    def define_metadata(cls, obj):
        obj.add_properties(
            {"reaction_rate": {"method": "_rxn_rate"}, "I": {"method": "_I"}}
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
    ReactionBlock for ADM1.
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

        self.conc_mol_va = pyo.Var(
            initialize=0.01,
            domain=pyo.NonNegativeReals,
            doc="molar concentration of va-",
            units=pyo.units.kmol / pyo.units.m**3,
        )

        self.conc_mol_bu = pyo.Var(
            initialize=0.01,
            domain=pyo.NonNegativeReals,
            doc="molar concentration of bu-",
            units=pyo.units.kmol / pyo.units.m**3,
        )
        self.conc_mol_pro = pyo.Var(
            initialize=0.01,
            domain=pyo.NonNegativeReals,
            doc="molar concentration of pro-",
            units=pyo.units.kmol / pyo.units.m**3,
        )
        self.conc_mol_ac = pyo.Var(
            initialize=0.01,
            domain=pyo.NonNegativeReals,
            doc="molar concentration of ac-",
            units=pyo.units.kmol / pyo.units.m**3,
        )
        self.conc_mol_hco3 = pyo.Var(
            initialize=0.01,
            domain=pyo.NonNegativeReals,
            doc="molar concentration of hco3",
            units=pyo.units.kmol / pyo.units.m**3,
        )
        self.conc_mol_nh3 = pyo.Var(
            initialize=0.01,
            domain=pyo.NonNegativeReals,
            doc="molar concentration of nh3",
            units=pyo.units.kmol / pyo.units.m**3,
        )
        self.conc_mol_co2 = pyo.Var(
            initialize=0.01,
            domain=pyo.NonNegativeReals,
            doc="molar concentration of co2",
            units=pyo.units.kmol / pyo.units.m**3,
        )
        self.conc_mol_nh4 = pyo.Var(
            initialize=0.01,
            domain=pyo.NonNegativeReals,
            doc="molar concentration of nh4",
            units=pyo.units.kmol / pyo.units.m**3,
        )
        self.S_H = pyo.Var(
            initialize=0.01,
            domain=pyo.NonNegativeReals,
            doc="molar concentration of H",
            units=pyo.units.kmol / pyo.units.m**3,
        )
        self.S_OH = pyo.Var(
            initialize=0.01,
            domain=pyo.NonNegativeReals,
            doc="molar concentration of OH",
            units=pyo.units.kmol / pyo.units.m**3,
        )

        mw_n = 14 * pyo.units.kg / pyo.units.kmol
        mw_nh3 = 16 * pyo.units.kg / pyo.units.kmol

        def concentration_of_va_rule(self):
            return (
                self.conc_mol_va
                == self.params.K_a_va
                ** self.state_ref.conc_mol_phase_comp["Liq", "S_va"]
                / (self.params.K_a_va + self.S_H)
            )

            self.concentration_of_va = pyo.Constraint(
                rule=concentration_of_va_rule,
                doc="constraint concentration of va-",
            )

        def concentration_of_bu_rule(self):
            return (
                self.conc_mol_bu
                == self.params.K_a_bu
                ** self.state_ref.conc_mol_phase_comp["Liq", "S_bu"]
                / (self.params.K_a_bu + self.S_H)
            )

            self.concentration_of_bu = pyo.Constraint(
                rule=concentration_of_bu_rule,
                doc="constraint concentration of bu-",
            )

        def concentration_of_pro_rule(self):
            return (
                self.conc_mol_pro
                == self.params.K_a_pro
                ** self.state_ref.conc_mol_phase_comp["Liq", "S_pro"]
                / (self.params.K_a_pro + self.S_H)
            )

            self.concentration_of_pro = pyo.Constraint(
                rule=concentration_of_pro_rule,
                doc="constraint concentration of pro-",
            )

        def concentration_of_ac_rule(self):
            return (
                self.conc_mol_ac
                == self.params.K_a_ac
                ** self.state_ref.conc_mol_phase_comp["Liq", "S_ac"]
                / (self.params.K_a_ac + self.S_H)
            )

            self.concentration_of_ac = pyo.Constraint(
                rule=concentration_of_ac_rule,
                doc="constraint concentration of ac-",
            )

        def concentration_of_hco3_rule(self):
            return (
                self.conc_mol_hco3
                == self.params.K_a_co2
                ** self.state_ref.conc_mol_phase_comp["Liq", "S_IC"]
                / (self.params.K_a_co2 + self.S_H)
            )

            self.concentration_of_hco3 = pyo.Constraint(
                rule=concentration_of_hco3_rule,
                doc="constraint concentration of hco3",
            )

        def concentration_of_nh3_rule(self):
            return (
                self.conc_mol_nh3
                == self.params.K_a_IN
                ** self.state_ref.conc_mol_phase_comp["Liq", "S_pro"]
                / (self.params.K_a_IN + self.S_H)
            )

            self.concentration_of_nh3 = pyo.Constraint(
                rule=concentration_of_nh3_rule,
                doc="constraint concentration of nh3",
            )

        def concentration_of_co2_rule(self):
            return (
                self.conc_mol_co2
                == self.state_ref.conc_mol_phase_comp["Liq", "S_IC"]
                - self.conc_mol_hco3
            )

            self.concentration_of_co2 = pyo.Constraint(
                rule=concentration_of_co2_rule,
                doc="constraint concentration of co2",
            )

        def concentration_of_nh4_rule(self):
            return (
                self.conc_mol_nh4
                == self.state_ref.conc_mol_phase_comp["Liq", "S_IN"] - self.conc_mol_nh3
            )

            self.concentration_of_pro = pyo.Constraint(
                rule=concentration_of_pro_rule,
                doc="constraint concentration of pro-",
            )

        def S_OH_rule(self):
            return self.S_OH == self.params.KW / self.S_H

            self.S_OH_cons = pyo.Constraint(
                rule=S_OH_rule,
                doc="constraint concentration of pro-",
            )

        def S_H_rule(self):
            return (
                self.state_ref.conc_mol_phase_comp["Liq", "S_cat"]
                + self.conc_mol_nh4
                + self.S_H
                - self.conc_mol_hco3
                - self.conc_mol_ac / 64
                - self.conc_mol_pro / 112
                - self.conc_mol_bu / 160
                - self.conc_mol_va / 208
                - self.S_OH
                - self.state_ref.conc_mol_phase_comp["Liq", "S_an"]
                == 0
            )

            self.S_H_cons = pyo.Constraint(
                rule=S_H_rule,
                doc="constraint concentration of pro-",
            )

        def concentration_of_pro_rule(self):
            return (
                self.conc_mol_pro
                == self.params.K_a_pro
                ** self.state_ref.conc_mol_phase_comp["Liq", "S_pro"]
                / (self.state_ref.conc_mol_phase_comp["Liq", "S_pro"] + self.S_H)
            )

            self.concentration_of_pro = pyo.Constraint(
                rule=concentration_of_pro_rule,
                doc="constraint concentration of pro-",
            )

        def rule_pH(self):
            return -pyo.log10(self.S_H / pyo.units.kmol * pyo.units.m**3)

        self.pH = pyo.Expression(rule=rule_pH, doc="pH of solution")

        def rule_I_IN_lim(self):
            return 1 / (
                1 + self.params.K_S_IN / (self.state_ref.conc_mass_comp["S_IN"] / mw_n)
            )

        self.I_IN_lim = pyo.Expression(
            rule=rule_I_IN_lim,
            doc="Inhibition function related to secondary substrate; inhibit uptake when inorganic nitrogen S_IN~ 0",
        )

        def rule_I_h2_fa(self):
            return 1 / (
                1 + self.state_ref.conc_mass_comp["S_h2"] / self.params.K_I_h2_fa
            )

        self.I_h2_fa = pyo.Expression(
            rule=rule_I_h2_fa,
            doc="hydrogen inhibition attributed to long chain fatty acids",
        )

        def rule_I_h2_c4(self):
            return 1 / (
                1 + self.state_ref.conc_mass_comp["S_h2"] / self.params.K_I_h2_c4
            )

        self.I_h2_c4 = pyo.Expression(
            rule=rule_I_h2_c4,
            doc="hydrogen inhibition attributed to valerate and butyrate uptake",
        )

        def rule_I_h2_pro(self):
            return 1 / (
                1 + self.state_ref.conc_mass_comp["S_h2"] / self.params.K_I_h2_pro
            )

        self.I_h2_pro = pyo.Expression(
            rule=rule_I_h2_pro,
            doc="hydrogen inhibition attributed to propionate uptake",
        )

        def rule_I_nh3(self):
            return 1 / (1 + self.conc_mol_nh3 / self.params.K_I_nh3)

        self.I_nh3 = pyo.Expression(
            rule=rule_I_nh3, doc="ammonia inibition attributed to acetate uptake"
        )

        def rule_I_pH_aa(self):
            return pyo.Expr_if(
                self.pH > self.params.pH_UL_aa,
                1,
                pyo.exp(
                    -3
                    * (
                        (self.pH - self.params.pH_UL_aa)
                        / (self.params.pH_UL_aa - self.params.pH_LL_aa)
                    )
                    ** 2
                ),
            )

        self.I_pH_aa = pyo.Expression(
            rule=rule_I_pH_aa,
            doc="pH inhibition of amino-acid-utilizing microorganisms",
        )

        def rule_I_pH_ac(self):
            return pyo.Expr_if(
                self.pH > self.params.pH_UL_ac,
                1,
                pyo.exp(
                    -3
                    * (
                        (self.pH - self.params.pH_UL_ac)
                        / (self.params.pH_UL_ac - self.params.pH_LL_ac)
                    )
                    ** 2
                ),
            )

        self.I_pH_ac = pyo.Expression(
            rule=rule_I_pH_ac, doc="pH inhibition of acetate-utilizing microorganisms"
        )

        def rule_I_pH_h2(self):
            return pyo.Expr_if(
                self.pH > self.params.pH_UL_h2,
                1,
                pyo.exp(
                    -3
                    * (
                        (self.pH - self.params.pH_UL_h2)
                        / (self.params.pH_UL_h2 - self.params.pH_LL_h2)
                    )
                    ** 2
                ),
            )

        self.I_pH_h2 = pyo.Expression(
            rule=rule_I_pH_h2, doc="pH inhibition of hydrogen-utilizing microorganisms"
        )

        # @classmethod
        # def _I(self):
        def rule_I(self, r):
            if r == "R5" or r == "R6":
                return self.I_pH_aa * self.I_IN_lim
            elif r == "R7":
                return self.I_pH_aa * self.I_IN_lim * self.I_h2_fa
            elif r == "R8" or r == "R9":
                return self.I_pH_aa * self.I_IN_lim * self.I_h2_c4
            elif r == "R10":
                return self.I_pH_aa * self.I_IN_lim * self.I_h2_pro
            elif r == "R11":
                return self.I_pH_ac * self.I_IN_lim * self.I_nh3
            elif r == "R12":
                return self.I_pH_h2 * self.I_IN_lim
            else:
                raise BurntToast()

        self.I = pyo.Expression(
            [f"R{i}" for i in range(5, 13)],
            rule=rule_I,
            doc="Process inhibition functions",
        )

        try:

            def rate_expression_rule(b, r):
                if r == "R1":
                    # R1:  Disintegration
                    return b.reaction_rate[r] == pyo.units.convert(
                        b.params.k_dis * b.conc_mass_comp_ref["X_c"],
                        to_units=pyo.units.kg / pyo.units.m**3 / pyo.units.s,
                    )
                elif r == "R2":
                    # R2: Hydrolysis of carbohydrates
                    return b.reaction_rate[r] == pyo.units.convert(
                        b.params.k_hyd_ch * b.conc_mass_comp_ref["X_ch"],
                        to_units=pyo.units.kg / pyo.units.m**3 / pyo.units.s,
                    )
                elif r == "R3":
                    # R3: Hydrolysis of proteins
                    return b.reaction_rate[r] == pyo.units.convert(
                        b.params.k_hyd_pr * b.conc_mass_comp_ref["X_pr"],
                        to_units=pyo.units.kg / pyo.units.m**3 / pyo.units.s,
                    )
                elif r == "R4":
                    # R4: Hydrolysis of lipids
                    return b.reaction_rate[r] == pyo.units.convert(
                        b.params.k_hyd_li * b.conc_mass_comp_ref["X_li"],
                        to_units=pyo.units.kg / pyo.units.m**3 / pyo.units.s,
                    )
                elif r == "R5":
                    # R5: Uptake of sugars
                    return b.reaction_rate[r] == pyo.units.convert(
                        b.params.k_m_su
                        * b.conc_mass_comp_ref["S_su"]
                        / (b.params.K_S_su + b.conc_mass_comp_ref["S_su"])
                        * b.conc_mass_comp_ref["X_su"]
                        * b.I[r],
                        to_units=pyo.units.kg / pyo.units.m**3 / pyo.units.s,
                    )
                elif r == "R6":
                    # R6: Uptake of amino acids
                    return b.reaction_rate[r] == pyo.units.convert(
                        b.params.k_m_aa
                        * b.conc_mass_comp_ref["S_aa"]
                        / (b.params.K_S_aa + b.conc_mass_comp_ref["S_aa"])
                        * b.conc_mass_comp_ref["X_aa"]
                        * b.I[r],
                        to_units=pyo.units.kg / pyo.units.m**3 / pyo.units.s,
                    )
                elif r == "R7":
                    # R7: Uptake of long chain fatty acids (LCFAs)
                    return b.reaction_rate[r] == pyo.units.convert(
                        b.params.k_m_fa
                        * b.conc_mass_comp_ref["S_fa"]
                        / (b.params.K_S_fa + b.conc_mass_comp_ref["S_fa"])
                        * b.conc_mass_comp_ref["X_fa"]
                        * b.I[r],
                        to_units=pyo.units.kg / pyo.units.m**3 / pyo.units.s,
                    )
                elif r == "R8":
                    # R8: Uptake of valerate
                    return b.reaction_rate[r] == pyo.units.convert(
                        b.params.k_m_c4
                        * b.conc_mass_comp_ref["S_va"]
                        / (b.params.K_S_c4 + b.conc_mass_comp_ref["S_va"])
                        * b.conc_mass_comp_ref["X_c4"]
                        * b.conc_mass_comp_ref["S_va"]
                        / (
                            b.conc_mass_comp_ref["S_bu"] + b.conc_mass_comp_ref["S_va"]
                        )  # TODO: consider adding eps value to this denominator to avoid division by zero (or reformulate)
                        * b.I[r],
                        to_units=pyo.units.kg / pyo.units.m**3 / pyo.units.s,
                    )
                elif r == "R9":
                    # R9:  Uptake of butyrate
                    return b.reaction_rate[r] == pyo.units.convert(
                        b.params.k_m_c4
                        * b.conc_mass_comp_ref["S_bu"]
                        / (b.params.K_S_c4 + b.conc_mass_comp_ref["S_bu"])
                        * b.conc_mass_comp_ref["X_c4"]
                        * b.conc_mass_comp_ref["S_bu"]
                        / (
                            b.conc_mass_comp_ref["S_bu"] + b.conc_mass_comp_ref["S_va"]
                        )  # TODO: consider adding eps value to this denominator to avoid division by zero (or reformulate)
                        * b.I[r],
                        to_units=pyo.units.kg / pyo.units.m**3 / pyo.units.s,
                    )
                elif r == "R10":
                    # R10: Uptake of propionate
                    return b.reaction_rate[r] == pyo.units.convert(
                        b.params.k_m_pro
                        * b.conc_mass_comp_ref["S_pro"]
                        / (b.params.K_S_pro + b.conc_mass_comp_ref["S_pro"])
                        * b.conc_mass_comp_ref["X_pro"]
                        * b.I[r],
                        to_units=pyo.units.kg / pyo.units.m**3 / pyo.units.s,
                    )
                elif r == "R11":
                    # R11: Uptake of acetate
                    return b.reaction_rate[r] == pyo.units.convert(
                        b.params.k_m_ac
                        * b.conc_mass_comp_ref["S_ac"]
                        / (b.params.K_S_ac + b.conc_mass_comp_ref["S_ac"])
                        * b.conc_mass_comp_ref["X_ac"]
                        * b.I[r],
                        to_units=pyo.units.kg / pyo.units.m**3 / pyo.units.s,
                    )
                elif r == "R12":
                    # R12: Uptake of hydrogen
                    return b.reaction_rate[r] == pyo.units.convert(
                        b.params.k_m_h2
                        * b.conc_mass_comp_ref["S_h2"]
                        / (b.params.K_S_h2 + b.conc_mass_comp_ref["S_h2"])
                        * b.conc_mass_comp_ref["X_h2"]
                        * b.I[r],
                        to_units=pyo.units.kg / pyo.units.m**3 / pyo.units.s,
                    )
                elif r == "R13":
                    # R13: Decay of X_su
                    return b.reaction_rate[r] == pyo.units.convert(
                        b.params.k_dec_X_su * b.conc_mass_comp_ref["X_su"],
                        to_units=pyo.units.kg / pyo.units.m**3 / pyo.units.s,
                    )
                elif r == "R14":
                    # R14: Decay of X_aa
                    return b.reaction_rate[r] == pyo.units.convert(
                        b.params.k_dec_X_aa * b.conc_mass_comp_ref["X_aa"],
                        to_units=pyo.units.kg / pyo.units.m**3 / pyo.units.s,
                    )
                elif r == "R15":
                    # R15: Decay of X_fa
                    return b.reaction_rate[r] == pyo.units.convert(
                        b.params.k_dec_X_fa * b.conc_mass_comp_ref["X_fa"],
                        to_units=pyo.units.kg / pyo.units.m**3 / pyo.units.s,
                    )
                elif r == "R16":
                    # R16: Decay of X_c4
                    return b.reaction_rate[r] == pyo.units.convert(
                        b.params.k_dec_X_c4 * b.conc_mass_comp_ref["X_c4"],
                        to_units=pyo.units.kg / pyo.units.m**3 / pyo.units.s,
                    )
                elif r == "R17":
                    # R17: Decay of X_pro
                    return b.reaction_rate[r] == pyo.units.convert(
                        b.params.k_dec_X_pro * b.conc_mass_comp_ref["X_pro"],
                        to_units=pyo.units.kg / pyo.units.m**3 / pyo.units.s,
                    )
                elif r == "R18":
                    # R18: Decay of X_ac
                    return b.reaction_rate[r] == pyo.units.convert(
                        b.params.k_dec_X_ac * b.conc_mass_comp_ref["X_ac"],
                        to_units=pyo.units.kg / pyo.units.m**3 / pyo.units.s,
                    )
                elif r == "R19":
                    # R19: Decay of X_h2
                    return b.reaction_rate[r] == pyo.units.convert(
                        b.params.k_dec_X_h2 * b.conc_mass_comp_ref["X_h2"],
                        to_units=pyo.units.kg / pyo.units.m**3 / pyo.units.s,
                    )
                else:
                    raise BurntToast()

            self.rate_expression = pyo.Constraint(
                self.params.rate_reaction_idx,
                rule=rate_expression_rule,
                doc="ADM1 rate expressions",
            )

        except AttributeError:
            # If constraint fails, clean up so that DAE can try again later
            self.del_component(self.reaction_rate)
            self.del_component(self.rate_expression)
            raise

    def get_reaction_rate_basis(b):
        return MaterialFlowBasis.mass