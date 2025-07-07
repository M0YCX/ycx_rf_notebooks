import math
from collections import defaultdict

# import ipywidgets as widgets
import pint

import plotly.io as pio
import schemdraw as schem
import schemdraw.elements as e

# import texttable as tt
# from eseries import E12, E24, E48, E96, erange
from IPython.display import HTML, display
from sympy import *

# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'

# Support rendering plots in github
pio.renderers.default = "jupyterlab+png"
schem.use("svg")
schem.config(fontsize=10)

ureg = pint.UnitRegistry()


class Cascode:
    # Boltzmann's constant
    k = 1.38 * 10**-23  # joules/kelvin

    # Magnitude of electric charge
    q = 1.6 * 10**-19  # coulomb

    def __init__(self):
        self.show_values = defaultdict(lambda: math.nan)
        self._solve()

    def calc(
        self,
        Vcc=12,
        delta_v1=0.61,
        delta_v2=0.62,
        beta1=100,
        beta2=200,
        R1=0,
        R2=0,
        R3=0,
        RS=0,
        RC=0,
        RE=0,
        temp_kelvin=290,
    ):
        self.Vcc = Vcc
        self.delta_v1 = delta_v1
        self.delta_v2 = delta_v2
        self.beta1 = beta1
        self.beta2 = beta2
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.RS = RS
        self.RC = RC
        self.RE = RE

        self.temp_kelvin = temp_kelvin
        self.thermal_voltage = self.k * self.temp_kelvin / self.q

        # To be calculated
        self.Ibias = 0
        self.Ic = 0
        self.Ib1 = 0
        self.Ib2 = 0
        self.Ic2 = 0
        self.Ie = 0

        self.Vb1 = 0
        self.Vb2 = 0
        self.Vc1 = 0
        self.Vc2 = 0
        self.Vcp2 = 0
        self.Ve1 = 0

        # Run solutions with provided component values:-
        values = dict(
            V_cc=self.Vcc,
            Delta_v1=self.delta_v1,
            Delta_v2=self.delta_v2,
            beta_1=self.beta1,
            beta_2=self.beta2,
            R_1=self.R1,
            R_2=self.R2,
            R_3=self.R3,
            R_S=self.RS,
            R_C=self.RC,
            R_E=self.RE,
        )
        self.show_values = defaultdict(lambda: math.nan)

        for k in values.keys():
            self.show_values[k] = float(values[k])

        for s in self.solve4:
            if s in self.res:
                ans = self.res[s].subs(values)
                # display(s, ans)

                try:
                    # str_val = f"{float(ans):.3f}"
                    self.show_values[f"{s}"] = float(ans)
                except TypeError as e:
                    print(f"{s}: {e}")

        # Calc remaining fields for schematic and validations:-
        self.show_values["V_c2"] = (
            self.show_values["V_cp2"]
            - self.show_values["I_c2"] * self.show_values["R_C"]
        )
        self.show_values["V_c1"] = (
            self.show_values["V_b2"] - self.show_values["Delta_v2"]
        )
        self.show_values["KCL_A"] = round(
            self.show_values["I_c"]
            - (self.show_values["I_bias"] + self.show_values["I_c2"]),
            17,
        )
        self.show_values["KCL_B"] = round(
            self.show_values["I_bias"]
            - (self.show_values["I_r2"] + self.show_values["I_b2"]),
            17,
        )
        self.show_values["KCL_C"] = round(
            self.show_values["I_r2"]
            - (self.show_values["I_r3"] + self.show_values["I_b1"]),
            17,
        )
        self.show_values["KCL_ie"] = round(
            (
                -(
                    self.show_values["I_c2"]
                    + self.show_values["I_b1"]
                    + self.show_values["I_b2"]
                )
                + self.show_values["I_e1"]
            ),
            17,
        )

    def __repr__(self):
        return f"Cascode(\n" + f"\n)"

    def print(self):
        print(self)
        return self

    def analysis(self, system_eqs=True, solution_eqs=True):
        if system_eqs:
            display(HTML(f"<h2>System of Simultaneous Equations:-</h2>"))
            for eq in self.eqs:
                display(eq)

        if solution_eqs:
            display(HTML(f"<h2>Solutions:-</h2>"))
            for s in self.solutions:
                display(s)

    def draw(self):
        d = schem.Drawing()
        d.config(fontsize=12)
        d += e.Dot(open=True).label("i/p")
        d += e.Line().length(4)
        d += (
            e.Line()
            .length(2)
            .label(
                "$I_{b1}$=" + f"{(self.show_values['I_b1']*ureg.amperes):.3f~#P}",
                color="red",
            )
            .label(
                "$V_{b1}$=" + f"{(self.show_values['V_b1']*ureg.volts):.3f~#P}",
                color="red",
                loc="bot",
            )
        )
        d += (Q1 := e.BjtNpn()).label(
            f"$Q_1$\n$\\beta_1$:{(self.show_values['beta_1']):.1f}\n"
            + "$\\Delta_{v1}$"
            + f":{(self.show_values['Delta_v1']*ureg.volts):.1f~#P}",
            color="blue",
        )

        d += (
            e.Line()
            .at(Q1.emitter)
            .down()
            .length(0.5)
            .label(
                "$V_{e1}$" + f"\n{((self.show_values['V_e1'])*ureg.volts):.3f~#P}",
                color="red",
                loc="bot",
            )
        )

        d += (
            e.Resistor()
            # .at(Q1.emitter)
            .down()
            .label(
                f"$R_E$\n{(self.show_values['R_E']*ureg.ohm):.1f~#P}",
                color="blue",
            )
            .label(
                "$I_{e1}$"
                + f"\n{(self.show_values['I_e1']*ureg.amperes):.3f~#P}"
                + f"\n err:{(self.show_values['KCL_ie']*ureg.amperes):.1f~#P}",
                color="red",
                loc="bot",
            )
            .length(2.3)
        )
        d += e.Ground()

        d += (
            e.Line()
            .length(1.7)
            .at(Q1.collector)
            .up()
            .label(
                "$V_{c1}$" + f"\n{((self.show_values['V_c1'])*ureg.volts):.3f~#P}",
                color="red",
                loc="bot",
            )
        )

        d += (
            (Q2 := e.BjtNpn())
            .anchor("emitter")
            .theta(0)
            .label(
                f"$Q_2$\n$\\beta_2$:{(self.show_values['beta_2']):.3f}\n"
                + "$\\Delta_{v2}$"
                + f":{(self.show_values['Delta_v2']*ureg.volts):.1f~#P}",
                color="blue",
            )
        )

        d += e.Line().at(Q2.collector).up().length(1)
        d += e.Dot().label(
            "$V_{c2}$" + f"\n{((self.show_values['V_c2'])*ureg.volts):.3f~#P}",
            color="red",
            loc="left",
        )
        d.push()
        d += e.Line().right().length(4)
        d += e.Dot(open=True).label("o/p")
        d.pop()
        d += (
            e.Resistor()
            .label(
                f"$R_C$\n{(self.show_values['R_C']*ureg.ohm):.1f~#P}",
                color="blue",
            )
            .label(
                "$I_{c2}$" + f"\n{(self.show_values['I_c2']*ureg.amperes):.3f~#P}",
                loc="bot",
                color="red",
            )
        )
        d += e.Dot().label(
            "$V^{\\prime}_{c2}$"
            + f"\n{((self.show_values['V_cp2'])*ureg.volts):.3f~#P}",
            color="red",
            loc="right",
        )
        d.push()
        d += (
            e.Annotate(th1=45)
            .delta(dx=-0.5, dy=0.5)
            .label(f"$A$ err:{(self.show_values['KCL_A']*ureg.amperes):.1f~#P}")
            .color("red")
        )
        d.pop()
        d.push()
        # d.push()
        d += e.Resistor().label(
            f"$R_S$\n{(self.show_values['R_S']*ureg.ohm):.1f~#P}", color="blue"
        )
        # d.pop()
        d += (
            e.Line()
            .right()
            .label(
                f"$I_c$\n{(self.show_values['I_c']*ureg.amperes):.3f~#P}", color="red"
            )
        )
        d += e.Dot(open=True).label(
            "$V_{cc}$\n" + f"{(self.show_values['V_cc']*ureg.volts):+.1f~#P}",
            color="blue",
        )
        d.pop()

        d += (
            e.Line()
            .left()
            .length(3.5)
            .label(
                "$I_{bias}$\n" + f"{(self.show_values['I_bias']*ureg.amperes):.3f~#P}",
                color="red",
                loc="bot",
            )
        )
        d += (
            e.Resistor()
            .down()
            .length(4.75)
            .label(
                f"$R_1$\n{(self.show_values['R_1']*ureg.ohm):.2f~#P}",
                color="blue",
            )
            .label(
                "$I_{R1}$\n" + f"{(self.show_values['I_r1']*ureg.amperes):.3f~#P}",
                color="red",
                loc="bot",
            )
        )
        d += (
            e.Resistor()
            .down()
            .length(3.05)
            .label(
                f"$R_2$\n{(self.show_values['R_2']*ureg.ohm):.2f~#P}",
                color="blue",
            )
            .label(
                "$I_{R2}$\n" + f"{(self.show_values['I_r2']*ureg.amperes):.3f~#P}",
                color="red",
                loc="bot",
            )
        )
        d += e.Dot()
        d.push()
        d += (
            e.Annotate(th1=45)
            .delta(dx=-0.5, dy=0.5)
            .label(f"$C$ err:{(self.show_values['KCL_C']*ureg.amperes):.1f~#P}")
            .color("red")
        )
        d.pop()
        d += (
            e.Resistor()
            .down()
            .length(3.5)
            .label(
                f"$R_3$\n{(self.show_values['R_3']*ureg.ohm):.2f~#P}",
                color="blue",
            )
            .label(
                "$I_{R3}$\n" + f"{(self.show_values['I_r3']*ureg.amperes):.3f~#P}",
                color="red",
                loc="bot",
            )
        )
        d += e.Ground()

        d += (
            e.Line()
            .left()
            .at(Q2.base)
            .length(2)
            .label(
                "$I_{b2}$=" + f"{(self.show_values['I_b2']*ureg.amperes):.3f~#P}",
                color="red",
            )
            .label(
                "$V_{b2}$=" + f"{(self.show_values['V_b2']*ureg.volts):.3f~#P}",
                color="red",
                loc="bot",
            )
        )
        d += e.Line().length(0.75)
        d += e.Dot()
        d.push()
        d += (
            e.Annotate(th1=45)
            .delta(dx=-0.5, dy=0.5)
            .label(f"$B$ err:{(self.show_values['KCL_B']*ureg.amperes):.1f~#P}")
            .color("red")
        )
        d.pop()
        d += e.Line().length(2).color("grey")
        d += e.Capacitor().color("grey").down().length(1)
        d += e.Ground().color("grey")

        display(d.draw())

    def _solve(self):
        # SymPy symbols:-
        (
            V_cc,
            Delta_v1,
            Delta_v2,
            beta_1,
            beta_2,
            R_1,
            R_2,
            R_3,
            R_S,
            R_C,
            R_E,
        ) = symbols("V_cc Delta_v1 Delta_v2 beta_1 beta_2 R_1 R_2 R_3 R_S R_C R_E")
        (
            I_bias,
            I_c,
            I_b1,
            I_b2,
            I_c2,
            I_e1,
            I_e2,
            V_b1,
            V_b2,
            V_c1,
            V_c2,
            V_cp2,
            V_e1,
            I_r1,
            I_r2,
            I_r3,
            # alpha,
        ) = symbols(
            "I_bias I_c I_b1 I_b2 I_c2 I_e1 I_e2 V_b1 V_b2 V_c1 V_c2 V_cp2 V_e1 I_r1 I_r2 I_r3"  # alpha
        )

        # System of simultaneous equations
        # fmt:off
        self.eqs = (
            # A
            Eq(
                #  ______I_c___________ =
                (V_cc - V_cp2) / R_S,
                # ______________I_c2___________________ + _____I_bias_________
                I_b1 * beta_1 * (beta_2 / (beta_2 + 1)) + (V_cp2 - V_b2) / R_1,
            ),
            # B
            #  ______I_bias_______ = ___________I_b2_____________ + ______I_R2______
            Eq((V_cp2 - V_b2) / R_1, I_b1 * beta_1 / (beta_2 + 1) + (V_b2 - V_b1) / R_2),
            # C
            #  ______ I_R2_______ = _I_b1_ + __I_R3__
            Eq((V_b2 - V_b1) / R_2, I_b1 + V_b1 / R_3),
            #  _________I_e1________   from I_b1
            Eq((V_b1 - Delta_v1) / R_E, I_b1 * (beta_1 + 1)),
            #
            # V_b2 from bias divider
            # Eq(V_b2, V_cp2 * (R_2 + R_3) / (R_1 + R_2 + R_3)),   wont solve
            # I_c
            Eq(I_c, (V_cc - V_cp2) / R_S),
            # I_e1
            Eq(I_e1, (V_b1 - Delta_v1) / R_E),
            # I_bias
            Eq(I_bias, (V_cp2 - V_b2) / R_1),
            # I_r*
            Eq(I_r1, I_bias),
            Eq(I_r2, (V_b2 - V_b1) / R_2),
            Eq(I_r3, V_b1 / R_3),
            # I_b
            Eq(I_b2, I_b1 * beta_1 / (beta_2 + 1)),
            # V_e1
            Eq(V_e1, I_e1 * R_E),
            # V_c1
            # Eq(V_c1, V_b2 - Delta_v2), wont solve, see calc() above
            # I_c2
            Eq(I_c2, I_b1 * beta_1 * (beta_2 / (beta_2 + 1))),
            # V_c2
            # Eq(V_c2, V_cp2 - I_c2 * R_C ),   wont solve, see calc() above
        )
        # fmt:on
        # for eq in self.eqs:
        #     display(eq)
        self.solve4 = (
            V_cp2,
            I_b1,
            I_b2,
            V_b1,
            V_b2,
            I_c,
            I_bias,
            I_r1,
            I_r2,
            I_r3,
            I_c2,
            I_e1,
            # I_e2,
            V_e1,
        )

        r = solve(self.eqs, self.solve4, minimal=False, dict=True)
        if len(r) == 0:
            raise ValueError("solve() failed to find any solutions...")
        self.res = r[0]
        self.solutions = []
        for s in self.solve4:
            if s in self.res:
                self.solutions.append(Eq(s, self.res[s]))
            else:
                self.solutions.append(f"{s}: NOT FOUND")
