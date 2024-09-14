import math
# import sys

import ipywidgets as widgets

# import numpy as np
import pint

# import plotly.graph_objects as go
import plotly.io as pio
import schemdraw as schem
import schemdraw.elements as e
from eseries import E12, E24, E48, E96, erange
from IPython.display import display
from ipywidgets import Layout, interactive
from ycx_complex_numbers import Complex, NetABCD, NetY, NetZ, Y, Z

# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'


# Support rendering plots in github
pio.renderers.default = "jupyterlab+png"
schem.use("svg")

ureg = pint.UnitRegistry()
ureg = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
layout = Layout(width="auto")
style = {"description_width": "200px"}

###############################################################


def complex_fba(
    F_mhz=10,
    FT_mhz=300,
    B0=100,
    Ie_mA=10,
    Rbp=10.0,
    Re=6.8,
    Le_nH=0,
    Ccb_pF=5,
    Cf_pF=10000,
    Lf_nH=20,
    Rf=1000,
    ZS_real=50,
    ZS_imag=0,
    ZL_real=50,
    ZL_imag=0,
    N=2,
):
    Ccb = Ccb_pF * 10**-12
    Le = Le_nH * 10**-9
    F = F_mhz * 10**6
    FT = FT_mhz * 10**6
    Ie = Ie_mA * 10**-3

    Cf = Cf_pF * 10**-12
    Lf = Lf_nH * 10**-9

    ZS = ZS_real + (1j * ZS_imag)
    ZL = ZL_real + (1j * ZL_imag)

    YS = 1 / ZS
    YL = 1 / ZL

    beta = Complex(B0 / (1 + (1j * B0 * F) / FT))

    jw = 1j * 2 * math.pi * F

    # base spreading resistor as ABCD matrix for cascading below
    Rbp_A = NetABCD(A=1, B=Rbp, C=0, D=1)

    # Emitter complex impedance
    re = 26 / Ie_mA
    Ze = Z(re + Re + (jw * Le))

    # feedback network as an admittance for later adding in parallel
    Zf = Z(Rf + 1j * (2 * math.pi * F * Lf - 1 / (2 * math.pi * F * Cf)))
    Yf = NetY(y11=1 / Zf, y12=-1 / Zf, y21=-1 / Zf, y22=1 / Zf)

    # output transformer N:1 as ABCD for cascade below
    ATR1 = NetABCD(A=N, B=0, C=0, D=1 / N)

    # Note: this is the same as adding the Y matrix of Ccb to the simple transistor model (ie in parallel)
    y11e = Y(1 / (Ze * (beta + 1)) + (jw * Ccb))
    y12e = Y(0 - (jw * Ccb))
    y21e = Y(beta / (Ze * (beta + 1)) - (jw * Ccb))
    y22e = Y(0 + (jw * Ccb))
    Ye = NetY(y11=y11e, y12=y12e, y21=y21e, y22=y22e)

    # Cascade the base spreading resistance to the hybrid-pi amplifier
    Ae = Ye.to_ABCD()
    A1 = Rbp_A * Ae

    # Add feedback in parallel
    Y1 = A1.to_Y() + Yf

    # Get interim output impedance to match Wes's Zout
    izout = Z(1 / Y1.in_out(ys=1 / ZS, yl=1 / ZL)["Yout"])

    Y1 = (Y1.to_ABCD() * ATR1).to_Y()

    yio = Y1.in_out(ys=1 / ZS, yl=1 / ZL)
    zin = Z(1 / yio["Yin"])
    zout = Z(1 / yio["Yout"])

    # Calc Transducer Gain
    Gt = (4 * YS.real * YL.real * abs(Y1.y21) ** 2) / (
        abs((YS + Y1.y11) * (YL + Y1.y22) - Y1.y12 * Y1.y21) ** 2
    )
    Gt_db = 10 * math.log10(Gt)

    # Calc input reflection coefficient & return loss
    GammaIn = (zin-ZS) / (zin+ZS)
    InRetLoss = -20*math.log10(abs(GammaIn))

    # Calc output reflection coefficient & return loss
    GammaOut = (zout-ZS) / (zout+ZS)
    OutRetLoss = -20*math.log10(abs(GammaOut))

    d = schem.Drawing()
    d.push()
    d += e.Resistor().label("$Z_S$" + f"\n{ZS}", color="blue").down().length(2)
    d += e.SourceSin().down().length(2)
    d += e.GroundChassis().label("${Z_{in}$" + f"\n{zin:.3f~S}\nReturn Loss={(InRetLoss * ureg.decibel):.3f~#P}", loc="bot", color="red")
    d.pop()
    d += e.Line().right().length(2)
    d += e.Dot(open=True).label("b", color="grey", loc="bot")
    d += (
        e.Resistor(ls="dashed")
        .right()
        .label(f"$R^`_b$\n{(Rbp * ureg.ohms):.1f~#P}", color="blue")
    )
    d += e.Line().right().length(1.5)
    d += e.Dot().label(f"$I_e$={(Ie * ureg.ampere):.1f~#P}", color="blue", loc="right")
    d.push()
    d += (
        e.Resistor(ls="dashed")
        .label("$r_e$" + f"={(re * ureg.ohms):.1f~#P}", color="red", loc="bot")
        .down()
        .length(2)
    )
    d += e.Dot(open=True).label("e", loc="right", color="grey")
    d.push()
    d += e.Gap().length(5).right().label(f"$Z_e$={(Ze.c * ureg.ohms):.2}", color="red")
    d.pop()
    d += (
        e.Resistor()
        .label("$R_e$" + f"\n{(Re * ureg.ohms):.1f~#P}", color="blue", loc="bot")
        .length(2)
    )
    d += (
        e.Inductor()
        .label("$L_e$" + f"\n{(Le * ureg.henrys):.1f~#P}", color="blue", loc="bot")
        .length(2)
    )
    d += e.GroundChassis().label(
        f"\n$\\beta$={beta:.1f}@{(F*ureg.hertz):.1f~#P}",
        color="red",
        loc="bot",
    )

    d.pop()
    d += (
        e.SourceI()
        .length(2)
        .reverse()
        .label(
            "$\\beta i_b$",
            color="black",
            loc="bot",
        )
    )
    d += e.Dot(open=True).label("c", loc="right", color="grey")
    d.push()
    d += (
        e.Gap()
        .right()
        .length(3)
        .label(
            "(intermediate) ${Z_{out}$" + f"\n{izout:.3f~S}", loc="right", color="red"
        )
    )
    d.pop()
    d.push()
    d += e.Line(ls="dashed").left().length(1)
    d += (
        e.Capacitor(ls="dashed")
        .length(2)
        .label("$C_{cb}$" + f"\n{(Ccb * ureg.farads):.1f~#P}", color="blue")
        .down()
    )
    d += e.Dot()

    d.pop()
    d += e.Line().up().length(1)
    d += e.Dot()
    d.push()

    # feedback
    d += (
        e.Capacitor()
        .label(f"$C_f$\n{(Cf * ureg.farads):.1f~#P}", color="blue")
        .left()
        .length(2)
    )
    d += (
        e.Inductor()
        .label("$L_f$" + f"\n{(Lf * ureg.henrys):.1f~#P}", color="blue")
        .flip()
        .length(2)
    )
    d += e.Line().length(1)
    d += (
        e.Resistor()
        .label("$R_f$" + f"\n{(Rf * ureg.ohms):.1f~#P}", color="blue")
        .down()
        .length(3)
    )
    d += e.Dot()

    d.pop()
    d += e.Line().right().length(6)
    d += e.Line().down().length(1)
    d += (
        TR1 := e.Transformer(t1=N, t2=1)
        .right()
        .label(f"{N}t:1\n$z${N**2}:1", color="blue")
        .flip()
    )
    d.push()
    d += e.Gap().up().length(1)
    d += (
        e.Gap()
        .right()
        .length(1)
        .label(
            "${Z_{out}$"
            + f"\n{zout:.3f~S}\nReturn Loss={(OutRetLoss * ureg.decibel):.3f~#P}",
            loc="right",
            halign="left",
            color="red",
        )
    )
    d.pop()
    d += e.Line().at(TR1.p1).length(0.5)
    d += e.GroundChassis()

    d += e.Line().at(TR1.s1).length(0.5)
    d += e.GroundChassis()

    d += e.Line().at(TR1.s2).right().length(5)
    d += (
        e.Resistor()
        .label("$Z_L$" + f"\n{ZL}", loc="bot", color="blue")
        .down()
        .length(2.1)
    )
    d += e.GroundChassis().label(
        "Gain\n$G_t$" + f"={(Gt_db * ureg.decibel):.4f~#P}", color="red", loc="bot"
    )

    display(d)
    return d


res_series = E24
interactive_complex_fba = interactive(
    complex_fba,
    F_mhz=widgets.FloatText(
        value=10.0,
        description="$F$ MHz",
        style=style,
        # layout=layout,
    ),
    FT_mhz=widgets.FloatText(
        value=300.0,
        description="$F_T$ MHz",
        style=style,
        # layout=layout,
    ),
    B0=widgets.FloatText(
        value=100.0,
        description="$B0$",
        style=style,
        # layout=layout,
    ),
    Ie_mA=widgets.FloatText(
        value=10.0,
        description="$I_e$ mA",
        style=style,
        # layout=layout,
    ),
    Rbp=widgets.FloatText(
        value=10.0,
        description="$R^`_b$ base spreading resistance",
        style=style,
        # layout=layout,
    ),
    Re=widgets.SelectionSlider(
        value=6.8,
        description="$R_e$",
        options=[
            0.0,
        ]
        + list(erange(res_series, 1, 1000.0)),
        style=style,
        layout=layout,
    ),
    Le_nH=widgets.FloatText(
        value=10.0,
        description="$Le$ nH",
        style=style,
        # layout=layout,
    ),
    Ccb_pF=widgets.FloatText(
        value=5.0,
        description="$C_{cb}$ pF",
        style=style,
        # layout=layout,
    ),
    Cf_pF=widgets.SelectionSlider(
        value=10000.0,
        description="$C_f$ pF",
        options=list(erange(res_series, 1, 10000.0)),
        style=style,
        layout=layout,
    ),
    Lf_nH=widgets.SelectionSlider(
        value=20.0,
        options=list(erange(res_series, 1, 10000.0)),
        description="$Lf$ nH",
        style=style,
        layout=layout,
    ),
    Rf=widgets.SelectionSlider(
        value=1000,
        description="$R_f$",
        options=list(erange(res_series, 1, 100000.0)),
        style=style,
        layout=layout,
    ),
    ZS_real=widgets.FloatText(
        value=50,
        description="$Z_S$",
        style=style,
        # layout=layout,
    ),
    ZS_imag=widgets.FloatText(
        value=0,
        description="$Z_S$ j",
        style=style,
        # layout=layout,
    ),
    ZL_real=widgets.FloatText(
        value=50,
        description="$Z_L$",
        style=style,
        # layout=layout,
    ),
    ZL_imag=widgets.FloatText(
        value=0,
        description="$Z_L$ j",
        style=style,
        # layout=layout,
    ),
    N=widgets.IntSlider(
        value=2,
        description="$N$",
        min=1,
        max=10,
        style=style,
        layout=layout,
    ),
)
