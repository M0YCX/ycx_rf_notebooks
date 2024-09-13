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


def cb_model(
    F_mhz=10,
    FT_mhz=300,
    B0=100,
    Ie_mA=10,
    Rbp=10.0,
    Ccb_pF=5,
    ZS_real=50,
    ZS_imag=0,
    ZL_real=50,
    ZL_imag=0,
    RE=300,
    RC=200,
    Nin=4,
    Nout=2,
):
    Ccb = Ccb_pF * 10**-12
    F = F_mhz * 10**6
    FT = FT_mhz * 10**6
    Ie = Ie_mA * 10**-3

    ZS = ZS_real + (1j * ZS_imag)
    ZL = ZL_real + (1j * ZL_imag)

    YS = 1 / ZS
    YL = 1 / ZL

    # Model the transistor's **intrinsic** elements first as a common-emitter, then use a conversion to common-base
    beta = Complex(B0 / (1 + (1j * B0 * F) / FT))
    alpha = beta / (beta + 1)

    jw = 1j * 2 * math.pi * F

    # base spreading resistor as ABCD matrix for cascading below
    Rbp_A = NetABCD(A=1, B=Rbp, C=0, D=1)

    # Emitter complex impedance
    re = 26 / Ie_mA
    Ze = Z(re)  # + (jw * Le)) - Le is extrinsic

    # Emitter bias shunt resistance RE
    YRE = NetY(
        y11=1 / 0.0001, y12=-1 / 0.0001, y21=-1 / 0.0001, y22=1 / 0.0001 + 1 / RE
    )

    # Collector bias shunt resistance RC
    YRC = NetY(
        y11=1 / 0.0001, y12=-1 / 0.0001, y21=-1 / 0.0001, y22=1 / 0.0001 + 1 / RC
    )

    # input/output transformers N:1 as ABCD for cascade below
    ATRin = NetABCD(A=Nin, B=0, C=0, D=1 / Nin)
    ATRout = NetABCD(A=Nout, B=0, C=0, D=1 / Nout)

    # Note: this is the same as adding the Y matrix of Ccb to the simple transistor model (ie in parallel)
    # e.g. Ye = Ytrans + Yccb
    y11e = Y(1 / (Ze * (beta + 1)) + (jw * Ccb))
    y12e = Y(0 - (jw * Ccb))
    y21e = Y(beta / (Ze * (beta + 1)) - (jw * Ccb))
    y22e = Y(0 + (jw * Ccb))
    Ye = NetY(y11=y11e, y12=y12e, y21=y21e, y22=y22e)

    # Cascade the base spreading resistance to the hybrid-pi amplifier
    Ae = Ye.to_ABCD()
    A1 = Rbp_A * Ae

    Y1 = A1.to_Y()

    # Get interim output impedance
    # izout = Z(1 / Y1.in_out(ys=1 / ZS, yl=1 / ZL)["Yout"])
    # Y1 = (Y1.to_ABCD() * ATR1).to_Y()

    # Y1 = Ye

    # Convert common-emitter matrix to common-base
    Y1 = Y1.exchange_to_cb(from_config="ce")
    # print(f"Y1={Y1}")

    # Cascade:-
    Y1 = (ATRin * YRE.to_ABCD() * Y1.to_ABCD() * YRC.to_ABCD() * ATRout).to_Y()
    # print(f"Y cascaded={Y1}")

    yio = Y1.in_out(ys=1 / ZS, yl=1 / ZL)
    zin = Z(1 / yio["Yin"])
    zout = Z(1 / yio["Yout"])

    # Calc Transducer Gain
    Gt = (4 * YS.real * YL.real * abs(Y1.y21) ** 2) / (
        abs((YS + Y1.y11) * (YL + Y1.y22) - Y1.y12 * Y1.y21) ** 2
    )
    Gt_db = 10 * math.log10(Gt)

    # Calc input reflection coefficient & return loss
    GammaIn = (zin - ZS) / (zin + ZS)
    InRetLoss = -20 * math.log10(abs(GammaIn))

    # Calc output reflection coefficient & return loss
    GammaOut = (zout - ZS) / (zout + ZS)
    OutRetLoss = -20 * math.log10(abs(GammaOut))

    d = schem.Drawing()

    d += e.GroundChassis()
    d += e.SourceSin().label("$V_{in}$", loc="top").up().length(2)
    d += e.Resistor().label("$Z_S$" + f"\n{ZS}", color="blue").up().length(2)

    d += e.Line().right().length(5)

    d += (
        TRin := e.Transformer(t1=int(Nin), t2=1)
        .right()
        .label(f"{Nin:.1f}:1\n$Z${Nin**2}:1", color="blue")
        .label(
            "${Z_{in}$"
            + f"\n{zin:.3f~S}\nReturn Loss={(InRetLoss * ureg.decibel):.3f~#P}",
            loc="left",
            color="red",
        )
        .flip()
    )
    d += e.Line().at(TRin.p1).length(0.5).down()
    d += e.GroundChassis()

    d += e.Line().at(TRin.s1).length(0.5).down()
    d += e.GroundChassis()

    d += e.Line().at(TRin.s2).right().length(3)

    d += e.Dot()
    d.push()
    d += (
        e.Resistor()
        .down()
        .label("$R_E$" + f"\n{(RE * ureg.ohms):.1f~#P}", color="blue")
    )
    d += e.GroundChassis()
    d.pop()
    d += e.Line().up().length(1)
    d += e.Dot(open=True).label("e", color="grey", loc="right")
    d += (
        e.Resistor(ls="dashed")
        .label("$r_{e}$" + f"\n{(re * ureg.ohms):.1f~#P}", color="red")
        .up()
    )
    d += e.Dot()
    d.push()
    d += e.Line(ls="dashed").left()
    d += (
        e.Resistor(ls="dashed")
        .label(f"$r^`_b$\n{(Rbp * ureg.ohms):.1f~#P}", color="blue")
        .up()
    )
    d += e.Line(ls="dashed").left().length(2)
    d += e.Dot(open=True).label("b", color="grey", loc="top")
    d += e.Line().left().length(1)
    d += e.Line().down().length(1)
    d += e.GroundChassis()
    d.pop()
    d += e.Line(ls="dashed").right()
    d += e.SourceI(ls="dashed").reverse().up().label("$\\beta I_b$", loc="bot")
    d += e.Dot()
    d.push()
    d += e.Line(ls="dashed").left(2)
    d += (
        e.Capacitor(ls="dashed")
        .label("$C_{cb}$" + f"\n{(Ccb * ureg.farads):.1f~#P}", color="blue")
        .down()
    )
    d += e.Dot()
    d.pop()
    d += e.Line(ls="dashed").right().length(1)
    d += e.Dot(open=True).label("c", color="grey", loc="top")
    d += e.Line().length(2)
    d += e.Dot()
    d.push()
    d += e.Resistor().label(f"$R_C$\n{(RC * ureg.ohms):.1f~#P}", color="blue").down()
    d += e.GroundChassis()
    d.pop()
    d += e.Line().length(2)

    d += e.Line().down().length(1)
    d += (
        TRout := e.Transformer(t1=int(Nout), t2=1)
        .right()
        .label(f"{Nout:.1f}:1\n$Z${Nout**2}:1", color="blue")
        .label(
            "${Z_{out}$"
            + f"\n{zout:.3f~S}\nReturn Loss={(OutRetLoss * ureg.decibel):.3f~#P}",
            loc="right",
            color="red",
        )
        .flip()
    )

    d += e.Line().at(TRout.p1).length(0.5)
    d += e.GroundChassis()

    d += e.Line().at(TRout.s1).length(0.5)
    d += e.GroundChassis()

    d += e.Line().at(TRout.s2).right().length(5)

    d += e.Resistor().label("$Z_L$" + f"\n{ZL}", loc="bot", color="blue").down()
    d += e.GroundChassis().label(
        "Gain\n$G_t$" + f"={(Gt_db * ureg.decibel):.4f~#P}",
        color="red",
        loc="bot",
    )

    display(d)
    return d


res_series = E24
interactive_cb_model = interactive(
    cb_model,
    F_mhz=widgets.FloatText(
        value=1.0,
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
    Ccb_pF=widgets.FloatText(
        value=5.0,
        description="$C_{cb}$ pF",
        style=style,
        # layout=layout,
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
    RE=widgets.FloatText(
        value=300.0,
        description="$R_E$ Emitter bias",
        style=style,
        # layout=layout,
    ),
    RC=widgets.FloatText(
        value=200.0,
        description="$R_C$ Collector bias",
        style=style,
        # layout=layout,
    ),
    Nin=widgets.IntSlider(
        value=4,
        description="$N_{in}$",
        min=1,
        max=10,
        style=style,
        layout=layout,
    ),
    Nout=widgets.IntSlider(
        value=2,
        description="$N_{out}$",
        min=1,
        max=10,
        style=style,
        layout=layout,
    ),
)
