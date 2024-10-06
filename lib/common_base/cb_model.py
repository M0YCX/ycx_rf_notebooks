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
from ycx_complex_numbers import Complex, Neta, NetY, NetZ, Y, Z

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
    VA=80,
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

    RO = VA/Ie

    # Model the transistor's **intrinsic** elements first as a common-emitter, then use a conversion to common-base
    beta = Complex(B0 / (1 + (1j * B0 * F) / FT))
    alpha = beta / (beta + 1)

    jw = 1j * 2 * math.pi * F

    # base spreading resistor as ABCD matrix for cascading below
    Rbp_A = Neta(a11=1, a12=Rbp, a21=0, a22=1)

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
    ATRin = Neta(a11=Nin, a12=0, a21=0, a22=1 / Nin)
    ATRout = Neta(a11=Nout, a12=0, a21=0, a22=1 / Nout)

    # Note: this is the same as adding the Y matrix of Ccb to the simple transistor model (ie in parallel)
    # e.g. Ye = Ytrans + Yccb
    # y11e = Y(1 / (Ze * (beta + 1)) + (jw * Ccb))
    # y12e = Y(0 - (jw * Ccb))
    # y21e = Y(beta / (Ze * (beta + 1)) - (jw * Ccb))
    # y22e = Y(0 + (jw * Ccb))
    y11e = Y(1 / (Ze * (beta + 1)) + 1/RO + (jw * Ccb))
    y12e = Y(-1/RO - (jw * Ccb))
    y21e = Y(beta / (Ze * (beta + 1)) - 1/RO - (jw * Ccb))
    y22e = Y(1/RO + (jw * Ccb))
    Ye = NetY(y11=y11e, y12=y12e, y21=y21e, y22=y22e)

    # Cascade the base spreading resistance to the hybrid-pi amplifier
    Ae = Ye.to_a()
    A1 = Rbp_A @ Ae

    Y1 = A1.to_Y()

    # Get interim output impedance
    # izout = Z(1 / Y1.in_out(ys=1 / ZS, yl=1 / ZL)["Yout"])
    # Y1 = (Y1.to_a() * ATR1).to_Y()

    # Y1 = Ye

    # Convert common-emitter matrix to common-base
    Y1 = Y1.exchange_to_cb(from_config="ce")
    # print(f"Y1={Y1}")

    # Cascade:-
    Y1 = (ATRin @ YRE.to_a() @ Y1.to_a() @ YRC.to_a() @ ATRout).to_Y()
    # print(f"Y cascaded={Y1}")
    S1 = Y1.to_S()

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
    InVSWR = (1 + abs(GammaIn)) / (1 - abs(GammaIn))

    # Calc output reflection coefficient & return loss
    GammaOut = (zout - ZL) / (zout + ZL)
    OutRetLoss = -20 * math.log10(abs(GammaOut))
    OutVSWR = (1 + abs(GammaOut)) / (1 - abs(GammaOut))

    d = schem.Drawing()

    d += e.GroundChassis()
    d += e.SourceSin().label("$V_{in}$", loc="top").up().length(2)
    d += e.Resistor().label("$Z_S$" + f"\n{ZS}", color="blue").up().length(2)

    d += e.Line().right().length(6)

    d += (
        TRin := e.Transformer(t1=int(Nin), t2=1)
        .right()
        .label(f"{Nin}:1t\n$z${Nin**2}:1", color="blue")
        # .label(
        #     "${Z_{in}$"
        #     + f"\n{zin:.3f~S}\nReturn Loss={(InRetLoss * ureg.decibel):.3f~#P}",
        #     loc="left",
        #     color="red",
        # )
        .flip()
    )
    d.push()
    d += (
        e.Gap()
        .down()
        .length(1)
        # .label(
        #     "${Z_{in}$"
        #     + f"\n{zin:.3f~S}\nReturn Loss={(InRetLoss * ureg.decibel):.3f~#P}",
        #     loc="left",
        #     halign="right",
        #     color="red",
        # )
    )
    d += (
        e.Gap()
        .left()
        .length(0.15)
        .label(
            "${Z_{in}$"
            + f"\n{zin:.3f~S}\nReturn Loss={(InRetLoss * ureg.decibel):.3f~#P}\nvswr={(InVSWR):.2f}",
            loc="left",
            halign="right",
            color="red",
        )
    )
    d.pop()
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
        e.Resistor(color="grey")
        .label("$r_{e}$" + f"\n{(re * ureg.ohms):.1f~#P}", color="red")
        .up()
    )
    d += e.Dot(color="grey")
    d.push()
    d += e.Line(color="grey").left()
    d += (
        e.Resistor(color="grey")
        .label(f"$r^`_b$\n{(Rbp * ureg.ohms):.1f~#P}", color="blue")
        .up()
    )
    d += e.Line(color="grey").left().length(1)
    d += e.Dot(open=True).label("b", color="grey", loc="top")
    d += e.Line().left().length(1)
    d += e.Line().down().length(1)
    d += e.GroundChassis()
    d.pop()
    d += e.Line(color="grey").right()
    d += e.SourceI(color="grey").reverse().up().label("$\\beta I_b$", loc="bot")
    d += e.Dot(color="grey")
    d.push()
    d += e.Line(color="grey").left(2)
    d += (
        e.Capacitor(color="grey")
        .label("$C_{cb}$" + f"\n{(Ccb * ureg.farads):.1f~#P}", color="blue")
        .down()
    )
    d += e.Dot(color="grey")
    d.pop()
    d += e.Line(color="grey").right().length(2)
    d += e.Dot(color="grey")
    d.push()
    d += e.Resistor(color="grey").label(f"$R_O$\n{(RO * ureg.ohms):.1f~#P}", color="red", loc="bot").down()
    d += e.Line(color="grey").left().length(2)
    d += e.Dot(color="grey").label(
        f"\n$\\beta$={beta:.1f}\n$\\alpha$={alpha:.2f}",
        color="red",
        loc="bottom",
    )
    d.pop()
    d += e.Line(color="grey").right().length(1)
    d += e.Dot(open=True).label("c", color="grey", loc="top")
    d += e.Line().length(2)
    d += e.Dot()
    d.push()
    d += e.Resistor().label(f"$R_C$\n{(RC * ureg.ohms):.1f~#P}", color="blue", loc="bot").down()
    d += e.GroundChassis()
    d.pop()
    d += e.Line().length(3)

    d += e.Line().down().length(1)
    d += (
        TRout := e.Transformer(t1=int(Nout), t2=1)
        .right()
        .label(f"{Nout}t:1\n$z${Nout**2}:1", color="blue")
        .flip()
    )
    # d.push()
    # d += e.Gap().up().length(1)
    # d += (
    #     e.Gap()
    #     .right()
    #     .length(1)
    #     .label(
    #         "${Z_{out}$"
    #         + f"\n{zout:.3f~S}\nReturn Loss={(OutRetLoss * ureg.decibel):.3f~#P}\nvswr={(OutVSWR):.2f}",
    #         loc="right",
    #         halign="left",
    #         color="red",
    #     )
    # )
    # d.pop()
    d += e.Line().at(TRout.p1).length(0.5)
    d += e.GroundChassis()

    d += e.Line().at(TRout.s1).length(0.5)
    d += e.GroundChassis()

    d += e.Gap().down().length(1)
    d += (
        e.Gap()
        .right()
        .length(0.5)
        .label(
            "${Z_{out}$"
            + f"\n{zout:.3f~S}\nReturn Loss={(OutRetLoss * ureg.decibel):.3f~#P}\nvswr={(OutVSWR):.2f}",
            loc="right",
            halign="left",
            color="red",
        )
    )

    d += e.Line().at(TRout.s2).right().length(5)

    d += e.Resistor().label("$Z_L$" + f"\n{ZL}", loc="bot", color="blue").down()
    d += e.GroundChassis().label(
        "Gain\n$G_t$" + f"={(Gt_db * ureg.decibel):.4f~#P}",
        color="red",
        loc="right",
    )

    display(d)

    print(f"Y:{Y1}")
    print(f"S:{S1}")
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
    VA=widgets.FloatText(
        value=80.0,
        description="$V_A$ Early Voltage",
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
