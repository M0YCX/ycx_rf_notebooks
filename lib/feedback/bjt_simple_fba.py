# import math

from IPython.display import display
import ipywidgets as widgets
# import numpy as np
import pint
# import plotly.graph_objects as go
import plotly.io as pio
import schemdraw as schem
import schemdraw.elements as e
from eseries import E12, E24, E48, erange, find_nearest
from ipywidgets import Layout, interactive
# from schemdraw import dsp  # , flow

# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'

# Support rendering plots in github
pio.renderers.default = "jupyterlab+png"
schem.use("svg")

ureg = pint.UnitRegistry()
layout = Layout(width="auto")
style = {"description_width": "50px"}

###############################################################


def beta_at_F(F=10 * 10**6, FT=300 * 10**6, B0=100):
    """Calc Beta at given frequency (from Hybrid-pi model)"""
    return B0 / (1 + (1j * B0 * (F / FT)))


def simple_fba(
    Rs=50.0,
    RL=50.0,
    Rf=250.0,
    RD=8.7,
    Ie_mA=20,
    B0=30,
    FT_mhz=300,
    F_mhz=10,
    Vin=2,
):
    FT = FT_mhz * 10**6
    F = F_mhz * 10**6
    Ie = Ie_mA * 10**-3
    re = 26 / (Ie * 10**3)
    Re = RD + re
    # Beta = abs(beta_at_F(F=F, FT=FT, B0=B0))
    Beta = B0

    A = 1 / (Re * (Beta + 1)) + 1 / Rf + 1 / Rs
    B = 1 / Rf
    C = Beta / (Re * (Beta + 1)) - 1 / Rf
    D = 1 / Rf + 1 / RL
    D2 = 1 / Rf + 1 / (2 * RL)

    Vb = (Vin * D / Rs) / (B * C + A * D)
    Vb2 = (Vin * D2 / Rs) / (B * C + A * D2)

    VL = -Vb * C / D
    VL2 = -Vb2 * C / D2

    Rin = Rs / (Vin / Vb - 1)
    Rout = (2 * RL * (VL2 - VL)) / (2 * VL - VL2)

    # Av = VL / Vin

    # Zs = Rs
    # ZL = RL

    # Zi = (Zs * Av) / Av
    # Zo = Av / (Av / ZL)

    d = schem.Drawing()

    d += e.GroundChassis()
    d += (
        e.SourceV()
        .up()
        .length(3)
        .label("$V_{in}$" + f"\n{(Vin*ureg.volt):.1f~#P}", color="blue")
    )
    d += (
        e.Resistor()
        .label("$R_s$" + f"\n{(Rs*ureg.ohm):.1f~#P}", color="blue")
        .right()
        .length(3)
    )
    d += e.Capacitor().length(2)
    d += e.Dot()
    d.push()
    d += e.Resistor().down()
    d += e.GroundChassis()
    d.pop()
    d.push()
    d += e.Line().length(1.5).up()
    d += e.Dot()
    d += e.Resistor().up()
    d += e.Line().right().length(1.75)
    d += e.Dot()
    d.push()
    d += e.Capacitor().down().length(1)
    d += e.GroundChassis()
    d.pop()
    d += e.Line().length(2)
    d += e.Dot()

    d.pop()
    d += e.Line().right()
    d += (
        TR1 := e.BjtNpn(circle=True).label(
            f"$\\beta$={Beta:.1f}", color="red"
        )
    )

    d += e.Line().up().at(TR1.collector).length(0.8)
    d += e.Dot()
    d.push()
    d += (
        e.Resistor()
        .label("$R_f$" + f"\n{(Rf*ureg.ohm):.1f~#P}", color="blue")
        .left()
        .length(1.8)
    )
    d += e.Capacitor().length(2)
    d.pop()
    d.push()
    d += e.Inductor().label("RFC", loc="bot").up()
    d += e.Resistor().length(3)
    d += e.Dot(open=True).label("$+V_{CC}$")

    d.pop()
    d += e.Capacitor().right()
    d += (
        eRL := e.Resistor()
        .down()
        .label("$R_L$" + f"\n{(RL*ureg.ohm):.1f~#P}", color="blue", loc="bot")
        .length(2)
    )
    d += e.GroundChassis()

    d += (
        e.Resistor()
        .label("$R_D$" + f"\n{(RD*ureg.ohm):.1f~#P}", color="blue")
        .down()
        .at(TR1.emitter)
        .length(2)
    )
    d += e.Dot()
    d.push()
    d += e.Resistor().down().length(2)
    d += e.GroundChassis()
    d.pop()
    d += e.Line().right().length(2)
    d += e.Capacitor().down().length(2)
    d += e.GroundChassis()

    ########################
    # small-signal equiv:-
    d += e.Gap().at(eRL.end).right()
    d.push()
    d += (
        e.SourceV()
        .label("$V_{in}$" + f"\n{(Vin*ureg.volt):.1f~#P}", color="blue")
        .down()
        .reverse()
        .length(2.5)
    )
    d += e.GroundChassis()
    d.pop()
    d += (
        e.Resistor()
        .label("$R_s$" + f"\n{(Rs*ureg.ohm):.1f~#P}", color="blue")
        .right()
        .length(3)
    )
    d.push()

    d += (
        e.Gap()
        .down()
        .label(
            "$R_{in}$" + f"= {(Rin*ureg.ohm):.3~#P}",
            # + "\n$Z_i$"
            # + f"= {(Zi*ureg.ohm):.3~#P}",
            color="red",
        )
    )

    d.pop()
    d += e.Arrow().length(0.5)
    d += e.Line().length(1)
    d += e.Dot(open=True).label(
        "$V_b$" + f"\n{(Vb*ureg.volt):.1f~#P}", color="red", loc="bot"
    )
    d += e.Arrow().label("$i_b$", loc="end").length(1.5)
    d += e.Line().length(1)
    d += e.Dot()
    d.push()

    d += (
        re := e.Resistor()
        .label(
            "$r_e$" + f"={(re*ureg.ohm):.1f~#P}\n@{(Ie *ureg.ampere):.1f~#P}",
            color="red",
            loc="bot",
        )
        .down()
        .length(2)
    )
    d += (
        RD := e.Resistor()
        .label("$R_D$" + f"\n{(RD*ureg.ohm):.1f~#P}", color="blue", loc="bot")
        .down()
        .length(2)
    )
    d += e.GroundChassis()

    d += (
        e.Arc2(arrow="<->", radius=0.6, color="red")
        .linewidth(0.5)
        .at(re.center, dx=2.0, dy=0)
        .to(RD.center, dx=2.0, dy=0)
        .label("$R_e$" + f"={(Re*ureg.ohm):.1f~#P}", color="red", ofst=(0.25, 0))
    )

    d.pop()
    d += e.Line().up().length(1)
    d += e.SourceI().reverse().up().label("$\\beta i_b$", loc="bot").length(1)
    d += e.Line().length(0.5)
    d += e.Arrow().reverse().length(0.75)
    d += e.Dot()
    d.push()

    d += (
        e.Gap()
        .up()
        .label(
            "$R_{out}$" + f"= {(Rout*ureg.ohm):.3f~#P}",
            # + "\n$Z_o$"
            # + f"= {(Zo*ureg.ohm):.3f~#P}"
            # + "\n$A_v$"
            # + f"= {(Av):.3f}",
            color="red",
        )
    )

    d.pop()
    d.push()
    d += e.Arrow().left().length(1)
    d += e.Line().length(1.5)
    d += (
        e.Resistor()
        .label("$R_f$" + f"\n{(Rf*ureg.ohm):.1f~#P}", color="blue")
        .down()
        .length(2)
    )
    d += e.Arrow().length(0.75)
    d += e.Line().length(0.5)

    d.pop()
    d += e.Line().length(0.75).right()
    d += e.Arrow().reverse().length(2.25)
    d += e.Dot(open=True).label("$V_L$" + f"\n{(VL*ureg.volt):.3f~#P}", color="red")
    d += e.Resistor().label("$R_L$" + f"\n{(RL*ureg.ohm):.1f~#P}", color="blue").down()
    d += e.GroundChassis()

    display(d)
    return d


res_series = E24
interactive_simple_fba = interactive(
    simple_fba,
    # trans_type=widgets.Select(
    #     description="Transistor Type",
    #     options=["npn", "pnp"],
    #     value="npn",
    #     rows=2,
    #     style=style,
    # ),
    Rs=widgets.SelectionSlider(
        value=50.0,
        description="$R_s$",
        options=[25, 50, 75, 100, 150, 200, 250, 300, 500, 750, 1000, 2000],
        style=style,
        layout=layout,
    ),
    RL=widgets.SelectionSlider(
        value=50.0,
        description="$R_L$",
        options=[25, 50, 75, 100, 150, 200, 250, 300, 500, 750, 1000, 2000],
        style=style,
        layout=layout,
    ),
    Rf=widgets.SelectionSlider(
        value=240.0,
        description="$R_f$",
        options=list(erange(res_series, 1.0, 10000.0)),
        style=style,
        layout=layout,
    ),
    RD=widgets.SelectionSlider(
        value=9.1,
        description="$R_D$",
        options=[0.0] + list(erange(res_series, 1.0, 2000.0)),
        style=style,
        layout=layout,
    ),
    Ie_mA=widgets.FloatText(
        value=20.0,
        description="$I_e$ mA",
        style=style,
    ),
    B0=widgets.FloatText(
        value=30.0,
        description="$\\beta_0$",
        style=style,
    ),
    FT_mhz=widgets.FloatText(
        value=300.0,
        description="$F_T$ MHz",
        style=style,
    ),
    F_mhz=widgets.FloatText(
        value=10.0,
        description="$F$ MHz",
        style=style,
    ),
    Vin=widgets.FloatText(
        value=2.0,
        description="$V_{in}$",
        style=style,
    ),
)