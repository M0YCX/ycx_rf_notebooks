import math
import sys

from IPython.display import display
import ipywidgets as widgets
# import numpy as np
import pint
# import plotly.graph_objects as go
import plotly.io as pio
import schemdraw as schem
import schemdraw.elements as e
from eseries import E12, E24, E48, E96, erange
from ipywidgets import Layout, interactive

# Support rendering plots in github
pio.renderers.default = "jupyterlab+png"
schem.use("svg")

ureg = pint.UnitRegistry()
layout = Layout(width="auto")
style = {"description_width": "100px"}

###############################################################


def draw_bjt_bias(
    trans_type="npn",
    Vcc=12.0,
    R1=4700.0,
    R2=2200.0,
    R3=510.0,
    R4=100.0,
    R5=100.0,
    Beta=100.0,
    Vdelta=0.65,
):
    Ie = Ib = Ibias = 0.0
    Vb = Vc = Vtc = Ve = 0.0

    Vb = (Vcc * R2 * R3 + Vdelta * R2 * (R4 + (R1 / (Beta + 1)))) / (
        R3 * R4 + R2 * R4 + R2 * R3 + R1 * R3 + R1 * R2 / (Beta + 1)
    )

    Vtc = (R1 * Vcc + R4 * Vb + Beta * R1 * R4 * (Vdelta - Vb) / (R3 * (Beta + 1))) / (
        R1 + R4
    )

    Ib = (Vb - Vdelta) / (R3 * (Beta + 1))

    Ie = Ib * (Beta + 1)

    Ve = Ie * R3

    Vc = Vtc - (Beta * Ib * R5)

    Ibias = Vb / R2  # to match EMRFD LADPAC 2008 biasnpn08.exe

    Isupply = Ie + Ibias

    Pcd = (Vc - Ve) * Ib * Beta  # match to EMRFD LADPAC 2008 biasnpn08.exe

    vb_alert = ""
    if Vb > Vc:
        vb_alert = "$V_b$ MUST be less than $V_c$!!!"

    d = schem.Drawing()

    d += e.GroundChassis().label(
        "$I_{bias}$" + f"\n{(Ibias * ureg.ampere):.3f~#P}", color="red", loc="bot"
    )
    d += e.Resistor().label(f"$R_2$\n{(R2 * ureg.ohms):.1f~#P}", color="blue").up()
    d += e.Dot().label(
        f"$V_b$\n{(Vb * ureg.volts):.3f~#P}\n{vb_alert}", loc="left", color="red"
    )
    d.push()
    d += e.Resistor().label(f"$R_1$\n{(R1 * ureg.ohms):.1f~#P}", color="blue")
    d += e.Line().length(1).right()
    d += e.Arrow().reverse().length(2)
    d += e.Dot().label(
        "$V^'_c$" + f"\n{(Vtc * ureg.volts):.3f~#P}", loc="right", color="red"
    )
    d.pop()
    d += (
        e.Arrow()
        .right()
        .label(f"$I_b$\n{(Ib * ureg.ampere):.3f~#P}", color="red")
        .length(1.75)
    )
    d += e.Line().length(0.5)

    if trans_type == "npn":
        d += (TR1 := e.BjtNpn(circle=True).label(f"$\\beta$={Beta}", color="blue"))
    elif trans_type == "pnp":
        d += (
            TR1 := e.BjtPnp(circle=True).label(f"$\\beta$={Beta}", color="blue")
        ).flip()
    else:
        raise "Invalid trans_type"

    d += (
        e.Resistor()
        .label(f"$R_5$\n{(R5 * ureg.ohms):.1f~#P}", color="blue")
        .up()
        .at(TR1.collector)
        .length(2.25)
    )
    d += e.Resistor().label(
        f"$R_4$\n{(R4 * ureg.ohms):.1f~#P}", color="blue", loc="bot"
    )
    vsymb = None
    if trans_type == "npn":
        vsymb = "+"
    elif trans_type == "pnp":
        vsymb = "-"
    d += e.Dot(open=True).label(
        "$" + vsymb + "V_{cc}$" + f"\n{(Vcc * ureg.volts):.1f~#P}", color="blue"
    )

    d += e.Line().length(0.25).up().at(TR1.collector)
    d += e.Dot()
    d += e.Line().right().length(1.5)
    d += e.Dot(open=True).label(
        "$V_c$" + f"\n{(Vc * ureg.volts):.3f~#P}", loc="right", color="red"
    )

    d += e.Line().length(0.25).down().at(TR1.emitter)
    d += e.Dot()
    d.push()
    d.push()
    d += (
        e.Arc2(arrow="<->", radius=0.6, color="red")
        .linewidth(0.5)
        .at(TR1.emitter, dx=-0.2, dy=-0.2)
        .to(TR1.base, dx=-0.2, dy=-0.2)
        .label("$\\Delta V$" + f"\n{(Vdelta * ureg.volts):~#P}", ofst=(-0.25, -0.25))
    )
    d.pop()
    d += e.Line().right().length(1.5)
    d += e.Dot(open=True).label(
        f"$V_e$\n{(Ve * ureg.volts):.3f~#P}", loc="right", color="red"
    )
    d.pop()
    d += (
        e.Arrow()
        .label(f"$I_e$\n{(Ie * ureg.ampere):.3f~#P}", loc="bot", color="red")
        .length(1)
    )
    d += (
        e.Resistor()
        .down()
        .label(f"$R_3$\n{(R3 * ureg.ohms):.1f~#P}", color="blue")
        .length(2)
    )
    d += e.GroundChassis()

    d += (
        e.Gap()
        .down()
        .length(3)
        .label(
            f"$I_{{supply}}$={(Isupply * ureg.ampere):.3f~#P}\n- - -\nCollector Dissipation={(Pcd * ureg.watts):.3f~#P}",
            color="red",
            loc="bot",
        )
    )

    display(d)
    return d


res_series = E24
interactive_draw_bjt_bias = interactive(
    draw_bjt_bias,
    trans_type=widgets.Select(
        description="Transistor Type",
        options=["npn", "pnp"],
        value="npn",
        rows=2,
        style=style,
    ),
    Vcc=widgets.FloatText(
        value=12.0,
        description="$V_{cc}$",
        style=style,
    ),
    R1=widgets.SelectionSlider(
        value=4700.0,
        description="$R_1$",
        options=list(erange(res_series, 100.0, 200000.0)),
        style=style,
        layout=layout,
    ),
    R2=widgets.SelectionSlider(
        value=2200.0,
        description="$R_2$",
        options=list(erange(res_series, 100.0, 200000.0)),
        style=style,
        layout=layout,
    ),
    R3=widgets.SelectionSlider(
        value=510.0,
        description="$R_3$",
        options=list(erange(res_series, 100.0, 200000.0)),
        style=style,
        layout=layout,
    ),
    R4=widgets.SelectionSlider(
        value=100.0,
        description="$R_4$",
        options=list(erange(res_series, 100.0, 200000.0)),
        style=style,
        layout=layout,
    ),
    R5=widgets.SelectionSlider(
        value=100.0,
        description="$R_5$",
        options=[
            0,
        ]
        + list(erange(res_series, 100.0, 200000.0)),
        style=style,
        layout=layout,
    ),
    Beta=widgets.FloatText(
        value=100.0,
        description="$\\beta$",
        style=style,
    ),
    Vdelta=widgets.FloatText(
        value=0.65,
        description="$\\Delta V$",
        style=style,
    ),
)
