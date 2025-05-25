import math

import ipywidgets as widgets
import numpy as np
import pint
import plotly.graph_objects as go
import plotly.io as pio
import schemdraw as schem
import schemdraw.elements as e
import skrf as rf
from eseries import E12, E24, E48, E96, erange
from IPython.display import display
from ipywidgets import Layout, interact, interactive
from plotly.subplots import make_subplots
from skrf import Network, plotting
from ycx_complex_numbers import Neta, NetS, NetY, NetZ, S, Y, Z, a
from charts import plot_smith_annotated

pio.renderers.default = "jupyterlab+png"
schem.use("svg")

ureg = pint.UnitRegistry()
ureg = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)

style = {"description_width": "200px"}
res_series = E24


def calc_jfet(
    Cdg=None,
    Cgs=None,
    Cds=None,
    L1=None,
    L2=None,
    L3=None,
    R1=1,
    R2=1,
    R3=1,
    gm=None,
    gm0=None,
    beta=None,
    F=None,
    VA=100,
    IDSS=None,
    Vgs=None,
    VgsOff=None,
    ID=None,
    Rch=None,
):
    ####
    # see https://www.electrical4u.com/parameters-of-jfet-or-specifications-of-jfet/
    Vp = VgsOff
    if ID is None or ID == 0:
        ID = IDSS * (1 - Vgs / VgsOff) ** 2
    else:  # calc Vgs from ID instead
        Vgs = (Vp * (IDSS - math.sqrt(ID * IDSS))) / IDSS

    if gm0 is None:
        gm0 = (-2 * IDSS) / Vp

    if gm is None:
        gm = (2 * IDSS) / abs(Vp) * (1 - Vgs / Vp)
        # gm = (-2 * Idss) / Vp * (1 + (ID * Rs) / Vp)
    if beta is None:
        beta = IDSS / Vp**2
    ####

    Ro = VA / ID
    if Rch is None:
        Rch = Ro  # ???

    gds = 1 / Rch  # ohms = 1/Rch (channel resistance???)

    tau = 3.365 * 10**-12
    # tau = 33.65 * 10**-12 # has opposite effect...

    jw = 1j * 2 * math.pi * F

    # lead inductances (CS)
    #       R        L
    ZL1 = Z(R1 + jw * L1)  # G
    ZL2 = Z(R2 + jw * L2)  # S
    ZL3 = Z(R3 + jw * L3)  # D

    Y = NetY(
        y11=jw * (Cgs + Cdg),
        y12=-jw * Cdg,
        y21=gm ** (math.e ** (jw * tau)) - jw * Cdg,
        y22=gds + jw * (Cds + Cdg),
    )

    aZL1 = Neta(a11=1, a12=ZL1, a21=0, a22=1)
    dummyR = 0.001
    z_ZL2 = NetZ(z11=dummyR + ZL2, z12=ZL2, z21=ZL2, z22=ZL2 + dummyR)
    aZL3 = Neta(a11=1, a12=ZL3, a21=0, a22=1)

    yj = Y
    zj = yj.to_Z()

    zj_l1 = zj + z_ZL2
    aj = aZL1 @ zj.to_a() @ aZL3

    Y = aj.to_Y()

    Ycg = None
    Ycs = None
    Ycd = None

    Ycs = Y
    Ycg = Y.exchange_to_cg(from_config="cs")
    Ycd = Y.exchange_to_cd(from_config="cs")

    return {
        "Ycs": Ycs,
        "Ycg": Ycg,
        "Ycd": Ycd,
        "gds": gds,
        "gm": gm,
        "gm0": gm0,
        "beta": beta,
        "ID": ID,
        "Vgs": Vgs,
    }


def draw_jfet(
    Cdg=None,
    Cgs=None,
    Cds=None,
    gm=None,
    gds=None,
    config=None,
):
    d = schem.Drawing()

    d += e.Dot(open=True).label("g", loc="left")
    d += e.Line().right().length(2)
    d += e.Dot()
    d += (
        e.Capacitor()
        .label("$C_{gs}$" + f"\n{(Cgs * ureg.farad):~#P}", color="blue")
        .down()
        .length(2)
    )
    d += e.Line().right().length(2)
    d += e.Dot()
    d.push()
    d += e.Line().down().length(2)
    d += e.Dot(open=True).label("s", loc="bot")
    d.pop()
    d += e.Line().right().length(2)
    d += (
        e.SourceI()
        .label("$g_m V_{gs}$" + f"\n$g_m=${gm:.3f}", color="blue")
        .reverse()
        .up()
    )
    d += e.Dot()
    d.push()
    d += (
        e.Capacitor()
        .label("$C_{dg}$" + f"\n{(Cdg * ureg.farad):~#P}", color="blue")
        .left()
        .length(4)
    )
    d += e.Line().down().length(1)
    d += e.Dot()
    d.pop()
    d += e.Line().right().length(2)
    d += e.Dot()
    d.push()
    d += (
        e.Resistor()
        .label("$g_{ds}$" + f"\n{(gds * ureg.mhos):.1f~#P}", loc="bot", color="red")
        .down()
    )
    d += e.Line().left().length(2)
    d += e.Dot()
    d.pop()
    d += e.Line().right().length(2)
    d += e.Dot()
    d.push()
    d += (
        e.Capacitor()
        .label("$C_{ds}$" + f"\n{(Cds * ureg.farad):.1f~#P}", loc="bot", color="blue")
        .down()
    )
    d += e.Line().left().length(2)
    d += e.Dot()
    d.pop()
    d += e.Line().right().length(2)
    d += e.Dot(open=True).label("d", loc="right")

    return d


def calc_plot_jfet(
    title="JFET",
    Cdg_pf=1,
    Cgs_pf=1,
    Cds_pf=1,
    L1_nH=3,
    L2_nH=3,
    L3_nH=3,
    R1=1,
    R2=1,
    R3=1,
    Idss_mA=10,
    VgsOff=None,
    config="cs",  # plot config
    Vgs=None,
    ID=0,
    Ffrom=1 * 10**6,
    Fto=1000 * 10**6,
):
    g11_abs = True
    b11_abs = True
    g12_abs = True
    b12_abs = True
    g21_abs = True
    b21_abs = True
    g22_abs = True
    b22_abs = True

    IDSS = Idss_mA * 10**-3
    Cdg = Cdg_pf * 10**-12
    Cgs = Cgs_pf * 10**-12
    Cds = Cds_pf * 10**-12

    L1 = L1_nH * 10**-9
    L2 = L2_nH * 10**-9
    L3 = L3_nH * 10**-9

    ydf = {}
    ydf["freq"] = []

    ydf["S"] = []

    ydf["y11_g"] = []  # real: conductance
    ydf["y11_g_val"] = []

    ydf["y11_b"] = []  # imaginary: susceptance
    ydf["y11_b_val"] = []

    ydf["y12_g"] = []
    ydf["y12_g_val"] = []

    ydf["y12_b"] = []
    ydf["y12_b_val"] = []

    ydf["y21_g"] = []
    ydf["y21_g_val"] = []

    ydf["y21_b"] = []
    ydf["y21_b_val"] = []

    ydf["y22_g"] = []
    ydf["y22_g_val"] = []

    ydf["y22_b"] = []
    ydf["y22_b_val"] = []

    gm = None

    beta = None
    gm = None

    for f in np.logspace(math.log10(Ffrom), math.log10(Fto), num=100):
        res = calc_jfet(
            Cdg=Cdg,
            Cgs=Cgs,
            Cds=Cds,
            L1=L1,
            L2=L2,
            L3=L3,
            R1=R1,
            R2=R2,
            R3=R3,
            F=f,
            IDSS=IDSS,
            ID=ID,
            Vgs=Vgs,
            VgsOff=VgsOff,
        )
        ID = res["ID"]
        Vgs = res["Vgs"]
        yrow = res[f"Y{config}"]
        beta = res["beta"]  # we only want the last one in the list
        gm = res["gm"]  # we only want the last one in the list

        ydf["freq"].append(f)
        ydf["S"].append(
            [
                [yrow.to_S()._c11.c, yrow.to_S()._c12.c],
                [yrow.to_S()._c21.c, yrow.to_S()._c22.c],
            ]
        )

        ydf["y11_g"].append(
            (yrow.y11.real if not g11_abs else abs(yrow.y11.real)) * 10**3
        )
        ydf["y11_g_val"].append(yrow.y11.real * 10**3)

        ydf["y11_b"].append(
            (yrow.y11.imag if not b11_abs else abs(yrow.y11.imag)) * 10**3
        )
        ydf["y11_b_val"].append(yrow.y11.imag * 10**3)

        ydf["y12_g"].append(
            (yrow.y12.real if not g12_abs else abs(yrow.y12.real)) * 10**3
        )
        ydf["y12_g_val"].append(yrow.y12.real * 10**3)

        ydf["y12_b"].append(
            (yrow.y12.imag if not b12_abs else abs(yrow.y12.imag)) * 10**3
        )
        ydf["y12_b_val"].append(yrow.y12.imag * 10**3)

        ydf["y21_g"].append(
            (yrow.y21.real if not g21_abs else abs(yrow.y21.real)) * 10**3
        )
        ydf["y21_g_val"].append(yrow.y21.real * 10**3)

        ydf["y21_b"].append(
            (yrow.y21.imag if not b21_abs else abs(yrow.y21.imag)) * 10**3
        )
        ydf["y21_b_val"].append(yrow.y21.imag * 10**3)

        ydf["y22_g"].append(
            (yrow.y22.real if not g22_abs else abs(yrow.y22.real)) * 10**3
        )
        ydf["y22_g_val"].append(yrow.y22.real * 10**3)

        ydf["y22_b"].append(
            (yrow.y22.imag if not b22_abs else abs(yrow.y22.imag)) * 10**3
        )
        ydf["y22_b_val"].append(yrow.y22.imag * 10**3)
    print(f"beta={(beta*ureg.siemens):.6f~#P}")
    print(f"gm={(gm*ureg.siemens):.6f~#P}")

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "|y11|",
            "|y12|",
            "|y21|",
            "|y22|",
        ),  # , "S11", "S12", "S21", "S22"),
        x_title="Frequency",
        y_title="mmhos",
        specs=[
            [{"secondary_y": True}, {"secondary_y": True}],
            [{"secondary_y": True}, {"secondary_y": True}],
            # [{}, {}],
            # [{}, {}],
        ],
    )

    ######################
    # Y-Parameters plots
    fig.add_trace(
        go.Scatter(
            x=ydf["freq"],
            y=ydf["y11_g"],
            name="G I/P Admittance",
            customdata=ydf["y11_g_val"],
        ),
        row=1,
        col=1,
        # layout =_yaxis_range=(1, 10),
    )
    fig.add_trace(
        go.Scatter(
            x=ydf["freq"],
            y=ydf["y11_b"],
            name="B I/P Admittance",
            customdata=ydf["y11_b_val"],
        ),
        # secondary_y=True,
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ydf["freq"],
            y=ydf["y12_g"],
            name="G Rev Transadmittance",
            customdata=ydf["y12_g_val"],
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=ydf["freq"],
            y=ydf["y12_b"],
            name="B Rev Transadmittance",
            customdata=ydf["y12_b_val"],
        ),
        # secondary_y=True,
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=ydf["freq"],
            y=ydf["y21_g"],
            name="G Fwd Transadmittance",
            customdata=ydf["y21_g_val"],
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ydf["freq"],
            y=ydf["y21_b"],
            name="B Fwd Transadmittance",
            customdata=ydf["y21_b_val"],
        ),
        # secondary_y=True,
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ydf["freq"],
            y=ydf["y22_g"],
            name="G O/P Admittance",
            customdata=ydf["y22_g_val"],
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=ydf["freq"],
            y=ydf["y22_b"],
            name="B O/P Admittance",
            customdata=ydf["y22_b_val"],
        ),
        # secondary_y=True,
        row=2,
        col=2,
    )

    fig.update_traces(
        hovertemplate=("<b>%{customdata:.6f}</b>"),
    )
    fig.update_layout(
        height=600,
        width=750,
        hoversubplots="axis",
        hovermode="x",
        title_text=f"{title} [Config:{config.upper()}, ID:{(ID*ureg.amperes):.3~#P}, Vgs:{(Vgs*ureg.volts):.1f~#P}]",
    )
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")

    fig.show()

    ntw = plot_smith_annotated(frequency=ydf["freq"], s=ydf["S"])
    return ntw


interactive_calc_plot_jfet = interactive(
    calc_plot_jfet,
    Cdg_pf=widgets.FloatText(
        value=2.5,
        description="$C_{dg}\\ pf$",
        style=style,
    ),
    Cgs_pf=widgets.FloatText(
        value=5.0,
        description="$C_{gs}\\ pf$",
        style=style,
    ),
    Cds_pf=widgets.FloatText(
        value=0.1,
        description="$C_{ds}\\ pf$",
        style=style,
    ),
    L1_nH=widgets.FloatText(
        value=0,
        description="$L_1\\ nH$",
        style=style,
    ),
    L2_nH=widgets.FloatText(
        value=0,
        description="$L_2\\ nH$",
        style=style,
    ),
    L3_nH=widgets.FloatText(
        value=0,
        description="$L_3\\ nH$",
        style=style,
    ),
    R1=widgets.FloatText(
        value=2,
        description="$R_1$",
        style=style,
    ),
    R2=widgets.FloatText(
        value=1,
        description="$R_2$",
        style=style,
    ),
    R3=widgets.FloatText(
        value=1,
        description="$R_3$",
        style=style,
    ),
    Idss_mA=widgets.FloatText(
        value=30.0,
        description="$I_{DSS}\\ mA$",
        style=style,
    ),
    Vgs=widgets.FloatText(
        value=-1.65,
        description="$V_{gs}$",
        style=style,
    ),
    VgsOff=widgets.FloatText(
        value=-3.91,
        description="$V_{gs(off)}(V_P)$",
        style=style,
    ),
    config="cs",
    Ffrom=100 * 10**6,
    # Fto=1000 * 10**6,
)
