import math

# import sys

import ipywidgets as widgets

import numpy as np
import pint

# import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import schemdraw as schem
import schemdraw.elements as e
from eseries import E12, E24, E48, E96, erange
from IPython.display import display
from ipywidgets import Layout, interactive, GridBox, interactive_output
from ycx_complex_numbers import Complex, NetABCD, NetY, NetZ, Y, Z
from ycx_rf_amplifiers.y_params import calc_linvill_stability2, calc_stern_stability2

# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'


# Support rendering plots in github
pio.renderers.default = "jupyterlab+png"
schem.use("svg")

ureg = pint.UnitRegistry()
ureg = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
layout = Layout(width="auto")
layout_grid1 = Layout(
    width="auto",
)
style = {"description_width": "200px"}

###############################################################


def _calc_complex_fba(
    ZS=None,
    ZL=None,
    B0=None,
    F=None,
    FT=None,
    Rbp=None,
    Ccb=None,
    Ie_mA=None,
    Re=None,
    Le=None,
    Rf=None,
    Lf=None,
    Cf=None,
    N=None,
):
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

    # Calc output reflection coefficient & return loss
    GammaOut = (zout - ZS) / (zout + ZS)
    OutRetLoss = -20 * math.log10(abs(GammaOut))

    # Calc Linvill stability
    linvillC = calc_linvill_stability2(y11=Y1.y11, y12=Y1.y12, y21=Y1.y21, y22=Y1.y22)

    # Calc Stern stability
    sternK = calc_stern_stability2(
        y11=Y1.y11,
        y12=Y1.y12,
        y21=Y1.y21,
        y22=Y1.y22,
        GS=1 / (ZS.real),
        GL=1 / (ZL.real),
    )

    return {
        "F": F,
        "Y": Y1,
        "S": S1,
        "zin": zin,
        "zin_mag": abs(zin),
        "izout": izout,
        "zout": zout,
        "zout_mag": abs(zout),
        "InRetLoss": InRetLoss,
        "OutRetLoss": OutRetLoss,
        "Gt_db": Gt_db,
        "re": re,
        "Ze": Ze,
        "beta": beta,
        "beta_mag": abs(beta),
        "linvillC": linvillC,
        "sternK": sternK,
    }


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

    fba = _calc_complex_fba(
        ZS=ZS,
        ZL=ZL,
        B0=B0,
        F=F,
        FT=FT,
        Rbp=Rbp,
        Ccb=Ccb,
        Ie_mA=Ie_mA,
        Re=Re,
        Le=Le,
        Rf=Rf,
        Lf=Lf,
        Cf=Cf,
        N=N,
    )

    schem.config(inches_per_unit=0.4, fontsize=10)
    d = schem.Drawing()
    d.push()
    d += e.Resistor().label("$Z_S$" + f"\n{ZS}", color="blue").down().length(2)
    d += e.SourceSin().down().length(2)
    d += e.GroundChassis().label(
        "${Z_{in}$"
        + f"\n{fba['zin']:.3f~S}\nReturn Loss={(fba['InRetLoss'] * ureg.decibel):.3f~#P}",
        loc="bot",
        color="red",
    )
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
        .label("$r_e$" + f"={(fba['re'] * ureg.ohms):.1f~#P}", color="red", loc="bot")
        .down()
        .length(2)
    )
    d += e.Dot(open=True).label("e", loc="right", color="grey")
    d.push()
    d += (
        e.Gap()
        .length(5)
        .right()
        .label(f"$Z_e$={(fba['Ze'].c * ureg.ohms):.2}", color="red")
    )
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
        f"\n$\\beta$={fba['beta']:.1f}@{(F*ureg.hertz):.1f~#P}",
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
            "(intermediate) ${Z_{out}$" + f"\n{fba['izout']:.3f~S}",
            loc="right",
            color="red",
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
            + f"\n{fba['zout']:.3f~S}\nReturn Loss={(fba['OutRetLoss'] * ureg.decibel):.3f~#P}",
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
        "Gain\n$G_t$" + f"={(fba['Gt_db'] * ureg.decibel):.4f~#P}",
        color="red",
        loc="bot",
    )

    display(d)

    fba_res = {}
    for f in np.logspace(3, math.log10(FT), num=100):
        fba = _calc_complex_fba(
            ZS=ZS,
            ZL=ZL,
            B0=B0,
            F=f,
            FT=FT,
            Rbp=Rbp,
            Ccb=Ccb,
            Ie_mA=Ie_mA,
            Re=Re,
            Le=Le,
            Rf=Rf,
            Lf=Lf,
            Cf=Cf,
            N=N,
        )

        for f_k, f_i in fba.items():
            if f_k not in fba_res:
                fba_res[f_k] = []
            fba_res[f_k].append(f_i)
    # print(fba_res)

    fig = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=(
            "|Beta|",
            "Gain dB",
            "I/P Return Loss dB",
            "O/P Return Loss dB",
            "Linvill Stability [>0 & <1]",
            "Stern Stability [>1]",
        ),
        x_title="Frequency",
    )

    fig.add_trace(
        go.Scatter(x=fba_res["F"], y=fba_res["beta_mag"], name="magnitude beta"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=fba_res["F"], y=fba_res["Gt_db"], name="transducer gain db"),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(x=fba_res["F"], y=fba_res["InRetLoss"], name="i/p ret-loss db"),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(x=fba_res["F"], y=fba_res["OutRetLoss"], name="o/p ret-loss db"),
        row=1,
        col=4,
    )

    fig.add_trace(
        go.Scatter(x=fba_res["F"], y=fba_res["linvillC"], name="Linvill Stability"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=fba_res["F"], y=fba_res["sternK"], name="Stern Stability"),
        row=2,
        col=2,
    )

    fig.update_layout(height=500, width=1400, title_text="Modeled Characteristics")
    fig.update_xaxes(type="log")
    fig.show()

    return d


res_series = E24

F_mhz = widgets.FloatText(
    value=10.0,
    description="$F$ MHz",
    style=style,
    layout=Layout(width="auto", grid_area="F_mhz"),
)

FT_mhz = widgets.FloatText(
    value=300.0,
    description="$F_T$ MHz",
    style=style,
    layout=Layout(width="auto", grid_area="FT_mhz"),
)
B0 = widgets.FloatText(
    value=100.0,
    description="$B0$",
    style=style,
    layout=Layout(width="auto", grid_area="B0"),
)
Ie_mA = widgets.FloatText(
    value=10.0,
    description="$I_e$ mA",
    style=style,
    layout=Layout(width="auto", grid_area="Ie_mA"),
)
Rbp = widgets.FloatText(
    value=10.0,
    description="$R^`_b$ base spreading resistance",
    style=style,
    layout=Layout(width="auto", grid_area="Rbp"),
)
Re = widgets.SelectionSlider(
    value=6.8,
    description="$R_e$",
    options=[
        0.0,
    ]
    + list(erange(res_series, 1, 1000.0)),
    style=style,
    layout=Layout(width="auto", grid_area="Re"),
)
Le_nH = widgets.FloatText(
    value=10.0,
    description="$Le$ nH",
    style=style,
    layout=Layout(width="auto", grid_area="Le_nH"),
)
Ccb_pF = widgets.FloatText(
    value=5.0,
    description="$C_{cb}$ pF",
    style=style,
    layout=Layout(width="auto", grid_area="Ccb_pF"),
)
Cf_pF = widgets.SelectionSlider(
    value=10000.0,
    description="$C_f$ pF",
    options=list(erange(res_series, 1, 10000.0)),
    style=style,
    layout=Layout(width="auto", grid_area="Cf_pF"),
)
Lf_nH = widgets.SelectionSlider(
    value=20.0,
    options=list(erange(res_series, 1, 10000.0)),
    description="$Lf$ nH",
    style=style,
    layout=Layout(width="auto", grid_area="Lf_nH"),
)
Rf = widgets.SelectionSlider(
    value=1000,
    description="$R_f$",
    options=list(erange(res_series, 1, 100000.0)),
    style=style,
    layout=Layout(width="auto", grid_area="Rf"),
)
ZS_real = widgets.FloatText(
    value=50,
    description="$Z_S$",
    style=style,
    layout=Layout(width="auto", grid_area="ZS_real"),
)
ZS_imag = widgets.FloatText(
    value=0,
    description="$Z_S$ j",
    style=style,
    layout=Layout(width="auto", grid_area="ZS_imag"),
)
ZL_real = widgets.FloatText(
    value=50,
    description="$Z_L$",
    style=style,
    layout=Layout(width="auto", grid_area="ZL_real"),
)
ZL_imag = widgets.FloatText(
    value=0,
    description="$Z_L$ j",
    style=style,
    layout=Layout(width="auto", grid_area="ZL_imag"),
)
N = widgets.IntSlider(
    value=1,
    description="$N$",
    min=1,
    max=20,
    style=style,
    layout=Layout(width="auto", grid_area="N"),
    continuous_update=True,
)
g1 = GridBox(
    children=[
        F_mhz,
        FT_mhz,
        B0,
        Ie_mA,
        Rbp,
        Le_nH,
        Re,
        Ccb_pF,
        Cf_pF,
        Lf_nH,
        Rf,
        ZS_real,
        ZS_imag,
        ZL_real,
        ZL_imag,
        N,
    ],
    layout=Layout(
        width="100%",
        grid_template_rows="auto auto auto",
        grid_template_columns="25% 25% 25% 25%",
        grid_template_areas="""
            "F_mhz FT_mhz B0 Ie_mA"
            "Rbp Re Re Le_nH"
            "Ccb_pF Cf_pF Cf_pF Lf_nH"
            "Rf Rf ZS_real ZS_imag"
            "ZL_real ZL_imag N N"
            """,
        # "main main . sidebar "
        # "footer footer footer footer"
    ),
)


interactive_complex_fba = (
    g1,
    interactive_output(
        complex_fba,
        {
            "F_mhz": F_mhz,
            "FT_mhz": FT_mhz,
            "B0": B0,
            "Ie_mA": Ie_mA,
            "Rbp": Rbp,
            "Le_nH": Le_nH,
            "Re": Re,
            "Ccb_pF": Ccb_pF,
            "Cf_pF": Cf_pF,
            "Lf_nH": Lf_nH,
            "Rf": Rf,
            "ZS_real": ZS_real,
            "ZS_imag": ZS_imag,
            "ZL_real": ZL_real,
            "ZL_imag": ZL_imag,
            "N": N,
        },
    ),
)
