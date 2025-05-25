import math
import ipywidgets as widgets
import numpy as np
import pint
import matplotlib.pyplot as plt
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import schemdraw as schem
import schemdraw.elements as e
import skrf as rf
from skrf import Network, plotting
from eseries import E12, E24, E48, E96, erange
from IPython.display import display
from ipywidgets import Layout, interactive, GridBox, interactive_output
from ycx_complex_numbers import Complex, Neta, Netb, NetY, NetZ, Y, Z
from ycx_rf_amplifiers.y_params import calc_linvill_stability2, calc_stern_stability2
from ycx_rf_amplifiers.s_params import calc_rollett_stability
from charts import plot_smith_annotated

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
    extLb=None,
    extCin=None,
    Ccb=None,
    Cce=None,
    Ie_mA=None,
    Re=None,
    extLe=None,
    Rf=None,
    Lf=None,
    Cf=None,
    extLc=None,
    extCout=None,
    extCrev=None,
    N=None,
):
    """
    Params:
        ZS      - Source Impedance
        ZL      - Load Impedance
        B0      - DC or low frequency Beta
        F       - Frequence to calculate characteristics for
        FT      - Transistor's transistion frequency,
        Rbp     - Base spreading resistance (intrinsic),
        extLb  - Input series inductance (extrinsic),
        extCin  - Input shunt capacitance (extrinsic),
        Ccb     - Collector to base capacitance (intrinsic),
        Cce     - Collector to emitter capacitance (intrinsic),
        Ie_mA   - Bias emitter current,
        Re      - Emitter resistor
        extLe      - Emitter inductance (extrinsic),
        Rf      - Feedback resistor,
        Lf      - Feedback inductance,
        Cf      - Feedback capacitance,
        extLc - Output series inductance (extrinsic),
        extCout - Output parallel capacitance (extrinsic),
        extCrev - Reverse capacitance (extrinsic),
        N       - Ideal transformer turns ratio,
    """
    YS = Y(1 / ZS)
    YL = Y(1 / ZL)

    # YL as a ABCD' matrix for reverse cascade through the output
    # transformer to calc the Stern stability
    YL_Ap = Netb(b11=1, b12=0, b21=-YL, b22=1)

    w = 2 * math.pi * F
    jw = 1j * w

    extLb_A = Neta(a11=1, a12=jw * extLb, a21=0, a22=1)  # L in series
    extLc_A = Neta(a11=1, a12=jw * extLc, a21=0, a22=1)  # L in series
    extLe_z = Z(1j * (w * extLe))
    extLe_Z = NetZ(z11=extLe_z, z12=extLe_z, z21=extLe_z, z22=extLe_z)

    extCin_y = Y(-jw * extCin)
    extCin_Y = NetY(y11=extCin_y, y12=-extCin_y, y21=-extCin_y, y22=extCin_y)

    extCout_A = Neta(a11=1, a12=0, a21=jw * extCout, a22=1)  # C in shunt

    extCrev_y = Y(-jw * extCrev)
    extCrev_Y = NetY(y11=extCrev_y, y12=-extCrev_y, y21=-extCrev_y, y22=extCrev_y)

    # base spreading resistor as ABCD matrix for cascading below
    Rbp_A = Neta(a11=1, a12=Rbp, a21=0, a22=1)

    ################################################
    # Hybrid-Pi model

    # Emitter complex impedance
    re = 26 / Ie_mA
    zre = Z(re)

    beta = Complex(B0 / (1 + (1j * B0 * F) / FT))

    # Note: this is the same as adding the Y matrix of Ccb to the simple transistor model (ie in parallel)
    # intrinsics:-
    y11e = Y(1 / (zre * (beta + 1)) + (jw * Ccb))
    y12e = Y(0 - (jw * Ccb))
    y21e = Y(beta / (zre * (beta + 1)) - (jw * Ccb))
    y22e = Y(0 + (jw * Ccb))
    Ye = NetY(y11=y11e, y12=y12e, y21=y21e, y22=y22e)

    # Cascade the extLb, extCin and base spreading resistance to the hybrid-pi amplifier
    Ae = Ye.to_a()
    Aintrinsic = Rbp_A @ Ae
    # if Cce provided than cascade that too
    if Cce is not None:
        ACce = Neta(a11=1, a12=0, a21=jw * Cce, a22=1)
        Aintrinsic = Aintrinsic @ ACce

    # The intrinsic model is complete at this point in the matrix Aintrinsic

    Awith_extrinsic = extLb_A @ Aintrinsic @ extLc_A
    Ywith_extrinsic = (Awith_extrinsic.to_Z() + extLe_Z).to_Y()
    Awith_extrinsic = (extCrev_Y + Ywith_extrinsic + extCin_Y).to_a() @ extCout_A

    extRe = Z(Re)
    extRe_Z = NetZ(z11=extRe, z12=extRe, z21=extRe, z22=extRe)
    YAwith_extrinsic = (Awith_extrinsic.to_Z() + extRe_Z).to_Y()

    Zf = Z(Rf + 1j * (2 * math.pi * F * Lf - 1 / (2 * math.pi * F * Cf)))
    Yf = NetY(y11=1 / Zf, y12=-1 / Zf, y21=-1 / Zf, y22=1 / Zf)
    Ywith_feedback = YAwith_extrinsic + Yf

    ##########
    # Get interim output impedance to match Wes's Zout
    izout = Z(1 / Ywith_feedback.in_out(ys=1 / ZS, yl=1 / ZL)["Yout"])

    ###################################################
    # output transformer N:1 as ABCD for cascade below
    ATR1 = None
    if N > 0:
        ATR1 = Neta(a11=N, a12=0, a21=0, a22=1 / N)
    else:
        ATR1 = Neta(a11=1 / abs(N), a12=0, a21=0, a22=abs(N))
    Y1 = (Ywith_feedback.to_a() @ ATR1).to_Y()
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
    # print(f"GammaOut={GammaOut}")
    OutVSWR = (1 + abs(GammaOut)) / (1 - abs(GammaOut))

    # Calc Linvill stability
    # linvillC = calc_linvill_stability2(y11=Ywith_feedback.y11, y12=Ywith_feedback.y12, y21=Ywith_feedback.y21, y22=Ywith_feedback.y22)
    linvillC = calc_linvill_stability2(y11=Y1.y11, y12=Y1.y12, y21=Y1.y21, y22=Y1.y22)

    # Calc ZL as seen through the output transformer
    TZL = (YL_Ap @ ATR1.to_b()).to_Z()

    # Calc Stern stability
    # fmt:off
    # sternK = calc_stern_stability2(
    #     y11=Ywith_feedback.y11,
    #     y12=Ywith_feedback.y12,
    #     y21=Ywith_feedback.y21,
    #     y22=Ywith_feedback.y22,
    #     GS=1 / (ZS.real),
    #     GL=1 / (abs(TZL.z11.real)), # TODO: I dont think this is correct! as it doesnt account for the effect of the transformer on Ywith_feedback...  I think i need a reverse ABCD' cascade matrix of the transformer to translate ZL to what is being seen by the amplifier matrix Ywith_feedback
    # )
    # fmt:on
    sternK = calc_stern_stability2(
        y11=Y1.y11,
        y12=Y1.y12,
        y21=Y1.y21,
        y22=Y1.y22,
        GS=1 / (ZS.real),
        GL=1 / (abs(TZL.z11.real)),
    )

    insertion_gain_db = 10 * math.log10(S1.s21.as_polar()["mag"] ** 2)

    rollett_stability = calc_rollett_stability(
        s11=S1.s11, s12=S1.s12, s21=S1.s21, s22=S1.s22
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
        "InVSWR": InVSWR,
        "OutVSWR": OutVSWR,
        "Gt_db": Gt_db,
        "re": re,
        "beta": beta,
        "beta_mag": abs(beta),
        "linvillC": linvillC,
        "sternK": sternK,
        "insertion_gain_db": insertion_gain_db,
        "rollett_stability": rollett_stability,
    }


def complex_fba(
    F_mhz=10,
    FT_mhz=300,
    B0=100,
    Ie_mA=10,
    Rbp=10.0,
    Re=6.8,
    extLe_nH=0,
    extLb_nH=0,
    extCin_pF=0,
    Ccb_pF=5,
    Cce_pF=0,
    Cf_pF=10000,
    Lf_nH=20,
    Rf=1000,
    ZS_real=50,
    ZS_imag=0,
    ZL_real=50,
    ZL_imag=0,
    extLc_nH=0,
    extCout_pF=0,
    extCrev_pF=0,
    N=2,
    from_mhz=None,
    to_mhz=None,
):
    Ccb = Ccb_pF * 10**-12
    extLe = extLe_nH * 10**-9
    F = F_mhz * 10**6
    FT = FT_mhz * 10**6
    Ie = Ie_mA * 10**-3

    Cf = Cf_pF * 10**-12
    Lf = Lf_nH * 10**-9

    Cce = None
    if Cce_pF != 0:
        Cce = Cce_pF * 10**-12

    extLb = extLb_nH * 10**-9
    extCin = extCin_pF * 10**-12

    extLc = extLc_nH * 10**-9
    extCout = extCout_pF * 10**-12

    extCrev = extCrev_pF * 10**-12

    ZS = Z(ZS_real + (1j * ZS_imag))
    ZL = Z(ZL_real + (1j * ZL_imag))

    intrinsic_color = "#8585b5"
    extrinsic_color = "#985858"
    feedback_color = "#589858"

    # plot range

    if from_mhz is None:
        from_hz = 1000
    else:
        from_hz = from_mhz * 10**6
    if to_mhz is None:
        to_hz = FT
    else:
        to_hz = to_mhz * 10**6
    plot_from_exp = math.log10(from_hz)
    plot_to_exp = math.log10(to_hz)

    fba = _calc_complex_fba(
        ZS=ZS,
        ZL=ZL,
        B0=B0,
        F=F,
        FT=FT,
        Rbp=Rbp,
        extLb=extLb,
        extCin=extCin,
        Ccb=Ccb,
        Cce=Cce,
        Ie_mA=Ie_mA,
        Re=Re,
        extLe=extLe,
        Rf=Rf,
        Lf=Lf,
        Cf=Cf,
        extLc=extLc,
        extCout=extCout,
        extCrev=extCrev,
        N=N,
    )

    schem.config(inches_per_unit=0.4, fontsize=10)
    d = schem.Drawing()
    d += e.Gap().right()  # make space on left for truncated ZS
    d.push()
    d += (
        e.Resistor()
        .label("$Z_S$" + f"\n{ZS.as_complex()}", color="blue")
        .down()
        .length(2)
    )
    d.push()

    d += e.Gap().down().length(4)
    d += (
        e.Gap()
        .right()
        .length(9)
        .label(
            "${Z_{in}$"
            + f"\n{fba['zin']:.3f~S}\nReturn Loss={(fba['InRetLoss'] * ureg.decibel):.3f~#P}\nvswr={(fba['InVSWR']):.2f}",
            halign="right",
            loc="top",
            color="red",
        )
    )

    d.pop()
    d += e.SourceSin().label("$V_{in}$", loc="top").down().length(2)
    d += e.GroundChassis()
    d.pop()
    d += e.Line().right().length(5)

    d.push()
    d += e.Dot(color=extrinsic_color)
    d += e.Line(color=extrinsic_color).length(4).down()
    d += (
        e.Capacitor(color=extrinsic_color)
        .length(6.5)
        .right()
        .label("$extC_{in}$" + f"\n{(extCin * ureg.farads):.1f~#P}", color="blue")
    )
    d += e.Dot(color=extrinsic_color)
    d.pop()

    d += (
        e.Inductor(color=extrinsic_color)
        .length(2)
        .label(
            "$extL_b$" + f"\n{(extLb * ureg.henrys):.2f~#P}", color="blue", loc="bot"
        )
    )
    d += e.Dot(open=True, color=intrinsic_color).label(
        "b", color=intrinsic_color, loc="bot"
    )

    d += (
        e.Resistor(color=intrinsic_color)
        .right()
        .label(f"$R^`_b$\n{(Rbp * ureg.ohms):.1f~#P}", color="blue")
    )
    d += e.Line(color=intrinsic_color).right().length(1.5)
    d += e.Dot(color=intrinsic_color)
    d.push()
    d += (
        e.Resistor(color=intrinsic_color)
        .label("$r_e$" + f"={(fba['re'] * ureg.ohms):.1f~#P}", color="red", loc="top")
        .down()
        .length(2)
    )
    d += e.Dot(open=True, color=intrinsic_color).label(
        "e", loc="left", color=intrinsic_color
    )
    d += (
        e.Inductor(color=extrinsic_color)
        .label(
            "$extL_e$" + f"\n{(extLe * ureg.henrys):.2f~#P}", color="blue", loc="bot"
        )
        .length(2)
    )
    d += (
        e.Resistor()
        .label("$R_e$" + f"\n{(Re * ureg.ohms):.1f~#P}", color="blue", loc="bot")
        .length(3)
    )
    d += (
        e.GroundChassis()
        .label(
            f"\n$\\beta$={fba['beta']:.1f}@{(F*ureg.hertz):.1f~#P}",
            color="red",
            loc="bot",
        )
        .label(f"\n $I_e$={(Ie * ureg.ampere):.1f~#P}", color="blue", loc="right")
    )

    d.pop()
    d += (
        e.SourceControlledI(color=intrinsic_color)
        .length(2)
        .reverse()
        .label(
            "$\\beta i_b$",
            color="black",
            loc="bot",
        )
    )
    d += e.Dot(open=True, color=intrinsic_color).label(
        "c", loc="right", ofst=(0.1, 0.2), color=intrinsic_color
    )
    if N != 1:
        d.push()
        d += e.Gap().up().length(3)
        d += (
            e.Gap()
            .right()
            .length(4)
            .label(
                "(intermediate) ${Z_{out}$" + f"\n{fba['izout']:.3f~S}",
                loc="right",
                color="red",
            )
        )
        d.pop()
    d.push()
    d += e.Line(color=intrinsic_color).left().length(1)
    d += (
        e.Capacitor(color=intrinsic_color)
        .length(2)
        .label("$C_{cb}$" + f"\n{(Ccb * ureg.farads):.1f~#P}", color="blue")
        .down()
    )
    d += e.Dot(color=intrinsic_color)
    d.pop()

    if Cce is not None:
        d.push()
        d += e.Line(color=intrinsic_color).right().length(1.5)
        d += (
            e.Capacitor(color=intrinsic_color)
            .length(4)
            .label(
                "$C_{ce}$" + f"\n{(Cce * ureg.farads):.1f~#P}", color="blue", loc="bot"
            )
            .down()
        )
        d += e.Line(color=intrinsic_color).left().length(1.5)
        d.pop()

    d += (
        e.Inductor(color=extrinsic_color)
        .up()
        .length(2)
        .label(
            "$extL_c$" + f"\n{(extLc * ureg.henrys):.2f~#P}", color="blue", loc="bot"
        )
    )
    d += e.Dot(color=extrinsic_color)
    d.push()

    d += (
        e.Capacitor(color=extrinsic_color)
        .left()
        .length(6.5)
        .label("$extC_{rev}$" + f"\n{(extCrev * ureg.farads):.1f~#P}", color="blue")
    )
    d += e.Line(color=extrinsic_color).down().length(4)

    # feedback
    d.pop()
    d.push()
    d += e.Line().up().length(2)
    d += (topDot := e.Dot())
    d += (
        e.Capacitor(color=feedback_color)
        .label(f"$C_f$\n{(Cf * ureg.farads):.1f~#P}", color="blue")
        .left()
        .length(4)
    )
    d += (
        e.Inductor(color=feedback_color)
        .label("$L_f$" + f"\n{(Lf * ureg.henrys):.1f~#P}", color="blue")
        .flip()
        .length(4)
    )
    d += e.Line(color=feedback_color).length(1)
    d += (
        e.Resistor(color=feedback_color)
        .label("$R_f$" + f"\n{(Rf * ureg.ohms):.1f~#P}", color="blue")
        .down()
        .length(6)
    )
    d += e.Dot()

    d.pop()
    d += e.Line(color=extrinsic_color).right().length(4)
    d += (
        e.Capacitor(color=extrinsic_color)
        .down()
        .length(8)
        .label(
            "$extC_{out}$" + f"\n{(extCout * ureg.farads):.1f~#P}",
            loc="bot",
            color="blue",
        )
    )
    d += e.Line(color=extrinsic_color).left().length(4)

    d += e.Line().at(topDot.center).right().length(7)
    d += e.Line().down().length(4)
    n1 = N
    n2 = 1
    if N < 0:
        n1 = 1
        n2 = abs(N)
    d += (
        TR1 := e.Transformer(t1=n1, t2=n2)
        .right()
        .label(f"ideal\nt:{n1}:{n2}\nz:{n1**2}:{n2**2}", color="blue")
        .flip()
    )

    d += e.Line().at(TR1.p1).length(0.5)
    d += e.GroundChassis()

    d += e.Line().at(TR1.s1).length(0.5)
    d += e.GroundChassis()

    d += e.Gap().down().length(1)
    d += (
        e.Gap()
        .right()
        .length(0.5)
        .label(
            "${Z_{out}$"
            + f"\n{fba['zout']:.3f~S}\nReturn Loss={(fba['OutRetLoss'] * ureg.decibel):.3f~#P}\nvswr={(fba['OutVSWR']):.2f}",
            loc="right",
            halign="left",
            color="red",
        )
    )

    d += e.Line().at(TR1.s2).right().length(5)
    d += (
        e.Resistor()
        .label("$Z_L$" + f"\n{ZL.as_complex()}", loc="bot", color="blue")
        .down()
        .length(3)
    )
    d += e.GroundChassis().label(
        "Gain\n$G_t$" + f"={(fba['Gt_db'] * ureg.decibel):.4f~#P}",
        color="red",
        loc="right",
    )
    d += e.Gap().right()  # make space on right for truncated ZL

    # Legend
    d += (
        e.Line(arrow="o-|", color=intrinsic_color)
        .length(2)
        .at(xy=(22, -5))
        .label("intrinsic", loc="right")
    )
    d += (
        e.Line(arrow="o-|", color=extrinsic_color)
        .length(2)
        .at(xy=(22, -5.5))
        .label("extrinsic", loc="right")
    )
    d += (
        e.Line(arrow="o-|", color=feedback_color)
        .length(2)
        .at(xy=(22, -6))
        .label("feedback", loc="right")
    )

    display(d)

    fba_res = {"Sarrs": []}
    for f in np.logspace(plot_from_exp, plot_to_exp, num=200):
        fba = _calc_complex_fba(
            ZS=ZS,
            ZL=ZL,
            B0=B0,
            F=f,
            FT=FT,
            Rbp=Rbp,
            extLb=extLb,
            extCin=extCin,
            Ccb=Ccb,
            Cce=Cce,
            Ie_mA=Ie_mA,
            Re=Re,
            extLe=extLe,
            Rf=Rf,
            Lf=Lf,
            Cf=Cf,
            extLc=extLc,
            extCout=extCout,
            extCrev=extCrev,
            N=N,
        )

        # collate a list of S-param matrices for scikit-rf network
        fba_res["Sarrs"].append(
            [
                [fba["S"].s11.c, fba["S"].s12.c],
                [fba["S"].s21.c, fba["S"].s22.c],
            ]
        )

        for f_k, f_i in fba.items():
            if f_k not in fba_res:
                fba_res[f_k] = []
            fba_res[f_k].append(f_i)

    fig = make_subplots(
        rows=3,
        cols=4,
        subplot_titles=(
            "|Beta|",
            "Gain dB",
            "I/P Return Loss dB",
            "O/P Return Loss dB",
            "Linvill Stability [>0 & <1]",
            "Stern Stability [>1]",
            "I/P VSWR",
            "O/P VSWR",
            "Insertion Gain dB",
            "Rollett Stability",
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

    colors = ["red" if v <= 0 or v >= 1 else "blue" for v in fba_res["linvillC"]]
    fig.add_trace(
        go.Scatter(
            x=fba_res["F"],
            y=fba_res["linvillC"],
            name="Linvill Stability",
            mode="markers+lines",
            marker={"color": colors, "size": 3},
            line={"color": intrinsic_color},
        ),
        row=2,
        col=1,
    )

    colors = ["red" if v <= 1 else "blue" for v in fba_res["sternK"]]
    fig.add_trace(
        go.Scatter(
            x=fba_res["F"],
            y=fba_res["sternK"],
            name="Stern Stability",
            mode="markers+lines",
            marker={"color": colors, "size": 3},
            line={"color": intrinsic_color},
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Scatter(x=fba_res["F"], y=fba_res["InVSWR"], name="i/p VSWR"),
        row=2,
        col=3,
    )
    fig.add_trace(
        go.Scatter(x=fba_res["F"], y=fba_res["OutVSWR"], name="o/p VSWR"),
        row=2,
        col=4,
    )

    fig.add_trace(
        go.Scatter(
            x=fba_res["F"],
            y=fba_res["insertion_gain_db"],
            name="insertion gain db |S21|^2",
        ),
        row=3,
        col=1,
    )

    rollett_colors = ["red" if v <= 1 else "blue" for v in fba_res["rollett_stability"]]
    fig.add_trace(
        go.Scatter(
            x=fba_res["F"],
            y=fba_res["rollett_stability"],
            name="Rollett Stability",
            mode="markers+lines",
            marker={"color": rollett_colors, "size": 3},
            line={"color": intrinsic_color},
        ),
        row=3,
        col=2,
    )

    fig.add_vline(
        x=F,
        line_width=1,
        line_dash="dash",
        line_color="red",
    )

    fig.update_layout(height=650, width=1400)
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log", row=2, col=1)
    fig.update_yaxes(type="log", row=2, col=2)
    fig.update_yaxes(type="log", row=2, col=3)
    fig.update_yaxes(type="log", row=2, col=4)
    fig.show()

    ntw = plot_smith_annotated(frequency=fba_res["F"], s=fba_res["Sarrs"], F=F)

    # fig2 = plt.figure(figsize=(12, 12))
    # ax11 = fig2.add_subplot(221)
    # ax12 = fig2.add_subplot(222, projection="polar")
    # ax21 = fig2.add_subplot(223, projection="polar")
    # ax22 = fig2.add_subplot(224)

    # def _annot_point(ax=None, x=None, y=None, f=None):
    #     font_size = 8
    #     if "PolarAxes" in str(type(ax)):
    #         c = Complex(complex(x, y))
    #         p = c.as_polar()
    #         theta = math.radians(p["angle"])
    #         r = p["mag"]
    #         ax.scatter(theta, r, marker="v", s=20, color="red")
    #         ax.text(
    #             theta,
    #             r,
    #             f"{(f*ureg.hertz):.0f~#P}",
    #             fontsize=font_size,
    #             ha="center",
    #             va="bottom",
    #             color="red",
    #         )
    #     else:
    #         ax.scatter(x, y, marker="v", s=20, color="red")
    #         # ax.annotate(f'M', (x, y), xytext=(-7, 7), textcoords='offset points', color='red')
    #         ax.text(
    #             x,
    #             y,
    #             f"{(f*ureg.hertz):.0f~#P}",
    #             fontsize=font_size,
    #             ha="center",
    #             va="bottom",
    #             color="red",
    #         )

    # def _annot(ax=None, m=None, n=None, ntw=None):
    #     f = ntw.frequency.f_scaled[0]
    #     x = ntw.s.real[0, m, n]
    #     y = ntw.s.imag[0, m, n]
    #     _annot_point(ax, x, y, f)

    #     f = ntw.frequency.f_scaled[-1]
    #     x = ntw.s.real[-1, m, n]
    #     y = ntw.s.imag[-1, m, n]
    #     _annot_point(ax, x, y, f)

    # ntw.plot_s_smith(m=0, n=0, draw_labels=True, ax=ax11)
    # ntw.plot_s_polar(m=0, n=1, ax=ax12)
    # ntw.plot_s_polar(m=1, n=0, ax=ax21)
    # ntw.plot_s_smith(m=1, n=1, draw_labels=True, ax=ax22)

    # for p in ((ax11, 0, 0), (ax12, 0, 1), (ax21, 1, 0), (ax22, 1, 1)):
    #     ax = p[0]
    #     m = p[1]
    #     n = p[2]
    #     _annot(ax=ax, m=m, n=n, ntw=ntw)

    return ntw


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
extLe_nH = widgets.FloatText(
    value=10.0,
    description="$extLe$ nH",
    style=style,
    layout=Layout(width="auto", grid_area="extLe_nH"),
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
N = widgets.SelectionSlider(
    options=list(range(-20, -1)) + list(range(1, 21)),
    value=2,
    description="$N$",
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
        extLe_nH,
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
            "Rbp Re Re extLe_nH"
            "Ccb_pF Cf_pF Cf_pF Lf_nH"
            "Rf Rf ZS_real ZS_imag"
            "ZL_real ZL_imag N N"
            """,
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
            "extLe_nH": extLe_nH,
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
