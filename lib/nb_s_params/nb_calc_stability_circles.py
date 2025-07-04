import math
import sys

# %matplotlib inline
# %matplotlib widget
# %config InlineBackend.figure_format = 'svg'
# import matplotlib_inline
import mplcursors

# matplotlib_inline.backend_inline.set_matplotlib_formats("svg")
# import numpy as np
import pint

# import plotly.graph_objects as go
# import skrf
# import skrf as rf
from IPython.display import HTML, display
from matplotlib import pyplot as plt
from matplotlib import style
from skrf import Network, plotting

ureg = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
sys.path.append("../../lib")

from ycx_rf_amplifiers.s_params import  calc_stability_circles, constant_gain_circle, calc_circle

# from complex_params import Complex, S
from ycx_complex_numbers import Complex, S


# def calc_circle(c, r):
#     theta = np.linspace(0, 2 * np.pi, 50)
#     return c + r * np.exp(1.0j * theta)


def nb_calc_stability_circles(s11, s22, s12, s21, gain_db=0, title="", Zi=50 + 0j, Zo=50 + 0j):
    stab_circles = calc_stability_circles(s11, s22, s12, s21)
    # print(stab_circles)
    Ds = stab_circles["Ds"]
    K = stab_circles["K"]
    B1 = stab_circles["B1"]
    MAG = stab_circles["MAG"]
    MSG = stab_circles["MSG"]
    C1 = stab_circles["C1"]
    C2 = stab_circles["C2"]
    rs1 = stab_circles["rs1"]
    Ps1 = stab_circles["Ps1"]
    rs2 = stab_circles["rs2"]
    Ps2 = stab_circles["Ps2"]

    cols = 3
    html = f"<hr><h2>{title}</h2>"
    html += "<table>"

    html += f'<tr><th colspan="{cols}" style="text-align:left;">S-Parameters:'
    s11alert = (
        '<td style="color: green; text-align: left;">$|s11| < 1$ center is stable</td>'
    )
    if abs(s11.c) >= 1:
        s11alert = '<td style="color: red; text-align: left;">$|s11| >= 1$ center is UNstable</td>'
    html += f"<tr><td>s11<td>{s11} {s11alert}"
    s22alert = (
        '<td style="color: green; text-align: left;">$|s22| < 1$ center is stable</td>'
    )
    if abs(s22.c) >= 1:
        s22alert = '<td style="color: red; text-align: left;">$|s22| >= 1$ center is UNstable</td>'
    html += f"<tr><td>s22<td>{s22} {s22alert}"
    html += f"<tr><td>s12<td>{s12}"
    html += f"<tr><td>s21<td>{s21}"

    html += f"<tr><td>$|s21|^2$<td>{(10 * math.log10(s21.as_polar()['mag']**2) * ureg.decibel):.3~#P}"

    html += f'<tr><th colspan="{cols}" style="text-align:left;">'
    html += (
        f"<tr><td>$D_S$<td>{Ds}</td>"
        + f'<td style="text-align: left;">magnitude is $< 1$: {abs(Ds.c) < 1}</td>'
    )

    check = "$(K \\leq 1)$ NOT Unconditionally Stable"  # X
    check_color = "red"
    if K > 1:
        check = "OK - Unconditionally Stable"
        check_color = "green"
    html += f'<tr><td>Rollett $K$<td style="color: {check_color};">{K:.3f}<td style="text-align:left; color: {check_color};">{check}'

    if K > 1:
        html += f'<tr><th colspan="{cols}" style="text-align:left;">Maximum Available Gain:</th>'
        html += f"<tr><td>$B_1$<td>{B1:.3f}</td>"
        html += f"<tr><td>$MAG$<td>{(MAG * ureg.decibel):.3~#P}</td>"
    else:
        html += f'<tr><th colspan="{cols}" style="text-align:left;">Maximum Stable Gain:</th>'
        html += f"<tr><td>$MSG$<td>{(MSG * ureg.decibel):.3~#P}</td>"

    html += f'<tr><th colspan="{cols}" style="text-align:left;">Stability Circles:'
    html += f'<tr><td>$C_1$<td>{C1} <td style="text-align: left">Conjugate: {C1.as_conjugate()}'

    html += f'<tr><td>$C_2$<td>{C2} <td style="text-align: left">Conjugate: {C2.as_conjugate()}'

    html += f'<tr><th colspan="{cols}" style="text-align:left; color:orange">Input Stability Circle:'
    html += "<tr><td>$r_{s1}$<td>" + f"{rs1}"

    html += (
        "<tr><td>$P_{s1}$<td>"
        + f"{Ps1:.4f}"
        + '</td><td style="color:orange;text-align:left">&#x274D;</td>'
    )

    html += f'<tr><th colspan="{cols}" style="text-align:left; color:red">Output Stability Circle:'
    html += "<tr><td>$r_{s2}$<td>" + f"{rs2}"

    html += (
        "<tr><td>$P_{s2}$<td>"
        + f"{Ps2:.4f}"
        + '</td><td style="color:red;text-align:left">&#x274D;</td>'
    )

    cs1 = calc_circle(rs1.c, Ps1)
    cs2 = calc_circle(rs2.c, Ps2)

    if gain_db > 0:
        html += f'<tr><th colspan="{cols}" style="text-align:left; color:green">Constant Gain Circle:'
        html += "<tr><td>desired gain<td>" + f"{gain_db} dB"
        cgc = constant_gain_circle(gain_db, s11, s22, s12, s21)
        s = [[cgc["ro"].c]]
        html += (
            "<tr><td>$r_o$<td>"
            + f"{cgc['ro']}"
            + '</td><td style="color:green;text-align:left">&#x2715;</td>'
        )
        html += (
            "<tr><td>$P_o$<td>"
            + f"{cgc['Po']:.4f}"
            + '</td><td style="color:green;text-align:left">&#x274D;</td>'
        )
        # html += "<tr><td>circle_points<td>" + f"{cgc['circle_points']}"

    html += "</table>"
    display(HTML(html))

    # print(cs1)
    plt.ioff()

    # def on_plot_hover(event):
    #     print("in on_plot_hover")
    #     # Iterating over each data member plotted
    #     for curve in plot.get_lines():
    #         # Searching which data member corresponds to current mouse position
    #         if curve.contains(event)[0]:
    #             print("over %s" % curve.get_gid())

    fig = plt.figure(figsize=(12, 8))
    # fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    # fig.canvas.layout.width = "100%"
    # fig.canvas.layout.height = "1900px"
    # fig = plt.figure()
    # fig.canvas.mpl_connect("motion_notify_event", on_plot_hover)
    # fig = go.Figure()
    ax1 = fig.add_subplot(111)
    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    ax3 = fig.add_subplot(111, sharex=ax1, frameon=False)

    with style.context("seaborn-v0_8-ticks"):
        # plotting.plot_smith(s21.c, marker="x", lw=2, color="black", ax=ax1)
        # plotting.plot_smith(s, marker="o")
        plotting.plot_smith(
            cs1,
            show_legend=True,
            label="input stability",
            lw=2,
            color="orange",
            ax=ax1,
            title="",
        )
        plotting.plot_smith(
            cs2,
            show_legend=True,
            label="output stability",
            lw=2,
            color="red",
            ax=ax1,
            title="",
        )

        if gain_db > 0:
            plotting.plot_smith(
                s,
                marker="x",
                label=f"constant gain @{gain_db}dB",
                lw=2,
                color="green",
                ax=ax1,
                title="",
            )
            plotting.plot_smith(
                cgc["circle_points"],
                # label=f"constant gain @{gain_db}dB",
                lw=2,
                color="green",
                ax=ax2,
                title="",
            )
            # print(f"ax1={vars(ax1)}")
            # for a in vars(fig).keys():
            #     print(f"{a}={vars(fig)[a]}")
            # print(f"_mouseover_set={vars(ax1._mouseover_set._od)}")
            # print(f"ax1 children={ax1._children}")
            # for child in ax1._children:
            #     print(f"child={child}")
            # # print(f"fig.patches={fig.patches}")
            # print(f"traces={fig.select_traces()}")

            # crs.connect(
            #     "add",
            #     lambda sel: sel.annotation.set_text(
            #         f"$R_L$={Complex(sel.target[0] + (1j * sel.target[1]))}\n$R_S$="
            #     ),
            # )
            csr = mplcursors.cursor(ax2, hover=False)

            @csr.connect("add")
            def on_add(sel):
                try:
                    gammaL = Complex(sel.target[0] + (1j * sel.target[1]))
                    zL = Zo * (1 + gammaL.c) / (1 - gammaL.c)
                    gammaS = Complex(
                        s11.c + (s12.c * s21.c * gammaL.c) / (1 - (gammaL.c * s22.c))
                    ).as_conjugate()
                    zS = Zi * (1 + gammaS.c) / (1 - gammaS.c)
                    ax3.clear()
                    plotting.plot_smith(
                        gammaS.c, marker="o", lw=2, color="orange", ax=ax3, title=""
                    )
                    # sel.extras.append(csr.add_highlight(a))
                except Exception as e:
                    sel.annotation.set_text(f"{e}")
                else:
                    sel.annotation.set_text(
                        f"$\\Gamma_L$={gammaL}, $Z_L$={zL:.4f}\n$\\Gamma_S$={gammaS}, $Z_S$={zS:.4f}\n(SimNEC: Use Conjugate of $Z_S$ & $Z_L$ for $Z$)"
                    )

        # @csr.connect("remove")
        # def on_remove(sel):
        #     try:
        #         gammaS = Complex(
        #             s11.c + (s12.c * s21.c * gammaL.c) / (1 - (gammaL.c * s22.c))
        #         ).as_conjugate()
        #         plotting.plot_smith(gammaS.c, marker="o", lw=2, color="green", ax=ax1)
        #     except Exception as e:
        #         sel.annotation.set_text(f"{e}")

        plt.show()


# gain_db = 12
# s11 = S().from_polar(0.4, 280)
# s22 = S().from_polar(0.78, 345)
# s12 = S().from_polar(0.048, 65)
# s21 = S().from_polar(5.4, 103)

# stability_circles(
#     s11, s22, s12, s21, gain_db, title="Book Example 2N5179 @200MHz [page: 153]"
# )