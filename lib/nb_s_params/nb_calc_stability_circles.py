import math
import sys

import mplcursors

import pint

from IPython.display import HTML, display
from matplotlib import pyplot as plt
from matplotlib import style
from skrf import Network, plotting

ureg = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
sys.path.append("../../lib")

from ycx_rf_amplifiers.s_params import (
    calc_stability_circles,
    constant_gain_circle,
    calc_circle,
)

from ycx_complex_numbers import Complex, S


def nb_calc_stability_circles(
    s11, s22, s12, s21, gain_db=0, title="", Zi=50 + 0j, Zo=50 + 0j
):
    stab_circles = calc_stability_circles(s11, s22, s12, s21)
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

    html += "</table>"
    display(HTML(html))

    plt.ioff()

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(111)
    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    ax3 = fig.add_subplot(111, sharex=ax1, frameon=False)

    with style.context("seaborn-v0_8-ticks"):
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

        plt.show()
