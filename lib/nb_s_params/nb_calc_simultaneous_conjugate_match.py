import pint
from IPython.display import HTML, display

from ycx_complex_numbers import Complex, S, Y, Z
from ycx_rf_amplifiers.s_params import calc_simultaneous_conjugate_match

ureg = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)


def nb_calc_simultaneous_conjugate_match(
    s11=None, s22=None, s21=None, s12=None, title="", Z0=Z(50 + 0j)
):
    for p in (s11, s22, s21, s12):
        if not isinstance(p, S):
            raise TypeError("All inputs must be type S Complex number instances")

    cj = calc_simultaneous_conjugate_match(s11=s11, s22=s22, s21=s21, s12=s12)

    """Simultanueous Conjugate Match from Transistor S-Paramters."""
    cols = 3
    html = f"<hr><h3>{title}</h3>"
    html += "<table>"

    html += f'<tr><th colspan="{cols}" style="text-align:left;">S-Parameters: (from datasheet or measurement)</th>'
    html += f"<tr><td>s11<td>{s11}</td>"
    html += f"<tr><td>s22<td>{s22}</td>"
    html += f"<tr><td>s12<td>{s12}</td>"
    html += f"<tr><td>s21<td>{s21}</td>"

    html += f'<tr><th colspan="{cols}" style="text-align:left;">Stability Factor:</th>'
    Ds = cj["Ds"]
    html += f"<tr><td>$D_S$<td>{Ds}</td>"

    K = cj["K"]
    check = "FAILED"  # X
    check_color = "red"
    if K > 1:
        check = "PASSED OK"
        check_color = "green"
    html += f'<tr><td>Rollett $K$<td style="color: {check_color};">{K:.3f}<td style="text-align:left; color: {check_color};">{check}</td>'
    if K <= 1:
        html += '<tr><td colspan="{cols}" style="color:red; text-align:right;">Try calculating using the Stability & Constant Gain Circles notebook...</td></table>'
        display(HTML(html))
        return

    html += f'<tr><th colspan="{cols}" style="text-align:left;">Maximum Available Gain:</th>'

    MAG_db = cj["maximum_available_gain_db"]
    html += f"<tr><td>$MAG$<td>{(MAG_db * ureg.decibel):.3~#P}</td>"

    IFG_db = cj["intrinsic_forward_gain_db"]
    html += f"<tr><td>vs the Intrinsic Forward Gain $dB$<td>{(IFG_db * ureg.decibel):.3~#P}</td>"

    html += f'<tr><th colspan="{cols}" style="text-align:left;">Simultaneous Conjugate Match:</th>'

    gammaS = cj["gammaS"]
    html += f'<tr style="border: 1px solid #b3b2b2"><td>$\Gamma_S$<td>{gammaS}</td>'
    ZS = Z0 * Complex(1 + gammaS.c) / Complex(1 - gammaS.c)
    html += f"<tr><td>$Z_S$<td>{ZS}<td>$Z_0$ = {Z0}</td>"
    YS = Y(1 / ZS.c)
    html += f"<tr><td>$Y_S$<td>{YS}</td>"

    gammaL = cj["gammaL"]
    html += f'<tr style="border: 1px solid #b3b2b2"><td>$\Gamma_L$<td>{gammaL}</td>'
    ZL = Z0 * Complex(1 + gammaL.c) / Complex(1 - gammaL.c)
    html += f"<tr><td>$Z_L$<td>{ZL}<td>$Z_0$ = {Z0}</td>"
    YL = Y(1 / ZL.c)
    html += f"<tr><td>$Y_L$<td>{YL}</td>"

    html += f'<tr><th colspan="{cols}" style="text-align:left;">Transducer Gain:</th>'
    GT_db = cj["transducer_gain_db"]
    html += f"<tr><td>Gain<td>{(GT_db * ureg.decibel):.3~#P}</td>"

    html += "</table>"
    display(HTML(html))
