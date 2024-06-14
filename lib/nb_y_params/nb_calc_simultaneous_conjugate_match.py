import pint
from IPython.display import HTML, display

from ycx_complex_numbers import Complex, Y, Z
from ycx_rf_amplifiers.y_params import calc_simultaneous_conjugate_match

ureg = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)


def nb_calc_simultaneous_conjugate_match(
    yi=None, yo=None, yf=None, yr=None, title=""
):
    """Simultanueous Conjugate Match from Transistor Y-Paramters
    Transistor Y-parameters at a given frequency and bias conditions
    in mmhos:
        yi = input admittance
        yo = output admittance
        yf = forward-transfer admittance
        yr = reverse-transfer admittance
    """
    for p in (yi, yo, yf, yr):
        if not isinstance(p, Y):
            raise TypeError("All inputs must be type Y Complex number instances")

    cj = calc_simultaneous_conjugate_match(yi=yi, yo=yo, yf=yf, yr=yr)

    html = f"<hr><h3>{title}</h3>"

    gi = yi.real
    bi = yi.imag

    go = yo.real
    bo = yo.imag

    # Linvill Stability
    C = cj["linvill_test"]

    check = "✗"
    check_color = "red"
    if C < 1:
        check = "✓"
        check_color = "green"
    # display(HTML(f"<li>Linvill Stability Factor C = {C:.2f} (must be less than 1) {check}"))
    # assert C < 1
    # print(f"C={C}")

    html += "<table>"
    html += f'<tr><th colspan="2" style="text-align:left;">Y-Parameters: (from datasheet or measurement)'
    html += f"<tr><td>yi<td>{yi}"
    html += f"<tr><td>yo<td>{yo}"
    html += f"<tr><td>yf<td>{yf}"
    html += f"<tr><td>yr<td>{yr}"

    html += f'<tr><th colspan="2" style="text-align:left;">Stability Factor:'
    html += f'<tr><td>Linvill $C$<td style="color: {check_color};">{C:.4f}<td style="text-align:left; color: {check_color};">{check}'

    if C < 1:
        # MAG
        mag_db = cj["maximum_available_gain_db"]
        html += f'<tr><th colspan="2" style="text-align:left;">Maximum Available Gain:'
        html += f"<tr><td>MAG $dB$<td>{(mag_db * ureg.decibel):.3~#P}"

        # Simultaneous Conjugate Match:-

        # Source:
        html += '<tr><th colspan="2" style="text-align:left;">Source:'

        YS = cj["YS"]
        html += f"<tr><td>$Y_S$<td>{YS} mmhos"

        ZS = 1 / (YS / 1000)
        html += f"<tr><td>$Z_S$<td>{ZS} ohms"

        tYS = YS.conjugate
        html += f"<tr><td>Transistor i/p $Y$<td>{tYS} mmhos<td>(conjugate of $Y_S$)"

        tZS = ZS.conjugate
        html += f"<tr><td>Transistor i/p $Z$<td>{tZS} ohms<td>(conjugate of $Z_S$)"

        # Load:
        html += '<tr><th colspan="2" style="text-align:left;">Load:'

        YL = cj["YL"]
        html += f"<tr><td>$Y_L$<td>{YL} mmhos"

        ZL = 1 / (YL / 1000)
        html += f"<tr><td>$Z_L$<td>{ZL} ohms"

        tYL = YL.conjugate
        html += f"<tr><td>Transistor o/p $Y$<td>{tYL} mmhos<td>(conjugate of $Y_L$)"

        tZL = ZL.conjugate
        html += f"<tr><td>Transistor o/p $Z$<td>{tZL} ohms<td>(conjugate of $Z_L$)"

        # Transducer Gain
        html += '<tr><th colspan="2" style="text-align:left;">Transducer Gain:'
        GT_db = cj["transducer_gain_db"]
        html += f"<tr><td>Gain $dB$<td>{(GT_db * ureg.decibel):.3~#P}"

    html += "</table>"
    display(HTML(html))
