import math
import pint
import matplotlib.pyplot as plt
import skrf as rf
from skrf import Network, plotting
from ycx_complex_numbers import Complex

ureg = pint.UnitRegistry()
ureg = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)

def plot_smith_annotated(frequency=None, s=None):
    ntw = rf.Network(frequency=frequency, s=s)

    fig2 = plt.figure(figsize=(12, 12))
    ax11 = fig2.add_subplot(221)
    ax12 = fig2.add_subplot(222, projection="polar")
    ax21 = fig2.add_subplot(223, projection="polar")
    ax22 = fig2.add_subplot(224)

    def _annot_point(ax=None, x=None, y=None, f=None):
        font_size = 8
        if "PolarAxes" in str(type(ax)):
            c = Complex(complex(x, y))
            p = c.as_polar()
            theta = math.radians(p["angle"])
            r = p["mag"]
            ax.scatter(theta, r, marker="v", s=20, color="red")
            ax.text(
                theta,
                r,
                f"{(f*ureg.hertz):.0f~#P}",
                fontsize=font_size,
                ha="center",
                va="bottom",
                color="red",
            )
        else:
            ax.scatter(x, y, marker="v", s=20, color="red")
            # ax.annotate(f'M', (x, y), xytext=(-7, 7), textcoords='offset points', color='red')
            ax.text(
                x,
                y,
                f"{(f*ureg.hertz):.0f~#P}",
                fontsize=font_size,
                ha="center",
                va="bottom",
                color="red",
            )

    def _annot(ax=None, m=None, n=None, ntw=None):
        f = ntw.frequency.f_scaled[0]
        x = ntw.s.real[0, m, n]
        y = ntw.s.imag[0, m, n]
        _annot_point(ax, x, y, f)

        f = ntw.frequency.f_scaled[-1]
        x = ntw.s.real[-1, m, n]
        y = ntw.s.imag[-1, m, n]
        _annot_point(ax, x, y, f)

    ntw.plot_s_smith(m=0, n=0, draw_labels=True, ax=ax11)
    ntw.plot_s_polar(m=0, n=1, ax=ax12)
    ntw.plot_s_polar(m=1, n=0, ax=ax21)
    ntw.plot_s_smith(m=1, n=1, draw_labels=True, ax=ax22)

    for p in ((ax11, 0, 0), (ax12, 0, 1), (ax21, 1, 0), (ax22, 1, 1)):
        ax = p[0]
        m = p[1]
        n = p[2]
        _annot(ax=ax, m=m, n=n, ntw=ntw)

    return ntw
