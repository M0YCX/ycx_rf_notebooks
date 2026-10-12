import os
import numpy as np
import skrf as rf
from matplotlib import pyplot as plt
from skrf import Network

def plot_s2p_3x2(touchstone_fileprefix=None, touchstone_filesuffix="s2p",  ZS=50+0j, ZL=50+0j, n=None):
    assert n is not None or touchstone_fileprefix is not None, "Either n (Network) or touchstone_fileprefix must be specified"
    # ZS = 200 + 0j
    # ZL = 50 + 0j

    if n is None:
        n = Network(f"{touchstone_fileprefix}.{touchstone_filesuffix}")
    # display(n)
    f = n.f

    fig = plt.figure(figsize=(12, 15))
    plt.subplots_adjust(hspace=0.3)
    ax1 = fig.add_subplot(321)

    ax2 = fig.add_subplot(322)
    ax2.grid()
    ax2.set_xscale("log")

    ax3 = fig.add_subplot(323, projection="polar")
    ax3.set_ylim((0, 1))

    ax4 = fig.add_subplot(324)
    ax4.grid()

    ax5 = fig.add_subplot(325)
    ax5.grid()
    ax5.set_xscale("log")

    ax6 = fig.add_subplot(326)
    ax6.grid()
    ax6.set_xscale("log")

    rf.plotting.smith(ax=ax1, draw_labels=True, ref_imm=50.0, chart_type="z")
    n.plot_s_smith(m=0, n=0, ax=ax1, lw=2)
    n.plot_s_smith(m=1, n=1, ax=ax1, lw=2)

    n.frequency.unit = "mhz"
    n.plot_s_db(ax=ax2, title="logmag")
    n.plot_s_polar(m=0, n=1, ax=ax3, lw=2)
    n.plot_s_polar(m=1, n=0, ax=ax3, lw=2)

    zin = []
    zout = []
    vswr_in = []
    vswr_out = []
    for i in range(0, len(f)):
        z = n.z[i]
        z11 = z[0][0]
        z12 = z[0][1]
        z21 = z[1][0]
        z22 = z[1][1]
        _zin = z11 - (z12 * z21) / (z22 + ZL)
        _zout = z22 - (z12 * z21) / (z11 + ZS)
        zin.append(abs(_zin))
        zout.append(abs(_zout))

        gamma_in = (_zin - ZS) / (_zin + ZS)
        vswr_in.append((1 + abs(gamma_in)) / (1 - abs(gamma_in)))

        gamma_out = (_zout - ZL) / (_zout + ZL)
        vswr_out.append((1 + abs(gamma_out)) / (1 - abs(gamma_out)))

    ax4.set_title(f"Impedance [$Z_{{S}}={ZS}\\Omega$, $Z_{{L}}={ZL}\\Omega$]: 4:1")
    ax4.set_xscale("log")
    ax4.set_ylabel("|Z|")
    ax4.set_xlabel("$F_{Hz}$")
    ax4.plot(f, zin, label="$|Z_{in}|$")
    ax4.plot(f, zout, label="$|Z_{out}|$")
    ax4.legend()

    ax5.set_title(f"VSWR [$Z_{{S}}={ZS}\\Omega$, $Z_{{L}}={ZL}\\Omega$]: 4:1")
    ax5.set_xscale("log")
    ax5.set_ylabel("VSWR")
    ax5.set_xlabel("$F_{Hz}$")
    ax5.plot(f, vswr_in, label="$VSWR_{in}$")
    ax5.plot(f, vswr_out, label="$VSWR_{out}$")
    ax5.legend()

    n.plot_s_deg(m=0, n=0, ax=ax6)
    n.plot_s_deg(m=1, n=0, ax=ax6)