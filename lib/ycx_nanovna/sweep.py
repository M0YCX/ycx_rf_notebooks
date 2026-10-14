import os
import numpy as np
import pynanovna
from skrf import Network

def _average_num_sweeps(vna, npoints, num_sweeps):
    s11_a = np.empty((num_sweeps, npoints), dtype=np.complex128)
    s21_a = np.empty((num_sweeps, npoints), dtype=np.complex128)
    f = None
    for i in range(0, num_sweeps):
        print(f"sweep {i+1}")
        _s11, _s21, f = vna.sweep()
        for ii in range(0, len(f)):
            s11_a[i][ii] = _s11[ii]
            s21_a[i][ii] = _s21[ii]
    # Average across sweeps
    return (
        np.array(np.mean(s11_a, 0, dtype=np.complex128)),
        np.array(np.mean(s21_a, 0, dtype=np.complex128)),
        f,
    )

def sweep_one_port(
    cal_filename, touchstone_fileprefix=None, touchstone_filesuffix="s1p"
):
    """Sweep one port s11/s21"""
    assert (
        cal_filename != None and cal_filename != ""
    ), "cal_filename must be a valid string and not empty"

    vna = pynanovna.VNA()
    vna.load_calibration(cal_filename)

    cdata = list(vna.calibration.dataset.data)
    start = cdata[0]
    stop = cdata[len(cdata) - 1]
    npoints = len(cdata)
    vna.set_sweep(start, stop, npoints)

    s11, s21, f = vna.sweep()

    npoints = len(f)
    s12 = np.zeros(npoints)
    s22 = np.zeros(npoints)

    S = []
    for i in range(0, npoints):
        S.append(
            [
                [s11[i], s12[i]],
                [s21[i], s22[i]],
            ]
        )

    n = Network(f_unit="Hz", z0=50 + 0j, f=f, s=S)
    if touchstone_fileprefix is not None:
        n.write_touchstone(f"{touchstone_fileprefix}.{touchstone_filesuffix}")
    return n


def sweep_two_port(
    cal_filename,
    touchstone_fileprefix=None,
    touchstone_filesuffix="s2p",
    num_sweeps=3,
):
    """Sweep two ports s11/s12/s21/s22, a bidirectional sweep, requiring
    the NanoVNA port connections to be reversed during the sweep.  This is
    an interactive function, prompting the user to reconnect the NanoVNA ports.
    """
    assert (
        cal_filename != None and cal_filename != ""
    ), "cal_filename must be a valid string and not empty"

    vna = pynanovna.VNA()
    vna.load_calibration(cal_filename)

    cdata = list(vna.calibration.dataset.data)
    start = cdata[0]
    stop = cdata[len(cdata) - 1]
    npoints = len(cdata)
    vna.set_sweep(start, stop, npoints)

    input("Ensure NanoVNA connections are restored for S11/S21 sweep")
    s11, s21, f = _average_num_sweeps(vna, npoints, num_sweeps)

    input("Reverse NanoVNA connections for S12/S22 sweep")
    s22, s12, f2 = _average_num_sweeps(vna, npoints, num_sweeps)

    input("Restore NanoVNA connections")

    npoints = len(f)
    S = []
    for i in range(0, npoints):
        S.append(
            [
                [s11[i], s12[i]],
                [s21[i], s22[i]],
            ]
        )

    n = Network(f_unit="Hz", z0=50 + 0j, f=f, s=S)
    if touchstone_fileprefix is not None:
        n.write_touchstone(f"{touchstone_fileprefix}.{touchstone_filesuffix}")
    return n
