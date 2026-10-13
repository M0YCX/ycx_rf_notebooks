import os
import pynanovna


def calibrate_steps(cal_filename, start=1e6, stop=200e6, npoints=101):
    assert (
        cal_filename != None and cal_filename != ""
    ), "cal_filename must be a valid string and not empty"

    vna = pynanovna.VNA(logging_level="info")
    display(vna.info())

    vna.set_sweep(start, stop, npoints)

    for step in ("open", "short", "load", "isolation", "through"):
        input(f"Calibrate {step.upper()}, press enter when ready")
        vna.calibration_step(step)

    vna.calibrate()
    vna.save_calibration(cal_filename)


def calibrate(cal_filename, start=1e6, stop=200e6, npoints=101):
    if os.path.isfile(cal_filename):
        resp = input(
            f"calibration file {cal_filename} already exists, do you want to rerun calibration and create a new file(y/n)?"
        )
        if resp.upper() == "Y":
            calibrate_steps(cal_filename, start=start, stop=stop, npoints=npoints)
        else:
            print(f"Skipping calibration, using exising file {cal_filename}")
    else:
        calibrate_steps(cal_filename, start=start, stop=stop, npoints=npoints)
