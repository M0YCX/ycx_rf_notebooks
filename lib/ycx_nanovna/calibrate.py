import pynanovna

def calibrate(cal_filename, start=1e6, stop=200e6, npoints=101):
    assert cal_filename != None and cal_filename != "", "cal_filename must be a valid string and not empty"

    vna = pynanovna.VNA(logging_level="info")
    display(vna.info())

    vna.set_sweep(start, stop, npoints)

    for step in ("open", "short", "load", "isolation", "through"):
        input(f"Calibrate {step.upper()}, press enter when ready")
        vna.calibration_step(step)

    vna.calibrate()
    vna.save_calibration(cal_filename)