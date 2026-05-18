# BAR BPM Analysis Report

- Config: `/Users/wange/Coding/Python/bpm/input_template.yaml`
- Chamber type: `round`
- Boundary perimeter: 314.154 mm
- Longitudinal image-charge denominator: 314.154 mm
- Button capacitance used in signal model: 3.400 pF
- Characteristic impedance: 50.0 ohm

## Linearity

- Linear scale factor `Kx`: 36.254 mm
- Linear scale factor `Ky`: 36.254 mm
- RMS position error before polynomial correction: 5232.18 um
- Max position error before polynomial correction: 8.819 mm
- RMS position error after polynomial correction: 2594.69 um
- Max position error after polynomial correction: 6.835 mm

## Signal Summary

- Peak voltage at button output: 6.581e+00 V
- Peak voltage after cable model: 4.047e-01 V
- Peak voltage after analog filter: 6.493e-03 V
- RMS voltage after analog filter: 1.371e-03 V

## Resolution

- Reference relative voltage error `sigma_V / V`: 0.0016
- Estimated horizontal resolution at the reference point: 29.00 um
- Estimated vertical resolution at the reference point: 29.00 um

## Notes

- The uncapped Gaussian is evaluated on a finite numerical window set by `bunch.longitudinal_grid.no_cut_span_sigma`.
