# RCS BPM Analysis Report

- Config: `/Users/wange/Coding/Python/bpm/RCS_bpm.yaml`
- Chamber type: `round`
- Boundary perimeter: 119.379 mm
- Longitudinal image-charge denominator: 119.379 mm
- Button capacitance used in signal model: 3.400 pF
- Characteristic impedance: 50.0 ohm
- Effective cable attenuation frequency `fc`: 4.657e+08 Hz

## Beam Parameters

- Default signal case used for the main results: `28nC, 6mm`
- Bunch charge: 28 nC
- Density kind: `gaussian`
- Input Gaussian sigma: 6 mm
- Gaussian cutoff_sigma: 0
- RMS sigma reconstructed from the normalized longitudinal profile: 6 mm
- Longitudinal grid_number: 200
- No-cut span sigma: 8

## Linearity

- Linear scale factor `Kx`: 20.573 mm
- Linear scale factor `Ky`: 20.573 mm
- RMS position error before polynomial correction: 7133.72 um
- Max position error before polynomial correction: 11.607 mm
- RMS position error after polynomial correction: 722.59 um
- Max position error after polynomial correction: 2.733 mm

## Signal Summary

- Peak voltage at button output: 1.090e+02 V
- Peak voltage after cable model: 6.889e+00 V
- Peak voltage after analog filter: 2.777e-01 V
- RMS voltage after analog filter: 5.853e-02 V

## Resolution

- Reference relative voltage error `sigma_V / V`: 0.0016
- Estimated horizontal resolution at the reference point: 16.46 um
- Estimated vertical resolution at the reference point: 16.46 um

## Notes

- The uncapped Gaussian is evaluated on a finite numerical window set by `bunch.longitudinal_grid.no_cut_span_sigma`.
