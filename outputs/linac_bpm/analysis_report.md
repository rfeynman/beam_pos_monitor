# LINAC BPM Analysis Report

- Config: `/Users/wange/Coding/Python/bpm/linac_bpm.yaml`
- Chamber type: `round`
- Boundary perimeter: 314.154 mm
- Longitudinal image-charge denominator: 314.154 mm
- Button capacitance used in signal model: 3.400 pF
- Characteristic impedance: 50.0 ohm
- Effective cable attenuation frequency `fc`: 4.657e+08 Hz

## Beam Parameters

- Default signal case used for the main results: `1nC, 1mm`
- Bunch charge: 1 nC
- Density kind: `gaussian`
- Input Gaussian sigma: 1 mm
- Gaussian cutoff_sigma: 0
- RMS sigma reconstructed from the normalized longitudinal profile: 1 mm
- Longitudinal grid_number: 200
- No-cut span sigma: 8

## Linearity

- Linear scale factor `Kx`: 36.444 mm
- Linear scale factor `Ky`: 36.444 mm
- RMS position error before polynomial correction: 5328.45 um
- Max position error before polynomial correction: 8.947 mm
- RMS position error after polynomial correction: 260.08 um
- Max position error after polynomial correction: 0.930 mm

## Signal Summary

- Peak voltage at button output: 1.046e+01 V
- Peak voltage after cable model: 1.138e+00 V
- Peak voltage after analog filter: 1.834e-02 V
- RMS voltage after analog filter: 3.872e-03 V

## Resolution

- Reference relative voltage error `sigma_V / V`: 0.0016
- Estimated horizontal resolution at the reference point: 29.15 um
- Estimated vertical resolution at the reference point: 29.15 um

## Notes

- The uncapped Gaussian is evaluated on a finite numerical window set by `bunch.longitudinal_grid.no_cut_span_sigma`.
