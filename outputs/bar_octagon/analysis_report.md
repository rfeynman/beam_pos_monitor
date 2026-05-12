# BAR BPM Analysis Report

- Config: `/Users/wange/Coding/Python/bpm/bar_bpm_octagon.yaml`
- Chamber type: `polygon`
- Boundary perimeter: 201.244 mm
- Button capacitance used in signal model: 3.400 pF
- Characteristic impedance: 50.0 ohm

## Linearity

- Linear scale factor `Kx`: 14.689 mm
- Linear scale factor `Ky`: 24.667 mm
- RMS position error before polynomial correction: 2430.75 um
- Max position error before polynomial correction: 6.650 mm
- RMS position error after polynomial correction: 25.04 um
- Max position error after polynomial correction: 0.083 mm

## Signal Summary

- Peak voltage at button output: 8.978e-04 V
- Peak voltage after cable model: 2.466e-04 V
- Peak voltage after analog filter: 1.658e-08 V
- RMS voltage after analog filter: 7.463e-09 V

## Resolution

- Reference relative voltage error `sigma_V / V`: 0.0016
- Estimated horizontal resolution at the reference point: 11.75 um
- Estimated vertical resolution at the reference point: 19.73 um

## Notes

- The sample polygon was taken from the BAR note figures because `BeampositionMonitor.gdf` was not found in the local Research tree.
- The uncapped Gaussian is evaluated on a finite numerical window set by `bunch.longitudinal_grid.no_cut_span_sigma`.
- The configured band-pass filter suppresses most of the long-bunch spectrum, so the filtered time-domain voltage is very small.
