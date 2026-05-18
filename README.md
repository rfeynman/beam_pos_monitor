# BPM Boundary-Element Reproduction Manual

This directory contains a Python tool that reproduces the BAR BPM-style analysis with a YAML-driven workflow.

Main files:

- [bpm_analysis.py](/Users/wange/Coding/Python/bpm/bpm_analysis.py): solver, plotting, and report generator
- [bar_bpm_octagon.yaml](/Users/wange/Coding/Python/bpm/bar_bpm_octagon.yaml): BAR octagon sample case
- [input_template.yaml](/Users/wange/Coding/Python/bpm/input_template.yaml): generic YAML template

Run:

```bash
python3 /Users/wange/Coding/Python/bpm/bpm_analysis.py /Users/wange/Coding/Python/bpm/bar_bpm_octagon.yaml
```

Sample outputs:

- [figure_11_linearity.png](/Users/wange/Coding/Python/bpm/outputs/bar_octagon/figure_11_linearity.png)
- [figure_12_polyfit.png](/Users/wange/Coding/Python/bpm/outputs/bar_octagon/figure_12_polyfit.png)
- [figure_13_resolution.png](/Users/wange/Coding/Python/bpm/outputs/bar_octagon/figure_13_resolution.png)
- [signal_summary.png](/Users/wange/Coding/Python/bpm/outputs/bar_octagon/signal_summary.png)
- [analysis_report.md](/Users/wange/Coding/Python/bpm/outputs/bar_octagon/analysis_report.md)

## 1. Physics, Equations, and Code Structure

This code has two physics blocks:

1. longitudinal button signal calculation
2. transverse BPM position calculation with a 2D boundary-element method

They are coupled only loosely. The longitudinal block estimates signal amplitude and filtered voltage. The transverse block estimates button charge sharing, linearity, polynomial correction, and resolution.

### 1.1 Longitudinal signal model

The BAR note models the image charge on a button as the convolution of the bunch line-charge density `rho(z)` with the button longitudinal width `w(z)`.

For a round button with radius `r_b`, the longitudinal width is:

```text
w(z) = 2 * sqrt(r_b^2 - z^2),   |z| <= r_b
```

Call this **Eq. (1.1)**.

and zero outside the button.

The image charge is computed numerically as:

```text
q_img(t) = integral rho(z - c t) w(z) dz / P
```

Call this **Eq. (1.2)**.

where:

- `c` is the speed of light
- `P` is the geometric normalization for the image charge pickup fraction

For the BAR/SLAC button formula, `P = 2*pi*b`, where `b` is the chamber radius or the transverse chamber dimension used by the model. In `bar_bpm_octagon.yaml`, the BAR table gives a 20 mm half-height, so the paper-matching value is `2*pi*20 mm = 125.663706 mm`. If `signal_model.image_charge_denominator_mm` is not supplied, the code falls back to the chamber perimeter as a generic non-round approximation.

The code path is:

- `build_line_density(...)`
- `button_width_kernel(...)`
- `signal_chain(...)`

Implementation details:

1. The YAML bunch profile is converted to a normalized line-charge density in `C/m`.
2. The profile is sampled on a uniform `z` grid.
3. The button width kernel is built from the button radius.
4. A discrete convolution gives image charge versus time.
5. A numerical derivative gives image current.

That derivative is:

```text
I_img(t) = d q_img(t) / d t
```

Call this **Eq. (1.3)**.

### 1.2 Button impedance model

The button is modeled as a capacitance `C_b` shunting a `50 ohm` line:

```text
Z_b(omega) = 1 / (1/Z0 + i omega C_b)
```

Call this **Eq. (1.4)**.

with:

- `Z0`: characteristic impedance, usually `50 ohm`
- `C_b`: button capacitance

The code transforms the image current to frequency domain, multiplies by `Z_b(omega)`, and transforms back to time domain:

```text
V_b(omega) = Z_b(omega) I_img(omega)
```

Call this **Eq. (1.5)**.

This is implemented in `signal_chain(...)`.

### 1.3 Cable and analog filter model

The BAR note uses a skin-effect cable model. In magnitude-only form, the attenuation is:

```text
|H_cable(f)| = exp(-sqrt(f / f_c))
```

For time-domain signals such as BAR Fig. 5, the code uses the complex form:

```text
H_cable(f) = exp(-(1 + i) sqrt(f / f_c))
```

Call this **Eq. (1.6)**.

The real exponential gives the quoted cable attenuation. The phase term is important because it gives the dispersive delay of the cable output relative to the button input. Without that phase term, the inverse FFT behaves like a zero-phase smoothing operation and the cable output is not delayed correctly.

After that, the analog front-end is modeled with standard filter transfer functions:

- `none`
- `lowpass_butter`
- `lowpass_bessel`
- `bandpass_butter`

This is not an exact circuit-level reproduction of a measured prototype board. It is a practical system model that lets you sweep cutoff, center frequency, and bandwidth from YAML.

For Fig. 7-style frequency plots, the code evaluates the spectra on a zero-padded FFT grid fine enough to resolve the narrowest comparison-filter feature. This matters for the 500 MHz, 20 MHz-bandwidth BPF: if the frequency-bin spacing is too coarse, the plotted transfer function can miss the passband peak and falsely show a much smaller `|H_BPF|`.

### 1.4 2D boundary-element method for BPM position

For the position calculation, the code follows the standard electrostatic BPM assumption used in the BAR note, Shintake, and related BPM literature:

- the beam is ultra-relativistic
- the problem is reduced to a 2D electrostatic cross-section
- the chamber wall is a perfect conductor
- the beam is represented as a transverse point charge

The Green function in 2D is:

```text
G(r, r') = (1 / 2 pi epsilon0) ln(1 / |r - r'|)
```

Call this **Eq. (1.7)**.

The chamber boundary is discretized into many short line segments. On each segment, the surface charge density is assumed constant. This gives the matrix system:

```text
[phi_i] = (rho0 / 2 pi epsilon0) [G_i0] + (1 / 2 pi epsilon0) [G_ij][sigma_j]
```

Because the conductor boundary is at zero potential:

```text
[sigma_j] = -rho0 [G_ij]^-1 [G_i0]
```

Call the inversion relation **Eq. (1.8)**.

The code does exactly this:

1. build the chamber boundary
2. split it into `N` short elements
3. compute the influence matrix `G_ij`
4. invert the matrix once
5. for each beam position, compute the induced boundary charge
6. integrate charge over each button arc/segment group

Relevant functions:

- `build_boundary(...)`
- `build_green_matrix(...)`
- `compute_button_charges(...)`

### 1.5 Difference-over-sum coordinates

Once four button charges are known, the measured normalized BPM coordinates are:

```text
Dx = (V_B + V_C - V_A - V_D) / (V_A + V_B + V_C + V_D)
Dy = (V_A + V_B - V_C - V_D) / (V_A + V_B + V_C + V_D)
```

This is the four-corner layout used for `buttons.layout: corners`. Call it **Eq. (1.9a)**.

For a round BPM with four buttons on the cardinal axes, the code also supports:

```text
Dx = (V_R - V_L) / (V_T + V_R + V_B + V_L)
Dy = (V_T - V_B) / (V_T + V_R + V_B + V_L)
```

This is the four-cardinal layout used for `buttons.layout: cardinal`. Call it **Eq. (1.9b)**.

Here the code uses induced charge from the electrostatic solve in place of voltage because the relative button sharing determines the position.

Near the center:

```text
X = Kx * Dx
Y = Ky * Dy
```

Call this **Eq. (1.10)**.

The code determines `Kx` and `Ky` from a linear fit around the origin:

- horizontal fit along `y = 0`
- vertical fit along `x = 0`

Implemented in:

- `button_difference_coordinates(...)`
- `fit_scale_factors(...)`

### 1.6 Linearity and polynomial correction

The raw linear BPM estimate is accurate only near the center. Away from center the mapping is nonlinear, especially for non-round chambers.

To reproduce BAR Fig. 11 and Fig. 12 behavior, the code:

1. generates a rectangular grid of true beam positions
2. computes measured BPM coordinates `(X, Y)`
3. fits a 2D polynomial map from measured coordinates back to true coordinates

The mapping is:

```text
x(X,Y) = sum Cx_ij X^i Y^j
y(X,Y) = sum Cy_ij X^i Y^j
```

Call this **Eq. (1.11)**.

with order set by YAML, typically `5`.

Implemented in:

- `fit_polynomial_map(...)`
- `apply_polynomial_map(...)`

### 1.7 Resolution model

The BAR note uses the standard small-offset approximation:

```text
sigma_x ~= Kx * sigma_V / (2 V)
sigma_y ~= Ky * sigma_V / (2 V)
```

Call this **Eq. (1.12)**.

The code plots `sigma_x` and `sigma_y` versus relative voltage error `sigma_V / V`.

Implemented in:

- `resolution_curves(...)`
- `plot_resolution(...)`

### 1.8 What the code actually computes

Given a YAML file, the solver produces:

1. a BPM linearity plot similar to BAR Fig. 11
2. a polynomial-corrected plot similar to BAR Fig. 12
3. a BPM resolution plot similar to BAR Fig. 13
4. a three-panel signal summary plot: bunch current/button width, raw image current/button voltage, and LPF/BPF filtered voltage
5. a Markdown analysis report with key numbers

### 1.9 Important modeling assumptions

- The transverse BPM solve is 2D electrostatic, not full 3D EM.
- Button coverage on the boundary is determined by distance from the declared button center to boundary-element midpoints.
- For polygon chambers, the exact result depends on the supplied polygon points.
- The BAR sample polygon in this repository was inferred from `BAR BPM.pdf`, because `BeampositionMonitor.gdf` was not found in the local research tree.
- The analog filter is represented by standard Butterworth filters, not by a detailed PCB netlist.

## 2. YAML Input Manual

The YAML file is the full user interface for the tool. Each section is described below.

## 2.1 Top-level structure

The expected top-level blocks are:

```yaml
bunch:
chamber:
buttons:
filter:
beam_grid:
output:
```

Each block has a specific meaning.

## 2.2 `bunch`: longitudinal beam definition

Example:

```yaml
bunch:
  charge_nC: 1.0
  density:
    kind: gaussian
    sigma_mm: 62.0
    cutoff_sigma: 0.0
  longitudinal_grid:
    dz_mm: 0.25
    no_cut_span_sigma: 8.0
```

### `charge_nC`

Total bunch charge in nC.

The code always normalizes the input density shape first so that:

```text
integral rho(z) dz = 1
```

and then multiplies by the physical bunch charge.

That means:

- for Gaussian input, the Gaussian shape is normalized
- for arbitrary sampled input, the interpolated curve is normalized

### `density.kind`

Allowed values:

- `gaussian`
- `array`

### Gaussian mode

Use:

```yaml
density:
  kind: gaussian
  sigma_mm: 62.0
  cutoff_sigma: 0.0
```

Meaning:

- `sigma_mm`: RMS bunch length in mm
- `cutoff_sigma`: truncation at `+- n sigma`

Rules:

- if `cutoff_sigma > 0`, the distribution is cut at `+- cutoff_sigma * sigma_mm`
- if `cutoff_sigma == 0`, the code does not use a hard physical cut; it uses a large numerical window controlled by `no_cut_span_sigma`

### Arbitrary sampled mode

This mode is for user-defined longitudinal shape.

Preferred compact form:

```yaml
density:
  kind: array
  samples: "{{-120,0.15},{-60,0.8},{0,1},{60,0.8},{120,0.15}}"
  interpolation_order: 5
```

This means:

- each pair is `{z_mm, peakcurrent}`
- `z_mm` is longitudinal position in mm
- `peakcurrent` is a relative amplitude, not an absolute current

The code accepts three formats:

1. compact string form:

```yaml
samples: "{{-120,0.15},{-60,0.8},{0,1},{60,0.8},{120,0.15}}"
```

2. compact YAML list form:

```yaml
samples: [[-120, 0.15], [-60, 0.8], [0, 1], [60, 0.8], [120, 0.15]]
```

3. verbose map form:

```yaml
samples:
  - {z_mm: -120, peakcurrent: 0.15}
  - {z_mm: -60, peakcurrent: 0.8}
  - {z_mm: 0, peakcurrent: 1}
  - {z_mm: 60, peakcurrent: 0.8}
  - {z_mm: 120, peakcurrent: 0.15}
```

### `interpolation_order`

This controls smoothing between sample points.

Requested behavior:

- the code tries to use spline interpolation up to 5th order

Practical rule:

- if you provide 6 or more sample points, `interpolation_order: 5` gives 5th-order spline interpolation
- if you provide fewer than 6 points, exact 5th-order interpolation is not mathematically available, so the code automatically uses the highest possible order `<= 5`

So with your 5-point example:

```yaml
samples: "{{-120,0.15},{-60,0.8},{0,1},{60,0.8},{120,0.15}}"
```

the code uses 4th-order interpolation, because 5 points are not enough for a true 5th-order spline.

### `longitudinal_grid`

Example:

```yaml
longitudinal_grid:
  dz_mm: 0.25
  no_cut_span_sigma: 8.0
```

Meaning:

- `dz_mm`: numerical sampling step along `z`
- `no_cut_span_sigma`: only used for uncapped Gaussian mode; the numerical window becomes approximately `+- no_cut_span_sigma * sigma`

Smaller `dz_mm`:

- improves resolution
- increases runtime

## 2.3 `chamber`: vacuum chamber cross-section

This defines the 2D transverse conductor boundary used by the electrostatic BPM solve.

### Physical meaning

The chamber block describes the inside vacuum aperture seen by the beam in cross-section.

- `x` is horizontal
- `y` is vertical
- units are mm

This geometry controls:

- field distortion
- image charge distribution
- BPM linearity
- fitted scale factors `Kx`, `Ky`

### `kind`

Allowed values:

- `round`
- `ellipse`
- `polygon`

### Round chamber

Example:

```yaml
chamber:
  kind: round
  radius_mm: 47.0
```

Meaning:

- the beam pipe inner boundary is a circle of radius `47 mm`

Use when the chamber is approximately cylindrical.

### Elliptical chamber

Example:

```yaml
chamber:
  kind: ellipse
  a_mm: 40.0
  b_mm: 20.0
```

Meaning:

- `a_mm`: horizontal semi-axis
- `b_mm`: vertical semi-axis

The ellipse equation is:

```text
(x/a)^2 + (y/b)^2 = 1
```

Use when the beam pipe is smoothly wider in one plane than the other.

### Polygon chamber

Example:

```yaml
chamber:
  kind: polygon
  points_mm:
    - [-10.0, 20.0]
    - [10.0, 20.0]
    - [40.0, 8.0]
    - [40.0, -8.0]
    - [10.0, -20.0]
    - [-10.0, -20.0]
    - [-40.0, -8.0]
    - [-40.0, 8.0]
```

Meaning:

- each entry is one chamber corner point `[x_mm, y_mm]`
- points must follow the boundary in order around the chamber
- the code automatically closes the polygon by connecting the last point back to the first

Use polygon mode when:

- the chamber has flats, chamfers, octagon-like shape, racetrack-like approximation, or other non-elliptic boundary

### `boundary_elements`

Example:

```yaml
boundary_elements: 320
```

Meaning:

- number of boundary segments used by the BEM discretization

Tradeoff:

- larger value: more accurate, slower
- smaller value: faster, less accurate

Typical values:

- `200` to `500` for routine runs

### `quadrature_order`

Example:

```yaml
quadrature_order: 8
```

Meaning:

- integration order used when computing segment-to-segment Green-function influence integrals

Usually you can leave this unchanged.

## 2.4 `buttons`: pickup electrode definition

This block defines how the four BPM pickups are placed relative to the chamber boundary.

Example:

```yaml
buttons:
  layout: cardinal
  radius_mm: 9.0
  thickness_mm: 2.0
  gap_mm: 0.3
  capacitance_pf: 3.4
  pickups:
    - label: T
      center_mm: [0.0, 49.0]
      color: tab:orange
    - label: R
      center_mm: [49.0, 0.0]
      color: tab:green
    - label: B
      center_mm: [0.0, -49.0]
      color: tab:red
    - label: L
      center_mm: [-49.0, 0.0]
      color: tab:purple
```

### Physical meaning of button parameters

#### `radius_mm`

Button electrode radius.

Used in two places:

1. longitudinal signal model, where it sets the width kernel `w(z)`
2. boundary plot/button coverage assignment, where it helps determine which boundary elements belong to each button

#### `thickness_mm`

Effective button thickness used in the capacitance estimate when `capacitance_pf` is not given.

This is not the chamber wall thickness. It is the effective pickup thickness used in the simple capacitance formula.

#### `gap_mm`

Gap between button edge and surrounding wall, used only when the code estimates button capacitance from geometry.

Smaller gap usually means larger capacitance.

#### `capacitance_pf`

If you know the button capacitance from design, measurement, or paper values, put it here.

When this value is present, the code uses it directly and ignores the geometric estimate from `thickness_mm` and `gap_mm`.

That is usually the better option.

### `pickups`

This is the list of four BPM buttons.

The required labels depend on `buttons.layout`.

For `buttons.layout: corners`:

- `A`
- `B`
- `C`
- `D`

with the convention:

- `A`: upper-left
- `B`: upper-right
- `C`: lower-right
- `D`: lower-left

For `buttons.layout: cardinal`:

- `T`
- `R`
- `B`
- `L`

with the convention:

- `T`: top
- `R`: right
- `B`: bottom
- `L`: left

#### `center_mm`

This is the center point of the physical button in chamber coordinates.

It does **not** need to lie exactly on a mesh node. The code finds boundary elements close to that point and assigns them to the button.

For a symmetric four-button BPM:

- top buttons have positive `y`
- bottom buttons have negative `y`
- right buttons have positive `x`
- left buttons have negative `x`

For a round chamber of radius `50 mm`, placing the cardinal button centers at
`[0,49]`, `[49,0]`, `[0,-49]`, `[-49,0]` is reasonable and supported.

#### `color`

Only affects plotting.

It is used to draw thicker colored boundary segments for each button in the generated figures.

## 2.5 `filter`: electronics and resolution settings

Example:

```yaml
filter:
  characteristic_impedance_ohm: 50.0
  cable:
    enabled: true
    attenuation_fc_hz: 4.65e8
    skin_effect_phase: true
  analog:
    type: lowpass_butter
    order: 4
    cutoff_hz: 1.3e8
  resolution:
    relative_error_min: 1.0e-4
    relative_error_max: 1.0e-2
    num_points: 200
    reference_relative_error: 1.6e-3
```

Optional BAR-style signal normalization:

```yaml
signal_model:
  image_charge_denominator_mm: 125.6637061436
```

This value is the denominator `P` in Eq. (1.2). Use `2*pi*b` to match the BAR/SLAC longitudinal button formula, or omit the block to use the chamber perimeter fallback.

### `characteristic_impedance_ohm`

Cable / electronics reference impedance, normally `50`.

### `cable`

#### `enabled`

Turn cable attenuation on or off.

#### `attenuation_fc_hz`

Parameter `f_c` in:

```text
|H_cable(f)| = exp(-sqrt(f / f_c))
```

This controls how fast high frequency content is attenuated.

For the BAR note's 50 m LMR240 cable, the quoted loss is 18 dB/100 m at 500 MHz. For 50 m that is 9 dB at 500 MHz, which gives `attenuation_fc_hz` close to `4.65e8`.

#### `skin_effect_phase`

When `true`, the code uses the full complex skin-effect response:

```text
H_cable(f) = exp(-(1 + i) sqrt(f / f_c))
```

Keep this `true` to reproduce BAR Fig. 5, where the cable output signal arrives later than the input signal.

### `analog`

Allowed filter types:

- `none`
- `lowpass_butter`
- `lowpass_bessel`
- `bandpass_butter`

#### Low-pass example

```yaml
analog:
  type: lowpass_butter
  order: 4
  cutoff_hz: 1.3e8
```

Use `lowpass_bessel` when approximating the low-pass Bessel-style EIC/RHIC prototype filter discussed in the BAR note:

```yaml
analog:
  type: lowpass_bessel
  order: 4
  cutoff_hz: 7.0e7
```

#### Band-pass example

```yaml
analog:
  type: bandpass_butter
  order: 2
  center_hz: 5.0e8
  bandwidth_hz: 2.0e7
```

### `resolution`

This block only affects the resolution plot and the summary values.

- `relative_error_min`, `relative_error_max`: x-axis range for Fig. 13-style plot
- `num_points`: number of plotted points
- `reference_relative_error`: one chosen operating point used in the report summary

## 2.6 `beam_grid`: transverse particle distribution

This block defines the rectangular cloud of test beam positions used for Fig. 11 and Fig. 12.

Example:

```yaml
beam_grid:
  x_half_size_mm: 15.0
  y_half_size_mm: 8.0
  nx: 61
  ny: 41
  linear_fit_half_range_mm: 5.0
  polynomial_order: 5
```

### Physical meaning

The user request specifies that the initial particles should always be distributed on a rectangular region. That is what this block does.

The code:

1. creates a rectangular grid from `-x_half_size_mm` to `+x_half_size_mm`
2. creates a rectangular grid from `-y_half_size_mm` to `+y_half_size_mm`
3. keeps only points that are inside the chamber

### Parameters

#### `x_half_size_mm`, `y_half_size_mm`

Half-width and half-height of the blue input distribution.

So the full rectangle is:

- width `= 2 * x_half_size_mm`
- height `= 2 * y_half_size_mm`

#### `nx`, `ny`

Number of grid points in horizontal and vertical direction.

The total candidate points before chamber clipping are roughly:

```text
nx * ny
```

#### `linear_fit_half_range_mm`

Half-range around the origin used to fit `Kx` and `Ky`.

Only near-center points are used for the linear scale-factor fit.

#### `polynomial_order`

Polynomial order used for the nonlinear calibration map from measured coordinates back to true coordinates.

Typical value: `5`

## 2.7 `output`

Example:

```yaml
output:
  directory: outputs/my_case
```

This is where figures and the report are written.

If the path is relative, it is resolved relative to the YAML file location.

## 2.8 Minimal YAML examples

### Minimal Gaussian ellipse case

```yaml
bunch:
  charge_nC: 1.0
  density:
    kind: gaussian
    sigma_mm: 62.0
    cutoff_sigma: 0.0
  longitudinal_grid:
    dz_mm: 0.25
    no_cut_span_sigma: 8.0

chamber:
  kind: ellipse
  a_mm: 40.0
  b_mm: 20.0
  boundary_elements: 320
  quadrature_order: 8

buttons:
  radius_mm: 9.0
  thickness_mm: 2.0
  gap_mm: 0.3
  capacitance_pf: 3.4
  pickups:
    - {label: A, center_mm: [-16, 16], color: tab:orange}
    - {label: B, center_mm: [16, 16], color: tab:green}
    - {label: C, center_mm: [16, -16], color: tab:red}
    - {label: D, center_mm: [-16, -16], color: tab:purple}

filter:
  characteristic_impedance_ohm: 50.0
  cable: {enabled: true, attenuation_fc_hz: 4.65e8}
  analog: {type: lowpass_butter, order: 4, cutoff_hz: 1.3e8}
  resolution:
    relative_error_min: 1e-4
    relative_error_max: 1e-2
    num_points: 200
    reference_relative_error: 1.6e-3

beam_grid:
  x_half_size_mm: 10.0
  y_half_size_mm: 6.0
  nx: 51
  ny: 31
  linear_fit_half_range_mm: 4.0
  polynomial_order: 5

output:
  directory: outputs/ellipse_case
```

### Minimal arbitrary-density polygon case

```yaml
bunch:
  charge_nC: 1.0
  density:
    kind: array
    samples: "{{-120,0.15},{-60,0.8},{0,1},{60,0.8},{120,0.15}}"
    interpolation_order: 5
  longitudinal_grid:
    dz_mm: 0.25

chamber:
  kind: polygon
  points_mm:
    - [-10, 20]
    - [10, 20]
    - [40, 8]
    - [40, -8]
    - [10, -20]
    - [-10, -20]
    - [-40, -8]
    - [-40, 8]
  boundary_elements: 320
  quadrature_order: 8

buttons:
  radius_mm: 9.0
  thickness_mm: 2.0
  gap_mm: 0.3
  capacitance_pf: 3.4
  pickups:
    - {label: A, center_mm: [-19.75, 16.1], color: tab:orange}
    - {label: B, center_mm: [19.75, 16.1], color: tab:green}
    - {label: C, center_mm: [19.75, -16.1], color: tab:red}
    - {label: D, center_mm: [-19.75, -16.1], color: tab:purple}

filter:
  characteristic_impedance_ohm: 50.0
  cable: {enabled: true, attenuation_fc_hz: 4.65e8}
  analog: {type: bandpass_butter, order: 2, center_hz: 5e8, bandwidth_hz: 2e7}
  resolution:
    relative_error_min: 1e-4
    relative_error_max: 1e-2
    num_points: 200
    reference_relative_error: 1.6e-3

beam_grid:
  x_half_size_mm: 15.0
  y_half_size_mm: 8.0
  nx: 61
  ny: 41
  linear_fit_half_range_mm: 5.0
  polynomial_order: 5

output:
  directory: outputs/polygon_case
```

## 2.9 Practical advice when writing a YAML file

1. Start from [input_template.yaml](/Users/wange/Coding/Python/bpm/input_template.yaml).
2. Set the chamber first, because that determines everything else.
3. Place the four button centers consistently with labels `A, B, C, D`.
4. Use `capacitance_pf` directly if you know it from design or measurement.
5. Use `gaussian` density first to verify geometry and scaling.
6. Switch to `array` density only after the geometry is working.
7. If the arbitrary density shape looks too sharp, add more sample points instead of only increasing interpolation order.
8. If BEM runtime is too slow, reduce `boundary_elements` or `nx`, `ny`.

## 2.10 Current limitation

The sample BAR polygon in this repository is not loaded from the missing `BeampositionMonitor.gdf`. It was inferred from the BAR note figure. If you provide the exact boundary coordinates from the GDF file, replace `chamber.points_mm` with those exact points and rerun.
